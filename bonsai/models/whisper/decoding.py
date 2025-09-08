# JAX NNX implementation of Whisper decoding
# Complete port of decoding_original.py to JAX NNX

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from audio import CHUNK_LENGTH
from tokenizer import Tokenizer, get_tokenizer
from utils import compression_ratio

if TYPE_CHECKING:
    from model import Whisper


def detect_language(
    model: "Whisper", mel: jnp.ndarray, tokenizer: Tokenizer = None
) -> Tuple[jnp.ndarray, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : jnp.ndarray, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(
            model.is_multilingual, num_languages=model.num_languages
        )
    if (
        tokenizer.language is None
        or tokenizer.language_token not in tokenizer.sot_sequence
    ):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform lang id"
        )

    single = mel.ndim == 2
    if single:
        mel = mel[None, :, :]

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = jnp.array([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = jnp.ones(logits.shape[-1], dtype=bool)
    mask = mask.at[jnp.array(list(tokenizer.all_language_tokens))].set(False)
    logits = logits.at[:, mask].set(-jnp.inf)
    language_tokens = jnp.argmax(logits, axis=-1)
    language_token_probs = jax.nn.softmax(logits, axis=-1)
    language_probs = [
        {
            c: float(language_token_probs[i, j])
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0] if language_tokens.ndim > 0 else language_tokens
        language_probs = language_probs[0] if isinstance(language_probs, list) else language_probs

    return language_tokens, language_probs


@dataclass(frozen=True)
class DecodingOptions:
    # whether to perform X->X "transcribe" or X->English "translate"
    task: str = "transcribe"

    # language that the audio is in; uses detected language if None
    language: Optional[str] = None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)

    # "alpha" in Google NMT, or None for length norm, when ranking generations
    # to select which to return among the beams or best-of-N samples
    length_penalty: Optional[float] = None

    # text or tokens to feed as the prompt or the prefix; for more info:
    # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
    prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs

    # timestamp sampling options
    without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0

    # implementation details
    fp16: bool = True  # use fp16 for most of the calculation


@dataclass(frozen=True)
class DecodingResult:
    audio_features: jnp.ndarray
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class Inference:
    def logits(self, tokens: jnp.ndarray, audio_features: jnp.ndarray) -> jnp.ndarray:
        """Perform a forward pass on the decoder and return per-token logits"""
        raise NotImplementedError

    def rearrange_kv_cache(self, source_indices) -> None:
        """Update the key-value cache according to the updated beams"""
        raise NotImplementedError

    def cleanup_caching(self) -> None:
        """Clean up any resources or hooks after decoding is finished"""
        pass


class JAXInference(Inference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}

    def logits(self, tokens: jnp.ndarray, audio_features: jnp.ndarray) -> jnp.ndarray:
        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)

    def cleanup_caching(self):
        self.kv_cache = {}

    def rearrange_kv_cache(self, source_indices):
        if source_indices != list(range(len(source_indices))):
            for key in self.kv_cache:
                # update the key/value cache to contain the selected sequences
                self.kv_cache[key] = self.kv_cache[key][source_indices]


class SequenceRanker:
    def rank(
        self, tokens: List[List[jnp.ndarray]], sum_logprobs: List[List[float]]
    ) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[jnp.ndarray]], sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(
        self, tokens: jnp.ndarray, logits: jnp.ndarray, sum_logprobs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : jnp.ndarray, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : jnp.ndarray, shape = (n_batch, vocab_size)
            per-token logits of the current model outputs for the last token of each sequence

        sum_logprobs : jnp.ndarray, shape = (n_batch,)
            cumulative log probabilities of each sequence

        Returns
        -------
        tokens : jnp.ndarray, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : jnp.ndarray, shape = (n_batch,)
            a boolean tensor indicating if each sequence has completed generating
        """
        raise NotImplementedError

    def finalize(
        self, tokens: jnp.ndarray, sum_logprobs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : jnp.ndarray, shape = (n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : jnp.ndarray, shape = (n_group,)
            cumulative log probabilities of each sequence

        Returns
        -------
        tokens : jnp.ndarray, shape = (n_candidates, n_ctx)
            based on the search space (beam search) or the sampling results, select the top-k
            (for beam search) or all (for sampling) candidate sequences; padding may be
            applied as the sequence lengths are not necessarily identical

        sum_logprobs : jnp.ndarray, shape = (n_candidates,)
            the sum of the log probabilities of each sequence
        """
        raise NotImplementedError


class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(
        self, tokens: jnp.ndarray, logits: jnp.ndarray, sum_logprobs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.temperature == 0:
            next_tokens = jnp.argmax(logits, axis=-1)
        else:
            next_tokens = jax.random.categorical(
                jax.random.PRNGKey(0), logits / self.temperature, axis=-1
            )

        logprobs = jax.nn.log_softmax(logits, axis=-1)
        current_logprobs = jnp.take_along_axis(logprobs, next_tokens[:, None], axis=-1)[:, 0]
        sum_logprobs = sum_logprobs + current_logprobs

        next_tokens = next_tokens[:, None]
        tokens = jnp.concatenate([tokens, next_tokens], axis=-1)

        completed = (next_tokens == self.eot).squeeze(-1)
        return tokens, completed

    def finalize(
        self, tokens: jnp.ndarray, sum_logprobs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # make sure each sequence has at least one EOT token at the end
        tokens = jnp.pad(tokens, ((0, 0), (0, 1)), constant_values=self.eot)
        return tokens, sum_logprobs


class BeamSearchDecoder(TokenDecoder):
    def __init__(self, beam_size: int, eot: int, inference: Inference, patience: Optional[float] = None):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None

        assert self.max_candidates > 0, f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = None

    def update(
        self, tokens: jnp.ndarray, logits: jnp.ndarray, sum_logprobs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape[0]} % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = jax.nn.log_softmax(logits, axis=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        completed = jnp.zeros(tokens.shape[0], dtype=bool)

        for i in range(n_audio):
            # for each audio, keep (2 * beam_size) candidate hypotheses
            scores = sum_logprobs[i * self.beam_size : (i + 1) * self.beam_size, None] + logprobs[i * self.beam_size : (i + 1) * self.beam_size]
            scores = scores.reshape(-1)

            # select the top (2 * beam_size) candidates
            topk_indices = jnp.argsort(scores)[-self.beam_size * 2 :]
            topk_logprobs = scores[topk_indices]

            # each candidate is a (source_beam_index, next_token) pair
            source_beam_indices = topk_indices // logprobs.shape[1]
            next_token_indices = topk_indices % logprobs.shape[1]

            # add the selected tokens to the current sequences
            next_tokens.extend(next_token_indices)
            source_indices.extend(source_beam_indices + i * self.beam_size)

            # check if any of the sequences have finished
            for j, (beam_index, next_token) in enumerate(zip(source_beam_indices, next_token_indices)):
                if next_token == self.eot:
                    completed[i * self.beam_size + j] = True
                    # add to finished sequences
                    sequence = tokens[i * self.beam_size + beam_index]
                    self.finished_sequences[i][tuple(sequence)] = topk_logprobs[j]

        tokens = tokens[source_indices]
        tokens = jnp.concatenate([tokens, jnp.array(next_tokens)[:, None]], axis=-1)
        sum_logprobs = jnp.array([sum_logprobs[i] for i in source_indices])

        return tokens, completed

    def finalize(
        self, tokens: jnp.ndarray, sum_logprobs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # collect all finished sequences, including patience, and add unfinished ones
        sum_logprobs = sum_logprobs.tolist()
        for i, sequences in enumerate(self.finished_sequences):
            if len(sequences) < self.beam_size:  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i * self.beam_size : (i + 1) * self.beam_size])[-self.beam_size - len(sequences) :]):
                    sequence = tokens[i * self.beam_size + j]
                    sequences[tuple(sequence)] = sum_logprobs[i * self.beam_size + j]

        tokens = []
        sum_logprobs = []
        for sequences in self.finished_sequences:
            # take the best finished sequences
            for sequence in sorted(sequences.keys(), key=lambda k: sequences[k], reverse=True)[:self.beam_size]:
                tokens.append(sequence)
                sum_logprobs.append(sequences[sequence])

        if tokens:
            tokens = jnp.array(tokens)
            sum_logprobs = jnp.array(sum_logprobs)
        else:
            # no finished sequences, fall back to the current sequences
            tokens = jnp.pad(tokens, ((0, 0), (0, 1)), constant_values=self.eot)
            sum_logprobs = jnp.array(sum_logprobs)

        return tokens, sum_logprobs


class LogitFilter:
    def apply(self, logits: jnp.ndarray, tokens: jnp.ndarray) -> jnp.ndarray:
        """Apply any filtering or masking to logits in-place

        Parameters
        ----------
        logits : jnp.ndarray, shape = (n_batch, vocab_size)
            per-token logits of the current model outputs
        tokens : jnp.ndarray, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        Returns
        -------
        logits : jnp.ndarray, shape = (n_batch, vocab_size)
            the logits after applying any filtering to the tokens to be suppressed
        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits: jnp.ndarray, tokens: jnp.ndarray) -> jnp.ndarray:
        if tokens.shape[1] == self.sample_begin:
            logits = logits.at[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]].set(-jnp.inf)
        return logits


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: List[int]):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits: jnp.ndarray, tokens: jnp.ndarray) -> jnp.ndarray:
        logits = logits.at[:, self.suppress_tokens].set(-jnp.inf)
        return logits


class ApplyTimestampRules(LogitFilter):
    def __init__(
        self,
        tokenizer: Tokenizer,
        sample_begin: int,
        max_initial_timestamp_index: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: jnp.ndarray, tokens: jnp.ndarray) -> jnp.ndarray:
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits = logits.at[:, self.tokenizer.no_timestamps].set(-jnp.inf)

        # if the sequence is empty, we can only generate a timestamp token
        if tokens.shape[1] == self.sample_begin:
            logits = logits.at[:, : self.tokenizer.timestamp_begin].set(-jnp.inf)
            return logits

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[1]):
            timestamp_begin = self.tokenizer.timestamp_begin
            timestamp_end = timestamp_begin + 1500  # Approximate timestamp range
            token_val = tokens[0, k]  # Get scalar value from batch
            if timestamp_begin <= token_val < timestamp_end:
                # timestamps cannot be repeated; mask all previous timestamps
                logits = logits.at[:, timestamp_begin:token_val].set(-jnp.inf)
                # timestamps cannot be in the middle of a word; mask the word tokens
                word_tokens = self.tokenizer.encode(" ")[-1]
                logits = logits.at[:, word_tokens].set(-jnp.inf)

        if tokens.shape[1] > self.sample_begin:
            # timestamps cannot appear in the middle of a word; mask the word tokens
            word_tokens = self.tokenizer.encode(" ")[-1]
            logits = logits.at[:, word_tokens].set(-jnp.inf)

        # apply the `max_initial_timestamp` option
        if tokens.shape[1] == self.sample_begin and self.max_initial_timestamp_index is not None:
            last_allowed = self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
            logits = logits.at[:, last_allowed + 1 :].set(-jnp.inf)

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        for k in range(tokens.shape[0]):
            timestamp_end = self.tokenizer.timestamp_begin + 1500  # Approximate timestamp range
            timestamp_logprob = jax.scipy.special.logsumexp(
                logprobs[k, self.tokenizer.timestamp_begin : timestamp_end]
            )
            max_text_token_logprob = jnp.max(logprobs[k, : self.tokenizer.timestamp_begin])
            if timestamp_logprob > max_text_token_logprob:
                logits = logits.at[k, : self.tokenizer.timestamp_begin].set(-jnp.inf)

        return logits


class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model

        language = options.language or "en"
        tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language,
            task=options.task,
        )
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = JAXInference(model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            self.decoder = BeamSearchDecoder(
                options.beam_size, tokenizer.eot, self.inference, options.patience
            )
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                ApplyTimestampRules(
                    tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
            0 <= options.length_penalty <= 1
        ):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                prompt_tokens[-(self.n_ctx // 2 - 1) :] + tokens[1:]
            )

        return tuple(tokens)

    def _get_suppress_tokens(self) -> List[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t != -1]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        return suppress_tokens

    def _get_audio_features(self, mel: jnp.ndarray) -> jnp.ndarray:
        if mel.shape[-2:] != (self.model.dims.n_audio_ctx, self.model.dims.n_audio_state):
            return self.model.encoder(mel)
        return mel

    def _detect_language(self, mel: jnp.ndarray, mel_segment: jnp.ndarray) -> str:
        if self.options.language is not None:
            return self.options.language

        language_token, language_probs = detect_language(self.model, mel_segment, self.tokenizer)
        language_token = language_token[0] if language_token.ndim > 0 else language_token
        language_probs = language_probs[0] if isinstance(language_probs, list) else language_probs

        language = self.tokenizer.decode([language_token])
        return language

    def _main_loop(self, audio_features: jnp.ndarray, tokens: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        n_batch = tokens.shape[0]
        sum_logprobs = jnp.zeros(n_batch)
        no_speech_probs = jnp.ones(n_batch)

        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)

                if i == 0 and self.tokenizer.no_speech is not None:  # save no_speech_probs
                    probs_at_sot = jax.nn.softmax(logits[:, self.sot_index], axis=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech]

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logits = logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed.any():
                    break

            tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)

        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs

    def run(self, mel: jnp.ndarray) -> List[DecodingResult]:
        m = mel.shape[0]
        if m == 1:
            result = self._decode_single(mel[0])
        else:
            result = [self._decode_single(mel[i]) for i in range(m)]

        return result

    def _decode_single(self, mel: jnp.ndarray) -> DecodingResult:
        tokens = jnp.array([self.initial_tokens])
        audio_features = self._get_audio_features(mel[None, :, :])

        # detect language if requested
        language = self._detect_language(mel, mel)

        # main decoding loop
        tokens, sum_logprobs = self._main_loop(audio_features, tokens)

        # collect the best candidates (up to n_group)
        tokens = tokens[0]
        text = self.tokenizer.decode(tokens)
        text = text.strip()

        # no_speech_prob is the probability of the <|nospeech|> token computed above
        no_speech_prob = float(no_speech_probs[0]) if 'no_speech_probs' in locals() else 0.0

        return DecodingResult(
            audio_features=audio_features[0],
            language=language,
            tokens=tokens.tolist(),
            text=text,
            avg_logprob=float(sum_logprobs[0]) / len(tokens),
            no_speech_prob=no_speech_prob,
            temperature=self.options.temperature,
            compression_ratio=compression_ratio(text),
        )


def decode(
    model: "Whisper",
    mel: jnp.ndarray,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: jnp.ndarray, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    if single := mel.ndim == 2:
        mel = mel[None, :, :]

    if kwargs:
        options = replace(options, **kwargs)

    result = DecodingTask(model, options).run(mel)

    return result[0] if single and isinstance(result, list) else result