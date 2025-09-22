"""
HuggingFace Whisper Tokenizer Implementation
Replaces the original tiktoken-based tokenizer with HuggingFace's WhisperTokenizer
"""

import base64
import os
import string
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Dict, List, Optional, Tuple, Union

# Use HuggingFace tokenizer instead of tiktoken
from transformers import WhisperTokenizer

LANGUAGES = {
    "en": "english",
    "zh": "chinese", 
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}


@dataclass
class Tokenizer:
    """A thin wrapper around HuggingFace WhisperTokenizer providing quick access to special tokens"""

    tokenizer: WhisperTokenizer
    num_languages: int
    language: Optional[str] = None
    task: Optional[str] = None
    sot_sequence: Tuple[int] = ()
    special_tokens: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        # Get special tokens from HuggingFace tokenizer
        self.special_tokens = {
            "<|startoftranscript|>": self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
            "<|endoftext|>": self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            "<|notimestamps|>": self.tokenizer.convert_tokens_to_ids("<|notimestamps|>"),
            "<|transcribe|>": self.tokenizer.convert_tokens_to_ids("<|transcribe|>"),
            "<|translate|>": self.tokenizer.convert_tokens_to_ids("<|translate|>"),
            "<|nocaptions|>": self.tokenizer.convert_tokens_to_ids("<|nocaptions|>"),
            "<|startoflm|>": self.tokenizer.convert_tokens_to_ids("<|startoflm|>"),
            "<|startofprev|>": self.tokenizer.convert_tokens_to_ids("<|startofprev|>"),
        }
        
        # Add language tokens
        for lang in LANGUAGES.keys():
            lang_token = f"<|{lang}|>"
            try:
                self.special_tokens[lang_token] = self.tokenizer.convert_tokens_to_ids(lang_token)
            except:
                pass  # Some language tokens might not exist

        sot: int = self.special_tokens["<|startoftranscript|>"]
        translate: int = self.special_tokens["<|translate|>"]
        transcribe: int = self.special_tokens["<|transcribe|>"]
        notimestamps: int = self.special_tokens["<|notimestamps|>"]
        
        langs = tuple(LANGUAGES.keys())[: self.num_languages]
        sot_sequence = [sot]
        if self.language is not None:
            sot_sequence.append(sot + 1 + langs.index(self.language))
        if self.task is not None:
            task_token: int = transcribe if self.task == "transcribe" else translate
            sot_sequence.append(task_token)

        self.sot_sequence = tuple(sot_sequence)

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        token_ids = [t for t in token_ids if t < self.timestamp_begin]
        return self.tokenizer.decode(token_ids, **kwargs)

    def decode_with_timestamps(self, token_ids: List[int], **kwargs) -> str:
        """
        Timestamp tokens are above other special tokens' id range and are ignored by `decode()`.
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        return self.tokenizer.decode(token_ids, **kwargs)

    @cached_property
    def eot(self) -> int:
        return self.special_tokens["<|endoftext|>"]

    @cached_property
    def transcribe(self) -> int:
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self) -> int:
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def sot_lm(self) -> int:
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self) -> int:
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def no_speech(self) -> int:
        return self.special_tokens["<|nocaptions|>"]

    @cached_property
    def no_timestamps(self) -> int:
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self) -> int:
        return self.special_tokens.get("<|0.00|>", 50364)  # Fallback to notimestamps

    @cached_property
    def language_token(self) -> int:
        """Returns the token id corresponding to the value of the `language` field"""
        if self.language is None:
            raise ValueError("This tokenizer does not have language token configured")

        return self.to_language_token(self.language)

    def to_language_token(self, language):
        if token := self.special_tokens.get(f"<|{language}|>", None):
            return token

        raise KeyError(f"Language {language} not found in tokenizer.")

    @cached_property
    def all_language_tokens(self) -> Tuple[int]:
        result = []
        for token, token_id in self.special_tokens.items():
            if token.strip("<>").startswith("|") and token.strip("<>").endswith("|"):
                lang = token.strip("<>").strip("|")
                if lang in LANGUAGES:
                    result.append(token_id)
        return tuple(sorted(result))

    @cached_property
    def all_language_codes(self) -> Tuple[str]:
        """All language codes for this tokenizer"""
        return tuple(LANGUAGES.keys())

    @cached_property
    def all_special_tokens(self) -> Tuple[int]:
        """All special tokens for this tokenizer"""
        return tuple(self.special_tokens.values())

    @cached_property
    def non_speech_tokens(self) -> Tuple[int]:
        """All the non-speech tokens"""
        return (
            self.sot_sequence
            + (self.no_speech,)
            + self.all_language_tokens
            + (self.no_timestamps,)
            + (self.sot_lm,)
            + (self.sot_prev,)
        )

    @cached_property
    def multilingual(self) -> bool:
        """Whether this tokenizer supports multiple languages"""
        return self.num_languages > 1


@lru_cache(maxsize=None)
def get_encoding(name: str = "gpt2", num_languages: int = 99):
    """Get HuggingFace WhisperTokenizer encoding"""
    # Load HuggingFace tokenizer
    model_name = "openai/whisper-tiny"  # Can be changed to other Whisper models
    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    
    # Create our wrapper
    return Tokenizer(
        tokenizer=tokenizer,
        num_languages=num_languages,
        language=None,
        task=None,
    )


def get_tokenizer(multilingual: bool = True, language: Optional[str] = None, task: str = "transcribe") -> Tokenizer:
    """Get HuggingFace Whisper tokenizer instance"""
    # Load HuggingFace tokenizer
    model_name = "openai/whisper-tiny"  # Can be changed to other Whisper models
    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    
    # Set language and task
    if language:
        tokenizer.set_prefix_tokens(language=language, task=task)
    
    # Create our wrapper
    return Tokenizer(
        tokenizer=tokenizer,
        num_languages=99 if multilingual else 1,
        language=language,
        task=task,
    )


def decode_with_timestamps(tokenizer: Tokenizer, token_ids: List[int]) -> List[Dict[str, Union[str, float]]]:
    """Decode token IDs to text with timestamps"""
    # This is a simplified implementation
    # In practice, you'd need to parse timestamp tokens
    text = tokenizer.decode(token_ids)
    return [{"text": text, "timestamp": 0.0}]


def detect_language(tokenizer: Tokenizer, token_ids: List[int]) -> Tuple[str, List[float]]:
    """Detect the language from the given tokens"""
    # This is a simplified implementation
    # In practice, you'd need to analyze the tokens to detect language
    return "en", [0.0]


# Test function
def test_hf_tokenizer():
    """Test the HuggingFace tokenizer"""
    print("🔍 TESTING HUGGINGFACE WHISPER TOKENIZER")
    print("=" * 50)
    
    # Test multilingual tokenizer
    tokenizer = get_tokenizer(multilingual=True)
    
    # Test basic encoding/decoding
    test_texts = [
        "Hello world!",
        "This is a test.",
        "Ladies and gentlemen, this is a difficult moment for America.",
        "Bonjour le monde!",  # French
        "Hola mundo!",  # Spanish
    ]
    
    print(f"\n📝 Basic encoding/decoding test:")
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"   '{text}' -> {tokens} -> '{decoded}'")
    
    # Test special tokens
    print(f"\n🔍 Special tokens test:")
    print(f"   SOT: {tokenizer.sot}")
    print(f"   EOT: {tokenizer.eot}")
    print(f"   Transcribe: {tokenizer.transcribe}")
    print(f"   Translate: {tokenizer.translate}")
    print(f"   No timestamps: {tokenizer.no_timestamps}")
    print(f"   No speech: {tokenizer.no_speech}")
    
    # Test language tokens
    print(f"\n🌍 Language tokens test:")
    for lang in ["en", "fr", "es", "de"]:
        try:
            lang_id = tokenizer.to_language_token(lang)
            print(f"   {lang}: {lang_id}")
        except Exception as e:
            print(f"   {lang}: ERROR - {e}")
    
    print(f"\n✅ HF tokenizer test completed!")

if __name__ == "__main__":
    test_hf_tokenizer()
