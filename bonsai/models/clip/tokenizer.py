import os
from typing import List, Optional, Callable, Tuple
import jax.numpy as jnp

def simple_whitespace_tokenizer(texts: List[str], max_length: int = 77) -> Tuple[jnp.ndarray, dict]:
    vocab = {"<pad>": 0, "<unk>": 1}
    next_id = 2
    batch = []
    for t in texts:
        toks = t.strip().lower().split()
        ids = []
        for w in toks[:max_length]:
            if w not in vocab:
                vocab[w] = next_id
                next_id += 1
            ids.append(vocab[w])
        ids += [0] * (max_length - len(ids))
        batch.append(ids)
    import numpy as _np
    return jnp.array(_np.array(batch, dtype=_np.int32)), vocab

def load_tokenizer(tokenizer_path: Optional[str]) -> Optional[Callable[[List[str], int], jnp.ndarray]]:
    if tokenizer_path is None:
        return None
    try:
        from tokenizers import Tokenizer
        if os.path.isdir(tokenizer_path):
            for fname in ("tokenizer.json", "bpe.json", "vocab.json"):
                p = os.path.join(tokenizer_path, fname)
                if os.path.exists(p):
                    tokenizer = Tokenizer.from_file(p)
                    break
            else:
                tokenizer = None
        elif os.path.exists(tokenizer_path):
            tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            tokenizer = None
        if tokenizer is None:
            return None

        def encode_texts(texts: List[str], max_length: int = 77):
            encs = [tokenizer.encode(t).ids[:max_length] for t in texts]
            padded = [e + [0]*(max_length - len(e)) if len(e) < max_length else e for e in encs]
            import numpy as _np
            return jnp.array(_np.array(padded, dtype=_np.int32))
        return encode_texts
    except Exception:
        return None
