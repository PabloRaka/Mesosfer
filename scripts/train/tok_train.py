"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
import os
import time
import argparse
import torch
from mesosfer.data.tokenizer import RustBPETokenizer, SPLIT_PATTERN
from mesosfer.utils.common import get_base_dir
from mesosfer.data.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Split-pattern presets for BPB sweeps. `num13` is the current default (\p{N}{1,3}).
# `num12` narrows number grouping to \p{N}{1,2}. Derived from SPLIT_PATTERN by a single
# substitution so the rest of the (long) regex never drifts out of sync.
SPLIT_PATTERN_PRESETS = {
    "num13": SPLIT_PATTERN,
    "num12": SPLIT_PATTERN.replace(r"\p{N}{1,3}", r"\p{N}{1,2}"),
}

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 2B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=98304, help='Vocabulary size (default: 98304 = 96 * 1024). Larger vocab improves EN+ID+code+cyber compression; note this architecture has untied wte/lm_head plus per-half-layer value embeddings, so vocab cost is amplified.')
parser.add_argument('--split-pattern-preset', choices=sorted(SPLIT_PATTERN_PRESETS.keys()), default='num13',
                    help='Pre-tokenization split-pattern preset (default: num13 = current \\p{N}{1,3}). num12 = \\p{N}{1,2}.')
parser.add_argument('--out-dir', type=str, default=None,
                    help='Output directory for the tokenizer (default: <base_dir>/tokenizer). Use a distinct dir per sweep candidate.')
args = parser.parse_args()
split_pattern = SPLIT_PATTERN_PRESETS[args.split_pattern_preset]
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")
print(f"split_pattern_preset: {args.split_pattern_preset}")

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size, pattern=split_pattern)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = args.out_dir if args.out_dir is not None else os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)
print(f"Saved tokenizer to {tokenizer_dir}")

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
from mesosfer.utils.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])
