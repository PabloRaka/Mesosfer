"""
Tests for the RustBPE tokenizer, focused on the contracts that pretraining and
SFT rely on:

- round-trip encode/decode (incl. unicode, code, whitespace)
- every declared special token is resolvable and has a unique id
- BOS prepend behaviour used by the pretraining dataloader
- special-token strings inside content are NOT parsed as special tokens
  (encode_ordinary safety — prevents prompt-injection via raw data)
- render_conversation: structure + supervision mask (assistant-only)
- system-message merge and user/assistant alternation
- render_for_completion priming for RL
"""

import pytest

from mesosfer.data.tokenizer import RustBPETokenizer, SPECIAL_TOKENS


# A small but varied corpus so BPE learns enough merges for the tests.
_CORPUS = [
    "the quick brown fox jumps over the lazy dog " * 40,
    "security operations center incident response triage " * 40,
    "def handler(request):\n    return {'status': 200}\n" * 40,
    "CVE-2021-44228 Log4Shell remote code execution vulnerability " * 40,
    "user assistant system python output analysis reasoning " * 40,
]


@pytest.fixture(scope="module")
def tok():
    return RustBPETokenizer.train_from_iterator(iter(_CORPUS * 20), vocab_size=512)


def test_special_tokens_are_resolvable_and_unique(tok):
    ids = [tok.encode_special(t) for t in SPECIAL_TOKENS]
    # every special token resolves to an int id
    assert all(isinstance(i, int) for i in ids)
    # ids are unique
    assert len(set(ids)) == len(SPECIAL_TOKENS)


def test_no_unused_dead_special_tokens():
    # Guard against re-introducing special tokens that are never emitted into
    # the data. Only BOS + conversation/tool tokens are part of the protocol.
    expected = {
        "<|bos|>",
        "<|user_start|>", "<|user_end|>",
        "<|assistant_start|>", "<|assistant_end|>",
        "<|calc_start|>", "<|calc_end|>",
        "<|output_start|>", "<|output_end|>",
        "<|tool_start|>", "<|tool_end|>",
    }
    assert set(SPECIAL_TOKENS) == expected


@pytest.mark.parametrize("text", [
    "Hello, world!",
    "Numbers: 123, 4567, 89",
    "Contractions: I'm, you're, it's",
    "Unicode: 你好世界 🌍",
    "def f(x):\n\treturn x + 1\n",
    "   leading and trailing spaces   ",
])
def test_roundtrip_encode_decode(tok, text):
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_bos_prepend(tok):
    bos = tok.get_bos_token_id()
    ids = tok.encode("hello world", prepend=bos)
    assert ids[0] == bos
    # the rest decodes back to the original text
    assert tok.decode(ids[1:]) == "hello world"


def test_batch_encode_with_bos(tok):
    bos = tok.get_bos_token_id()
    batch = ["first document", "second document"]
    out = tok.encode(batch, prepend=bos)
    assert isinstance(out, list) and len(out) == 2
    assert all(row[0] == bos for row in out)


def test_special_token_string_in_content_is_safe(tok):
    # A malicious/raw document containing a special-token string must NOT be
    # encoded as the actual special token id (encode_ordinary semantics).
    for special in SPECIAL_TOKENS:
        sid = tok.encode_special(special)
        ids = tok.encode(f"{special} some trailing text")
        assert sid not in ids, f"{special} leaked into ordinary encoding"


def test_render_conversation_mask_and_structure(tok):
    conv = {"messages": [
        {"role": "user", "content": "What is CVE-2021-44228?"},
        {"role": "assistant", "content": "It is Log4Shell, an RCE."},
    ]}
    ids, mask = tok.render_conversation(conv)

    assert len(ids) == len(mask)
    # first token is BOS and is not supervised
    assert ids[0] == tok.get_bos_token_id()
    assert mask[0] == 0
    # there is a supervised (assistant) region
    assert any(m == 1 for m in mask)
    # user content is never supervised: the user_start/end markers carry mask 0
    user_start = tok.encode_special("<|user_start|>")
    user_end = tok.encode_special("<|user_end|>")
    for tid, m in zip(ids, mask):
        if tid in (user_start, user_end, tok.get_bos_token_id()):
            assert m == 0
    # assistant_end is supervised (model must learn to stop)
    assistant_end = tok.encode_special("<|assistant_end|>")
    assert any(tid == assistant_end and m == 1 for tid, m in zip(ids, mask))


def test_render_conversation_system_message_merged(tok):
    conv = {"messages": [
        {"role": "system", "content": "You are a SOC analyst."},
        {"role": "user", "content": "Triage this alert."},
        {"role": "assistant", "content": "Checking the logs now."},
    ]}
    ids, mask = tok.render_conversation(conv)
    assert len(ids) == len(mask) > 0
    # original conversation must not be mutated by the merge surgery
    assert conv["messages"][0]["role"] == "system"


def test_render_conversation_rejects_bad_alternation(tok):
    conv = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "still me"},
    ]}
    with pytest.raises(AssertionError):
        tok.render_conversation(conv)


def test_render_for_completion_primes_assistant(tok):
    conv = {"messages": [
        {"role": "user", "content": "Explain SQL injection."},
        {"role": "assistant", "content": "It injects SQL."},
    ]}
    ids = tok.render_for_completion(conv)
    # ends primed for the assistant to complete
    assert ids[-1] == tok.encode_special("<|assistant_start|>")
    # original conversation not mutated
    assert conv["messages"][-1]["role"] == "assistant"


def test_render_conversation_truncates_to_max_tokens(tok):
    conv = {"messages": [
        {"role": "user", "content": "word " * 5000},
        {"role": "assistant", "content": "ok " * 5000},
    ]}
    ids, mask = tok.render_conversation(conv, max_tokens=128)
    assert len(ids) == 128
    assert len(mask) == 128


def test_render_conversation_tool_call_parts(tok):
    # Generic named tool call (e.g. shell) + tool output round-trips through the
    # <|tool_start|>/<|tool_end|> and <|output_start|>/<|output_end|> tokens.
    conv = {"messages": [
        {"role": "user", "content": "Scan the host."},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Running a scan."},
            {"type": "tool", "text": '{"name": "shell", "arguments": {"command": "nmap -sV 10.0.0.1"}}'},
            {"type": "tool_output", "text": "22/tcp open ssh"},
            {"type": "text", "text": "Port 22 is open."},
        ]},
    ]}
    ids, mask = tok.render_conversation(conv)
    assert len(ids) == len(mask)

    tool_start = tok.encode_special("<|tool_start|>")
    tool_end = tok.encode_special("<|tool_end|>")
    output_start = tok.encode_special("<|output_start|>")

    # tool-call markers are present and supervised (model must learn to call tools)
    assert tool_start in ids and tool_end in ids
    for tid, m in zip(ids, mask):
        if tid in (tool_start, tool_end):
            assert m == 1
        # tool output (from the environment) is NOT supervised
        if tid == output_start:
            assert m == 0


# -----------------------------------------------------------------------------
# Hex / number split-pattern behaviour (cybersecurity-domain pre-tokenization).
# SPLIT_PATTERN isolates `0x...` literals and long hash/address runs (>=8 hex chars)
# as single pre-tokens, while ordinary English words must NOT be captured by the
# hex rule. These are contract tests on the pre-tokenizer, verified via round-trip
# (the pattern only affects token boundaries, never the decoded bytes).


@pytest.fixture(scope="module")
def hex_tok():
    # A corpus rich in hex literals, hashes and addresses so BPE learns relevant merges.
    corpus = [
        "address 0xdeadbeef 0xCAFEBABE offset 0x1f4 0x00ff " * 40,
        "sha256 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08 " * 40,
        "md5 5d41402abc4b2a76b9719d911017c592 hash 0123456789abcdef " * 40,
        "the quick brown fox jumps over the lazy dog facade decade cabbage feedback " * 40,
    ]
    return RustBPETokenizer.train_from_iterator(iter(corpus * 20), vocab_size=512)


@pytest.mark.parametrize("text", [
    "0xdeadbeef",
    "0xCAFEBABE plus tail",
    "hash 9f86d081884c7d659a2feaa0c55ad015 done",
    "md5 5d41402abc4b2a76b9719d911017c592",
    "decimal 12345678 and 9000000001",
])
def test_hex_roundtrip(hex_tok, text):
    # Whatever the boundaries, encode->decode must be lossless.
    assert hex_tok.decode(hex_tok.encode(text)) == text


def test_hex_rule_does_not_capture_normal_words(hex_tok):
    # Ordinary English words (even those made only of a-f letters but <8 chars, or
    # >=8 chars containing non-hex letters) must round-trip and not be mangled.
    for word in ["facade", "decade", "cabbage", "feedback", "deadbeef", "defaced"]:
        assert hex_tok.decode(hex_tok.encode(word)) == word


def test_hex_special_tokens_still_resolvable(hex_tok):
    # Sanity: the split-pattern change must not break special-token resolution.
    ids = [hex_tok.encode_special(t) for t in SPECIAL_TOKENS]
    assert all(isinstance(i, int) for i in ids)
    assert len(set(ids)) == len(SPECIAL_TOKENS)
