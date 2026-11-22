"""Unit tests for chunked softcapped cross-entropy and GQA model configuration."""

import torch
import torch.nn.functional as F
import pytest

from mesosfer.model.gpt import GPT, GPTConfig
from mesosfer.eval.engine import Engine


class _TinyTokenizer:
    _special = {
        "<|tool_start|>": 256, "<|tool_end|>": 257,
        "<|output_start|>": 258, "<|output_end|>": 259,
        "<|assistant_end|>": 260, "<|bos|>": 261,
    }
    def encode_special(self, s): return self._special[s]
    def get_bos_token_id(self): return 261
    def encode(self, s): return list(s.encode("utf-8"))
    def decode(self, ids): return bytes(t for t in ids if t < 256).decode("utf-8", errors="replace")


def _tiny_gpt(n_head=8, n_kv_head=2, vocab_size=262):
    cfg = GPTConfig(
        sequence_len=64,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=64,
        window_pattern="L",
        ve_layers=1,
    )
    model = GPT(cfg)
    model.init_weights()
    return model


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("with_ignore", [False, True])
def test_chunked_cross_entropy_matches_unchunked_fwd_and_bwd(reduction, with_ignore):
    torch.manual_seed(0)
    model = _tiny_gpt(n_head=4, n_kv_head=4, vocab_size=127)
    B, T, C = 2, 24, 64  # N = 48 tokens > chunk_size=16 (3 chunks)
    x_ref = torch.randn(B, T, C, requires_grad=True)
    x_chk = x_ref.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 127, (B, T))
    if with_ignore:
        # Mask an entire middle chunk to -1 plus partial tokens in other chunks
        targets[0, 16:] = -1
        targets[1, :8] = -1
        targets[0, 0] = -1

    # Reference unchunked
    model.zero_grad(set_to_none=True)
    loss_ref = model._chunked_cross_entropy(x_ref, targets, softcap=15, reduction=reduction, chunk_size=1024)
    (loss_ref.sum() if reduction == "none" else loss_ref).backward()
    gx_ref = x_ref.grad.detach().clone()
    gw_ref = model.lm_head.weight.grad.detach().clone()

    # Chunked (chunk_size=16 -> 3 chunks, middle chunk all -1 when with_ignore=True)
    model.zero_grad(set_to_none=True)
    loss_chk = model._chunked_cross_entropy(x_chk, targets, softcap=15, reduction=reduction, chunk_size=16)
    (loss_chk.sum() if reduction == "none" else loss_chk).backward()
    gx_chk = x_chk.grad.detach().clone()
    gw_chk = model.lm_head.weight.grad.detach().clone()

    assert loss_chk.shape == loss_ref.shape
    torch.testing.assert_close(loss_chk, loss_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gx_chk, gx_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gw_chk, gw_ref, atol=1e-5, rtol=1e-5)


def test_gqa_forward_backward_optimizer_and_kvcache(monkeypatch):
    import mesosfer.model.optim as optim_mod
    for fn_name in ("adamw_step_fused", "muon_step_fused"):
        fn = getattr(optim_mod, fn_name, None)
        if fn is not None and hasattr(fn, "__wrapped__"):
            monkeypatch.setattr(optim_mod, fn_name, fn.__wrapped__)
    torch.manual_seed(42)
    gqa = _tiny_gpt(n_head=8, n_kv_head=2, vocab_size=262)
    mha = _tiny_gpt(n_head=8, n_kv_head=8, vocab_size=262)

    # GQA reduces both value_embeds and transformer_matrices vs MHA
    gqa_counts = gqa.num_scaling_params()
    mha_counts = mha.num_scaling_params()
    assert gqa_counts["value_embeds"] * 4 == mha_counts["value_embeds"]
    assert gqa_counts["transformer_matrices"] < mha_counts["transformer_matrices"]

    # Training forward + backward + MuonAdamW step
    opt = gqa.setup_optimizer()
    idx = torch.randint(0, 262, (2, 16))
    targets = torch.randint(0, 262, (2, 16))
    loss = gqa(idx, targets)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    # KV-cache inference via Engine
    gqa.eval()
    engine = Engine(gqa, _TinyTokenizer())
    results, _ = engine.generate_batch([261, 72, 101, 108], num_samples=2, max_tokens=4, temperature=0.0)
    assert len(results) == 2 and all(len(r) > 4 for r in results)
