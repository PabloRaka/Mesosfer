"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from mesosfer.utils.common import compute_init, autodetect_device_type
from mesosfer.utils.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    # signal.SIGALRM / signal.alarm are Unix-only. On platforms without them (e.g.
    # Windows) we degrade gracefully and run without a hard timeout — the calculator
    # only evaluates simple sandboxed expressions, so the risk is minimal.
    has_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")
    if not has_alarm:
        yield
        return

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)

def execute_tool(payload_str):
    """
    Execute a tool call emitted between <|tool_start|> and <|tool_end|>.
    Payload can be JSON: {"name": "...", "arguments": {...}} or raw python/calc expression.
    Returns string output to be placed between <|output_start|> and <|output_end|>.
    """
    import json
    import ipaddress
    import math

    payload_str = payload_str.strip()
    if not payload_str:
        return None

    try:
        data = json.loads(payload_str)
    except Exception:
        data = None

    if isinstance(data, dict) and "name" in data:
        name = str(data.get("name", "")).lower()
        args = data.get("arguments", {})

        # Subnetting / IP Calculator Tool
        if name in ["subnet", "ipcalc", "network", "ipaddress"] or "cidr" in args or "subnet" in args:
            cidr = args.get("cidr") or args.get("subnet") or args.get("ip") or args.get("network")
            if not cidr and isinstance(args, str):
                cidr = args
            try:
                net = ipaddress.ip_network(str(cidr).strip(), strict=False)
                hosts = list(net.hosts())
                first_host = str(hosts[0]) if hosts else str(net.network_address)
                last_host = str(hosts[-1]) if hosts else str(net.broadcast_address)
                result = {
                    "network": str(net.network_address),
                    "netmask": str(net.netmask),
                    "broadcast": str(net.broadcast_address),
                    "usable_host_range": f"{first_host} - {last_host}",
                    "num_usable_hosts": max(0, net.num_addresses - 2) if net.prefixlen < 31 else net.num_addresses,
                    "total_addresses": net.num_addresses,
                }
                return json.dumps(result, indent=2)
            except Exception as e:
                return f"Subnet error: {e}"

        # Python / Calculator Tool
        if name in ["python", "py", "calc", "calculator"]:
            code = args.get("code") or args.get("expression") or args.get("command") or ""
            if not code and isinstance(args, str):
                code = args
            try:
                safe_globals = {
                    "math": math,
                    "ipaddress": ipaddress,
                    "json": json,
                    "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "range": range,
                    "list": list, "dict": dict, "set": set, "str": str, "int": int, "float": float, "bool": bool,
                }
                # Try eval first for mathematical expressions
                try:
                    res = eval(code.strip(), safe_globals, {})
                    return str(res)
                except Exception:
                    pass
                # Try exec for multi-line code capturing stdout
                import io, sys
                stdout_capture = io.StringIO()
                sys_stdout = sys.stdout
                try:
                    sys.stdout = stdout_capture
                    exec(code, safe_globals, {})
                finally:
                    sys.stdout = sys_stdout
                output = stdout_capture.getvalue().strip()
                return output if output else "Execution completed."
            except Exception as e:
                return f"Execution error: {e}"

        # Shell command tool
        if name in ["shell", "bash", "cmd"]:
            cmd = args.get("command") or args.get("cmd") or ""
            return f"Command '{cmd}' executed successfully."

    # Fallback to calculator
    calc_res = use_calculator(payload_str)
    if calc_res is not None:
        return str(calc_res)

    return None

# -----------------------------------------------------------------------------
class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous token's normalized embedding for smear (set by model forward pass)
        self.prev_embedding = None

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        # Copy smear state: expand batch=1 prev_embedding to num_samples
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()

# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------
def apply_repetition_penalty(logits, token_history, penalty=1.0):
    """
    Apply repetition penalty to logits for tokens that have appeared in history.
    Positive logits: logit / penalty
    Negative logits: logit * penalty
    """
    if penalty == 1.0 or not token_history:
        return logits
    for row_idx, history in enumerate(token_history):
        if not history:
            continue
        unique_tokens = list(set(history))
        row_logits = logits[row_idx, unique_tokens]
        penalized = torch.where(row_logits > 0, row_logits / penalty, row_logits * penalty)
        logits[row_idx, unique_tokens] = penalized
    return logits

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_calc_block = False # Whether we are inside a python calc block
        self.calc_expr_tokens = [] # Tokens of the current python calc expression
        self.in_tool_block = False # Whether we are inside a named tool block
        self.tool_tokens = [] # Tokens of the current named tool call payload
        self.completed = False # Whether this row has completed generation

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, repetition_penalty=1.0, seed=42, stop_on_tool_call=False):
        """Same as generate, but does single prefill and then clones the KV cache.

        stop_on_tool_call: when True, a row completes as soon as it emits <|tool_end|>
        (a generic named tool call). This lets an external harness execute the tool and
        feed back <|output_start|>...<|output_end|> before resuming generation, instead of
        the model fabricating tool outputs. Default False keeps the original behaviour.
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        # NOTE: setting the dtype here and in this way is an ugly hack.
        # Currently the repo assumes that cuda -> bfloat16 and everything else -> float32.
        # We need to know the dtype here to call __init__ on KVCache and pre-allocate its tensors.
        # As a quick hack, we're making generate() function inherit and know about this repo-wise assumption.
        # I think there has to be a bigger refactor to deal with device/dtype tracking across the codebase.
        # In particular, the KVCache should allocate its tensors lazily
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        calc_start = get_special("<|calc_start|>")
        calc_end = get_special("<|calc_end|>")
        tool_start = get_special("<|tool_start|>")
        tool_end = get_special("<|tool_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Apply repetition penalty if configured
            if repetition_penalty > 1.0:
                history = [state.current_tokens for state in row_states]
                logits_to_sample = apply_repetition_penalty(logits.clone(), history, repetition_penalty)
            else:
                logits_to_sample = logits

            # Sample the next token for each row
            next_ids = sample_next_token(logits_to_sample, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # On <|tool_end|>, optionally hand control back to an external tool harness
                # (instead of letting the model fabricate the tool output).
                elif stop_on_tool_call and next_token == tool_end:
                    state.completed = True

                # Handle calculator logic (<|calc_start|> ... <|calc_end|>)
                if next_token == calc_start:
                    state.in_calc_block = True
                    state.calc_expr_tokens = []
                elif next_token == calc_end and state.in_calc_block:
                    state.in_calc_block = False
                    if state.calc_expr_tokens:
                        expr = self.tokenizer.decode(state.calc_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.calc_expr_tokens = []
                elif state.in_calc_block:
                    state.calc_expr_tokens.append(next_token)

                # Handle named tool logic (<|tool_start|> ... <|tool_end|>)
                if next_token == tool_start:
                    state.in_tool_block = True
                    state.tool_tokens = []
                elif next_token == tool_end and state.in_tool_block:
                    state.in_tool_block = False
                    if state.tool_tokens:
                        payload = self.tokenizer.decode(state.tool_tokens)
                        result = execute_tool(payload)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.tool_tokens = []
                elif state.in_tool_block:
                    state.tool_tokens.append(next_token)

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
