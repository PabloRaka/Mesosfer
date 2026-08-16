"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat.chat_cli
"""
import argparse
import os
import re
import shutil
import time
import torch
from mesosfer.utils.common import compute_init, autodetect_device_type
from mesosfer.eval.engine import Engine
from mesosfer.utils.checkpoint_manager import load_model

DEFAULT_SYSTEM_PROMPT = (
    "[Instruksi Sistem: Identitas asisten adalah Mesosfer. "
    "Mesosfer adalah AI yang cerdas, sopan, dan berfokus pada cybersecurity defensif, pemrograman, dan penalaran teknis. "
    "Jawablah pertanyaan pengguna secara langsung, akurat, dan ramah sebagai Mesosfer.]"
)

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|rl")
parser.add_argument('-g', '--model-tag', '--depth', type=str, default=None, help='Model tag or depth to load (e.g. d12)')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Maximum new tokens per response (default: 512)')
parser.add_argument('--repetition-penalty', '--rep-penalty', type=float, default=1.1, help='Repetition penalty (default: 1.1)')
parser.add_argument('--system-prompt', type=str, default=DEFAULT_SYSTEM_PROMPT, help='System prompt for Mesosfer persona')
parser.add_argument('--no-system-prompt', action='store_true', help='Disable system prompt')
parser.add_argument('--plain', action='store_true', help='Disable styled terminal UI')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
args = parser.parse_args()

if args.no_system_prompt:
    active_system_prompt = ""
else:
    active_system_prompt = args.system_prompt.strip()


# Terminal UI helpers
USE_COLOR = (not args.plain) and os.isatty(1)
TERM_WIDTH = max(72, min(shutil.get_terminal_size((96, 24)).columns, 110))


class C:
    reset = "\033[0m" if USE_COLOR else ""
    bold = "\033[1m" if USE_COLOR else ""
    dim = "\033[2m" if USE_COLOR else ""
    orange = "\033[38;5;208m" if USE_COLOR else ""
    cyan = "\033[38;5;45m" if USE_COLOR else ""
    green = "\033[38;5;120m" if USE_COLOR else ""
    gray = "\033[38;5;244m" if USE_COLOR else ""
    white = "\033[38;5;255m" if USE_COLOR else ""
    
    # Mesosfer Aurora Gradient Palette
    teal = "\033[38;5;51m" if USE_COLOR else ""
    blue = "\033[38;5;39m" if USE_COLOR else ""
    indigo = "\033[38;5;99m" if USE_COLOR else ""
    purple = "\033[38;5;141m" if USE_COLOR else ""
    magenta = "\033[38;5;201m" if USE_COLOR else ""


def strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def visible_len(text):
    return len(strip_ansi(text))


def fit_text(text, width):
    if visible_len(text) <= width:
        return text + " " * (width - visible_len(text))
    plain = strip_ansi(text)
    return plain[:max(0, width - 1)] + "…"


class CodeBlockState:
    def __init__(self):
        self.in_code = False
        self.buffer = ""
        
    def feed(self, text):
        self.buffer += text
        out = ""
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            stripped = line.strip()
            if stripped.startswith("```"):
                if not self.in_code:
                    self.in_code = True
                    lang = stripped[3:].strip() or "code"
                    width = min(80, TERM_WIDTH - 4)
                    out += f"\n{C.purple}╭─ {C.teal}{C.bold}{lang}{C.reset}{C.purple}{'─' * (width - 4 - len(lang))}╮{C.reset}\n"
                else:
                    self.in_code = False
                    width = min(80, TERM_WIDTH - 4)
                    out += f"{C.purple}╰{'─' * (width - 2)}╯{C.reset}\n"
            else:
                if self.in_code:
                    out += f"{C.purple}│{C.reset}  {C.cyan}{line}{C.reset}\n"
                else:
                    out += line + "\n"
        return out

    def flush(self):
        out = ""
        if self.buffer:
            if self.in_code:
                out += f"{C.purple}│{C.reset}  {C.cyan}{self.buffer}{C.reset}"
            else:
                out += self.buffer
            self.buffer = ""
        if self.in_code:
            self.in_code = False
            width = min(80, TERM_WIDTH - 4)
            out += f"\n{C.purple}╰{'─' * (width - 2)}╯{C.reset}\n"
        return out


def rule(label="", color=C.purple):
    if not label:
        print(f"{color}{'─' * TERM_WIDTH}{C.reset}")
        return
    text = f" {label} "
    side = max(0, TERM_WIDTH - visible_len(text))
    print(f"{color}{'─' * (side // 2)}{C.reset}{C.bold}{label}{C.reset}{color}{'─' * (side - side // 2)}{C.reset}")


def boxed(title, left_lines, right_lines):
    inner = TERM_WIDTH - 2
    left_w = min(34, max(26, inner // 3))
    sep = "│"
    right_w = inner - left_w - 1
    # Sleek purple borders with teal highlighted title
    print(f"{C.purple}╭{'─' * (TERM_WIDTH - 2)}╮{C.reset}")
    print(f"{C.purple}│{C.reset}{C.bold}{C.teal}{fit_text(' ' + title, TERM_WIDTH - 2)}{C.reset}{C.purple}│{C.reset}")
    print(f"{C.purple}├{'─' * left_w}┬{'─' * right_w}┤{C.reset}")
    rows = max(len(left_lines), len(right_lines))
    for i in range(rows):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""
        print(f"{C.purple}│{C.reset}{fit_text(left, left_w)}{C.purple}{sep}{C.reset}{fit_text(right, right_w)}{C.purple}│{C.reset}")
    print(f"{C.purple}╰{'─' * left_w}┴{'─' * right_w}╯{C.reset}")


def print_welcome(meta):
    step = meta.get("step", "?") if isinstance(meta, dict) else "?"
    model_config = meta.get("model_config", {}) if isinstance(meta, dict) else {}
    depth = model_config.get("n_layer", "?")
    embd = model_config.get("n_embd", "?")
    
    # Elegant planet block art representing layers of the Mesosfer atmosphere
    left = [
        "",
        f"     {C.bold}{C.white}Welcome back.{C.reset}",
        "",
        f"     {C.teal}    .▄███▄.     {C.reset}",
        f"     {C.cyan}   .█ █ █ █.    {C.reset}",
        f"     {C.blue}  ▄█████████▄   {C.reset}",
        f"     {C.indigo} ══{C.purple}█████████{C.indigo}══  {C.reset}",
        f"     {C.magenta}   ▀█▀   ▀█▀    {C.reset}",
        "",
        f"   {C.gray}{args.source.upper()} · Step {step}{C.reset}",
    ]
    
    cwd_name = os.path.basename(os.getcwd())
    right = [
        f"{C.purple}{C.bold}Tips for getting started{C.reset}",
        f"{C.white}/help{C.reset}  for commands and shortcuts",
        f"{C.white}/save{C.reset}  saves conversation to markdown",
        f"{C.white}\\ at end{C.reset} of line for multi-line",
        "",
        f"{C.purple}{C.bold}Session Info{C.reset}",
        f"model tag : {args.model_tag or 'auto'}",
        f"layers    : {depth} · embd: {embd}",
        f"sampler   : temp: {args.temperature} · rep-pen: {args.repetition_penalty}",
        f"system    : {'Mesosfer Persona' if active_system_prompt else 'None'}",
    ]
    boxed(" Mesosfer Chat CLI ", left, right)
    print(f"{C.gray}? for shortcuts · Ctrl+C or /exit to quit{C.reset}")


def print_assistant_header():
    print(f"\n{C.purple}{C.bold}✦ Mesosfer{C.reset}")


def print_footer():
    print()


def print_console(message):
    print(f"{C.cyan}›{C.reset} {message}")

# Init the model and tokenizer

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

try:
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
except FileNotFoundError as e:
    if args.source in ["sft", "rl"]:
        print(f"\n{C.orange}⚠️  {args.source.upper()} checkpoint not found: {e}{C.reset}")
        print(f"{C.purple}Falling back to 'base' model...{C.reset}\n")
        args.source = "base"
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    else:
        raise e

# Special tokens for the chat state machine
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# Create Engine for efficient generation
engine = Engine(model, tokenizer)
max_seq_len = meta.get("model_config", {}).get("sequence_len", 2048) if isinstance(meta, dict) else 2048

if not args.prompt:
    print_welcome(meta)

conversation_tokens = [bos]
chat_history = []

while True:

    if args.prompt:
        # Get the prompt from the launch command
        user_input = args.prompt
    else:
        # Get the prompt interactively from the console, supporting multi-line ending with \
        try:
            lines = []
            while True:
                prompt_symbol = f"\n{C.teal}{C.bold}❯{C.reset} " if not lines else f"{C.gray}...{C.reset} "
                line = input(prompt_symbol)
                if line.endswith("\\"):
                    lines.append(line[:-1])
                    continue
                else:
                    lines.append(line)
                    break
            user_input = "\n".join(lines).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.gray}Goodbye.{C.reset}")
            break

    # Handle special commands
    if user_input.lower() in ['quit', 'exit', '/quit', '/exit']:
        print(f"{C.gray}Goodbye.{C.reset}")
        break

    if user_input.lower() in ['clear', '/clear']:
        conversation_tokens = [bos]
        chat_history = []
        print_console("Conversation cleared.")
        continue

    if user_input.lower() in ['help', '/help', '?']:
        print_console("Commands: /clear, /system [prompt], /save [name.md], /temperature <0-2>, /topk <1-200>, /rep <1.0-2.0>, /maxtokens <1-4096>, /exit")
        print_console("Multi-line: End a line with '\\' to write the next line.")
        print_console("Ctrl+C during generation will stop the response stream safely.")
        continue

    if user_input.lower().startswith('/system'):
        parts = user_input.split(maxsplit=1)
        if len(parts) == 1:
            curr = active_system_prompt if active_system_prompt else "(none)"
            print_console(f"Current System Prompt: {curr}")
        else:
            active_system_prompt = parts[1].strip()
            # reset conversation tokens so new system prompt applies cleanly
            conversation_tokens = [bos]
            chat_history = []
            print_console("System prompt updated and conversation context reset.")
        continue

    if user_input.lower().startswith('/save'):
        if not chat_history:
            print_console("No conversation history to save yet.")
            continue
        parts = user_input.split()
        filename = ""
        if len(parts) > 1:
            filename = parts[1]
            if not filename.endswith(".md"):
                filename += ".md"
        else:
            import datetime
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_{now}.md"
            
        os.makedirs("data/chat_history", exist_ok=True)
        filepath = os.path.join("data/chat_history", filename)
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("# Mesosfer Chat Session\n")
                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model Source: {args.source.upper()}\n")
                if active_system_prompt:
                    f.write(f"System Prompt: {active_system_prompt}\n\n")
                else:
                    f.write("\n")
                f.write("---\n\n")
                for turn in chat_history:
                    role = turn["role"].upper()
                    content = turn["content"]
                    f.write(f"### 👤 {role}\n\n{content}\n\n")
            print_console(f"Chat history saved to {C.bold}{C.green}{filepath}{C.reset}")
        except Exception as e:
            print_console(f"Error saving history: {e}")
        continue

    if user_input.lower().startswith('/temperature'):
        parts = user_input.split()
        if len(parts) == 1:
            print_console(f"Temperature: {args.temperature}")
        else:
            try:
                args.temperature = max(0.0, min(2.0, float(parts[1])))
                print_console(f"Temperature set to {args.temperature}")
            except ValueError:
                print_console("Invalid temperature.")
        continue

    if user_input.lower().startswith('/topk'):
        parts = user_input.split()
        if len(parts) == 1:
            print_console(f"Top-k: {args.top_k}")
        else:
            try:
                args.top_k = max(1, min(200, int(parts[1])))
                print_console(f"Top-k set to {args.top_k}")
            except ValueError:
                print_console("Invalid top-k.")
        continue

    if user_input.lower().startswith(('/rep', '/repetition_penalty', '/rep_penalty')):
        parts = user_input.split()
        if len(parts) == 1:
            print_console(f"Repetition Penalty: {args.repetition_penalty}")
        else:
            try:
                args.repetition_penalty = max(1.0, min(2.5, float(parts[1])))
                print_console(f"Repetition penalty set to {args.repetition_penalty}")
            except ValueError:
                print_console("Invalid repetition penalty.")
        continue

    if user_input.lower().startswith(('/maxtokens', '/max_tokens', '/max-tokens')):
        parts = user_input.split()
        if len(parts) == 1:
            print_console(f"Max tokens: {args.max_tokens}")
        else:
            try:
                args.max_tokens = max(16, min(max_seq_len, int(parts[1])))
                print_console(f"Max tokens set to {args.max_tokens}")
            except ValueError:
                print_console("Invalid max tokens.")
        continue

    if not user_input:
        continue

    # Add User message to the conversation and history
    # If this is the first turn and a system prompt is active, prepend it
    is_first_turn = (len(chat_history) == 0)
    if is_first_turn and active_system_prompt:
        turn_prompt_text = f"{active_system_prompt}\n\n{user_input}"
    else:
        turn_prompt_text = user_input

    user_tokens = tokenizer.encode(turn_prompt_text)
    input_tokens_count = len(user_tokens)
    
    # Calculate context before generation
    curr_ctx = len(conversation_tokens) + input_tokens_count + 3 # + user_start, user_end, assistant_start
    ctx_pct = (curr_ctx / max_seq_len) * 100
    print(f"{C.gray}  ↳ Input: {C.bold}{C.cyan}{input_tokens_count}{C.reset}{C.gray} tokens │ Context: {curr_ctx:,}/{max_seq_len:,} ({ctx_pct:.1f}%){C.reset}")

    conversation_tokens.append(user_start)
    conversation_tokens.extend(user_tokens)
    conversation_tokens.append(user_end)
    chat_history.append({"role": "user", "content": user_input})

    # Kick off the assistant
    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }
    response_tokens = []
    response_text_list = []
    print_assistant_header()
    renderer = CodeBlockState()
    t_start = time.time()
    try:
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0] # pop the batch dimension (num_samples=1)
            response_tokens.append(token)
            if token == assistant_end:
                break
            token_text = tokenizer.decode([token])
            response_text_list.append(token_text)
            print(renderer.feed(token_text), end="", flush=True)
    except KeyboardInterrupt:
        print(f"\n{C.orange}[Generation interrupted by user]{C.reset}")
    finally:
        print(renderer.flush(), end="", flush=True)
    t_end = time.time()
    print_footer()

    # Accumulate full text for the chat history
    assistant_response = "".join(response_text_list).strip()
    chat_history.append({"role": "assistant", "content": assistant_response})

    # we have to ensure that the assistant end token is the last token
    # so even if generation ends due to max tokens, we have to append it to the end
    if not response_tokens or response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    # Show output token stats
    out_tokens_count = len(response_tokens)
    elapsed = max(0.001, t_end - t_start)
    tok_per_sec = out_tokens_count / elapsed
    total_ctx = len(conversation_tokens)
    total_pct = (total_ctx / max_seq_len) * 100
    print(f"{C.gray}  ↳ Output: {C.bold}{C.cyan}{out_tokens_count}{C.reset}{C.gray} tokens ({tok_per_sec:.1f} tok/s) │ Total Context: {total_ctx:,}/{max_seq_len:,} ({total_pct:.1f}%){C.reset}\n")

    # In the prompt mode, we only want a single response and exit
    if args.prompt:
        break
