"""
Terminal UI helpers for mesosfer scripts.

Pure stdlib — no external dependencies required.
Gracefully degrades to plain text when not running in a TTY
(e.g. CI, piped output, headless servers).

Public API
----------
  box(title, subtitle)          → print a bordered header box
  menu(title, items, subtitle)  → interactive arrow-key menu, returns index
  confirm(question)             → y/n prompt, returns bool
  section(label)                → print a section divider
  badge(label, value, color)    → print a colored key=value badge
  spinner(msg)                  → context manager for a simple spinner
"""

from __future__ import annotations

import os
import sys
import contextlib
import threading
import time
import itertools
from typing import Sequence

if sys.platform == "win32":
    import msvcrt
    tty = None
    termios = None
else:
    try:
        import tty
        import termios
    except ImportError:
        tty = None
        termios = None


# ── Terminal capability detection ─────────────────────────────────────────────

IS_TTY: bool = sys.stdout.isatty() and sys.stdin.isatty()

# Detect terminal width (fallback 80)
def _term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


# ── ANSI primitives ───────────────────────────────────────────────────────────

RESET  = "\033[0m"   if IS_TTY else ""
BOLD   = "\033[1m"   if IS_TTY else ""
DIM    = "\033[2m"   if IS_TTY else ""
ITALIC = "\033[3m"   if IS_TTY else ""

# Foreground colors
BLACK   = "\033[30m" if IS_TTY else ""
RED     = "\033[31m" if IS_TTY else ""
GREEN   = "\033[32m" if IS_TTY else ""
YELLOW  = "\033[33m" if IS_TTY else ""
BLUE    = "\033[34m" if IS_TTY else ""
MAGENTA = "\033[35m" if IS_TTY else ""
CYAN    = "\033[36m" if IS_TTY else ""
WHITE   = "\033[37m" if IS_TTY else ""

# Bright variants
BRIGHT_BLACK   = "\033[90m" if IS_TTY else ""
BRIGHT_RED     = "\033[91m" if IS_TTY else ""
BRIGHT_GREEN   = "\033[92m" if IS_TTY else ""
BRIGHT_YELLOW  = "\033[93m" if IS_TTY else ""
BRIGHT_BLUE    = "\033[94m" if IS_TTY else ""
BRIGHT_MAGENTA = "\033[95m" if IS_TTY else ""
BRIGHT_CYAN    = "\033[96m" if IS_TTY else ""
WHITE_BRIGHT   = "\033[97m" if IS_TTY else ""

# Cursor / screen control
CLEAR_LINE  = "\033[2K\r"  if IS_TTY else ""
CURSOR_UP   = "\033[1A"    if IS_TTY else ""
HIDE_CURSOR = "\033[?25l"  if IS_TTY else ""
SHOW_CURSOR = "\033[?25h"  if IS_TTY else ""


def _c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}" if IS_TTY else text


# ── Convenience print helpers ─────────────────────────────────────────────────

def info(msg: str)    -> None: print(_c(CYAN,          msg))
def success(msg: str) -> None: print(_c(BRIGHT_GREEN,  msg))
def warn(msg: str)    -> None: print(_c(YELLOW,        msg))
def err(msg: str)     -> None: print(_c(BRIGHT_RED,    msg))
def dim(msg: str)     -> None: print(_c(BRIGHT_BLACK,  msg))


# ── Box / header ──────────────────────────────────────────────────────────────

_BOX_H  = "─"
_BOX_TL = "╭"
_BOX_TR = "╮"
_BOX_BL = "╰"
_BOX_BR = "╯"
_BOX_V  = "│"


def box(title: str, subtitle: str = "", width: int = 0) -> None:
    """
    Print a rounded-corner box:

        ╭─────────────────────────────────────╮
        │   Upload Mesosfer Artifacts to HF   │
        │   ~/.cache/mesosfer                 │
        ╰─────────────────────────────────────╯
    """
    w = width or min(_term_width() - 2, 60)
    inner = w - 2  # space inside the vertical bars

    top    = _BOX_TL + _BOX_H * w + _BOX_TR
    bottom = _BOX_BL + _BOX_H * w + _BOX_BR

    def _row(text: str, color: str = "") -> str:
        padded = f"  {text}"
        padded = padded.ljust(inner)[:inner]
        colored = _c(color, padded) if color else padded
        return f"{_c(BRIGHT_BLACK, _BOX_V)}{colored}{_c(BRIGHT_BLACK, _BOX_V)}"

    print()
    print(_c(BRIGHT_BLACK, top))
    print(_row(f"{BOLD}{BRIGHT_CYAN}{title}{RESET}", ""))
    if subtitle:
        print(_row(""))
        print(_row(subtitle, BRIGHT_BLACK))
    print(_c(BRIGHT_BLACK, bottom))
    print()


# ── Section divider ───────────────────────────────────────────────────────────

def section(label: str) -> None:
    """Print:   ── Label ──────────────────"""
    w = min(_term_width() - 2, 60)
    line = f"  {_c(BRIGHT_BLACK, '──')} {_c(BOLD + WHITE, label)} "
    remaining = w - len(label) - 6
    if remaining > 0:
        line += _c(BRIGHT_BLACK, _BOX_H * remaining)
    print(line)


# ── Badge ─────────────────────────────────────────────────────────────────────

def badge(label: str, value: str, color: str = CYAN) -> None:
    """Print:   label  value"""
    print(f"  {_c(BRIGHT_BLACK, label.ljust(14))}  {_c(color, value)}")


# ── Arrow-key interactive menu ────────────────────────────────────────────────

_KEY_UP    = b"\x1b[A"
_KEY_DOWN  = b"\x1b[B"
_KEY_ENTER = (b"\r", b"\n")
_KEY_QUIT  = (b"q", b"Q", b"\x1b", b"\x03")  # q, Q, ESC, Ctrl-C


def _read_key() -> bytes:
    """Read one keypress (raw mode). Returns bytes."""
    if sys.platform == "win32" and msvcrt is not None:
        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):  # Prefix for arrow keys on Windows
            ch2 = msvcrt.getch()
            if ch2 == b"H":  # Up arrow
                return _KEY_UP
            elif ch2 == b"P":  # Down arrow
                return _KEY_DOWN
            return ch + ch2
        return ch

    if termios is None or tty is None:
        try:
            return sys.stdin.read(1).encode()
        except Exception:
            return b"\n"

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1)
        if ch == b"\x1b":
            # Escape sequence — read up to 2 more bytes
            rest = os.read(fd, 2)
            return ch + rest
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _render_menu(items: Sequence[str], cursor: int, descriptions: Sequence[str]) -> None:
    """Render menu items to stdout."""
    for i, item in enumerate(items):
        is_sel = i == cursor
        prefix = f"  {_c(BRIGHT_CYAN, '▶')} " if is_sel else "    "
        label  = _c(BOLD + WHITE, item) if is_sel else _c(WHITE, item)
        desc   = descriptions[i] if i < len(descriptions) else ""
        line   = f"{prefix}{label}"
        if desc:
            line += f"  {_c(BRIGHT_BLACK, desc)}"
        print(line)
    # hint
    print()
    print(_c(BRIGHT_BLACK, "  ↑/↓ navigate   Enter select   q quit"))


def _clear_menu(n_items: int) -> None:
    """Move cursor up and clear the rendered menu lines."""
    # items + 1 blank + 1 hint line
    total = n_items + 2
    for _ in range(total):
        sys.stdout.write(CURSOR_UP + CLEAR_LINE)
    sys.stdout.flush()


def menu(
    title: str,
    items: Sequence[str],
    descriptions: Sequence[str] = (),
    subtitle: str = "",
    default: int = 0,
) -> int:
    """
    Display an arrow-key navigable menu.

    Returns the selected index, or -1 if cancelled (q / ESC / Ctrl-C).
    Falls back to numbered input when not in a TTY.

    Example
    -------
        idx = menu(
            "What do you want to upload?",
            ["Model checkpoint", "Tokenizer", "Dataset", "Exit"],
            ["model weights + optimizer", "tokenizer.pkl + token_bytes.pt",
             "parquet shards", ""],
        )
    """
    box(title, subtitle)

    if not IS_TTY:
        # Non-interactive fallback
        for i, item in enumerate(items, 1):
            desc = descriptions[i - 1] if i - 1 < len(descriptions) else ""
            suffix = f"  {_c(BRIGHT_BLACK, desc)}" if desc else ""
            print(f"  [{i}] {item}{suffix}")
        print()
        while True:
            try:
                raw = input("  Choice: ").strip()
            except (KeyboardInterrupt, EOFError):
                return -1
            if raw.lower() in ("q", "quit", "exit"):
                return -1
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(items):
                    return idx
            warn(f"  Enter a number between 1 and {len(items)}, or q to quit.")

    # TTY: arrow-key mode
    cursor = default
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()
    _render_menu(items, cursor, descriptions)

    try:
        while True:
            key = _read_key()
            if key == _KEY_UP:
                cursor = (cursor - 1) % len(items)
            elif key == _KEY_DOWN:
                cursor = (cursor + 1) % len(items)
            elif key in _KEY_ENTER:
                _clear_menu(len(items))
                selected_label = _c(BRIGHT_GREEN, items[cursor])
                print(f"  {_c(BRIGHT_BLACK, '▶')} {selected_label}")
                print()
                return cursor
            elif key in _KEY_QUIT:
                _clear_menu(len(items))
                dim("  Cancelled.")
                print()
                return -1
            else:
                # Number shortcut: '1'..'9'
                try:
                    n = int(key.decode("utf-8")) - 1
                    if 0 <= n < len(items):
                        cursor = n
                        _clear_menu(len(items))
                        _render_menu(items, cursor, descriptions)
                        continue
                except (ValueError, UnicodeDecodeError):
                    pass
                continue
            _clear_menu(len(items))
            _render_menu(items, cursor, descriptions)
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()


# ── Confirm prompt ────────────────────────────────────────────────────────────

def confirm(question: str, default: bool = False) -> bool:
    """y/n prompt. Returns bool."""
    hint = "[Y/n]" if default else "[y/N]"
    try:
        raw = input(f"  {_c(YELLOW, '?')} {question} {_c(BRIGHT_BLACK, hint)}: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    if not raw:
        return default
    return raw in ("y", "yes")


# ── Spinner ───────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def spinner(msg: str):
    """
    Context manager that shows a spinner while work is happening.

        with spinner("Fetching file list…"):
            files = api.list_repo_files(...)
    """
    if not IS_TTY:
        print(f"  {msg}")
        yield
        return

    frames = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    stop_event = threading.Event()

    def _spin():
        sys.stdout.write(HIDE_CURSOR)
        while not stop_event.is_set():
            frame = next(frames)
            sys.stdout.write(f"\r  {_c(CYAN, frame)} {msg}")
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write(CLEAR_LINE)
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop_event.set()
        t.join()
