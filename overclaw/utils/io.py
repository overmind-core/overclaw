"""Shared terminal I/O utilities."""

from __future__ import annotations

import getpass
import sys


def read_api_key_masked(label: str) -> str:
    """Read an API key from the TTY, echoing each character as ``*``.

    Paste-friendly: the ``*`` echo lets the user see how many characters have
    been accepted without revealing the value.  Falls back to ``getpass`` (no
    echo at all) when stdin is not a terminal.
    """
    prompt = f"  {label}: "
    if not sys.stdin.isatty():
        return getpass.getpass(prompt).strip()

    sys.stdout.write(prompt)
    sys.stdout.flush()
    chars: list[str] = []

    try:
        if sys.platform == "win32":
            import msvcrt

            while True:
                ch = msvcrt.getwch()
                if ch in "\r\n":
                    break
                if ord(ch) == 3:
                    raise KeyboardInterrupt
                if ch in "\x08\x7f":
                    if chars:
                        chars.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                    continue
                if ch.isprintable():
                    chars.append(ch)
                    sys.stdout.write("*")
                    sys.stdout.flush()
        else:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while True:
                    ch = sys.stdin.read(1)
                    if ch in "\r\n":
                        break
                    if ch == "\x03":
                        raise KeyboardInterrupt
                    if ch in "\x08\x7f":
                        if chars:
                            chars.pop()
                            sys.stdout.write("\b \b")
                            sys.stdout.flush()
                        continue
                    if ch and ch.isprintable():
                        chars.append(ch)
                        sys.stdout.write("*")
                        sys.stdout.flush()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
    finally:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return "".join(chars).strip()
