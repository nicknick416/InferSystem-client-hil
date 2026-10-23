"""Keyboard read regression: FD readiness must not mix with text buffering."""
import io
import os
import sys
import pytest
from HIL.runtime import Keyboard


def test_consecutive_keys_are_not_hidden_in_text_buffer(monkeypatch):
    read_fd, write_fd = os.pipe()
    stream = io.TextIOWrapper(io.BufferedReader(io.FileIO(read_fd, 'rb', closefd=False)))
    monkeypatch.setattr(sys, 'stdin', stream)
    keyboard = Keyboard(); keyboard.fd = read_fd
    try:
        os.write(write_fd, b'ppq')
        assert keyboard.read() == 'p'
        assert keyboard.read() == 'p'
        assert keyboard.read() == 'q'
        assert keyboard.read() == ''
    finally:
        stream.close(); os.close(read_fd); os.close(write_fd)


def test_keyboard_uppercase_and_empty_poll():
    read_fd, write_fd = os.pipe()
    keyboard = Keyboard(); keyboard.fd = read_fd
    try:
        assert keyboard.read() == ''
        os.write(write_fd, b'P')
        assert keyboard.read() == 'p'
    finally:
        os.close(read_fd); os.close(write_fd)
