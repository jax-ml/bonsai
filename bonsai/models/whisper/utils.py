"""
Minimal utilities for JAX Whisper implementation.

This module contains only the essential utility functions needed for the JAX Whisper model,
removing all PyTorch-specific and subtitle formatting code.
"""

import sys

system_encoding = sys.getdefaultencoding()

if system_encoding != "utf-8":

    def make_safe(string):
        # replaces any character not representable using the system default encoding with an '?',
        # avoiding UnicodeEncodeError (https://github.com/openai/whisper/discussions/729).
        return string.encode(system_encoding, errors="replace").decode(system_encoding)

else:

    def make_safe(string):
        # utf-8 can encode any Unicode code point, so no need to do the round-trip encoding
        return string


def exact_div(x, y):
    """Exact division that asserts no remainder."""
    assert x % y == 0
    return x // y


def str2bool(string):
    """Convert string to boolean."""
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_int(string):
    """Convert string to int or None."""
    return None if string == "None" else int(string)


def optional_float(string):
    """Convert string to float or None."""
    return None if string == "None" else float(string)