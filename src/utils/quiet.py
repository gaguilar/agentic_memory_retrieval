"""Utilities to suppress verbose output during background operations."""

import logging
from contextlib import contextmanager


@contextmanager
def suppress_verbose_extraction_output():
    """
    Temporarily suppress noisy logs during memory extraction.

    Use when showing a spinner (e.g. "Extracting memories...") so only
    the spinner is visible, not httpx/transformers/embedding logs.
    """
    noisy_loggers = [
        "httpx",
        "urllib3",
        "httpcore",
        "memory.tag_embedder",
        "memory.consolidator",
        "memory.linker",
        "conversation.manager",
    ]
    saved = {name: logging.getLogger(name).level for name in noisy_loggers}
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        yield
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)
