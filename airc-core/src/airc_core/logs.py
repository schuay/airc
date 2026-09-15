# Copyright 2026 The airc developers
# SPDX-License-Identifier: MIT

"""Quieting for third-party loggers, shared by the suite's entry points.

Each component configures its own logging (basicConfig at its main), and each
one used to carry its own list of loggers to raise to WARNING -- airc-watchers
had "httpx", airc-room had four names, and the rest had none. The lists drifted
apart and then went stale together: when the Anthropic, Google and OpenAI SDKs
moved to httpx 2.x, which logs under "httpx2", every component started
reporting one INFO line per model call again.
"""

from __future__ import annotations

import logging

# Loggers whose INFO output is one line per request, per import or per token
# refresh, saying nothing the suite's own logs do not.
#
# httpx and httpx2 are two installed packages, not two spellings: httpx 2.x
# ships under the second name so it can coexist with 1.x, and it takes its
# logger name from the package. Both are present in a deployed venv and both
# log "HTTP Request: ...", so silencing either alone leaves half the calls in
# the journal.
_NOISY = (
    "httpx",
    "httpx2",
    "httpcore",
    "google.auth",
    "google_genai",
    "numexpr.utils",
)


def quiet_noisy_loggers() -> None:
    """Raise the noisy third-party loggers to WARNING.

    Call after basicConfig; a logger's level is independent of the root's, so
    the order only matters for the DEBUG check below.

    Does nothing when the root logger is at DEBUG. An operator who asked for
    debug output is usually chasing transport, auth or sandbox egress, which is
    exactly what these loggers carry -- and a level set here would override the
    request, since setLevel on the named logger wins over the root's.
    """
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        return
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
