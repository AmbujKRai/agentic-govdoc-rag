"""
Resilient wrapper around Groq chat completions. The free tier has a fairly
low TPM (tokens-per-minute) cap, and we genuinely hit it running just 4
queries back-to-back during testing (each query makes 3-5 calls: route,
retrieve x1-2, check_sufficiency x1-2, generate). This only gets worse once
the eval harness runs a full golden set. Retry with backoff instead of
letting the whole run die on a transient 429.
"""

import re
import time

from groq import Groq, RateLimitError

MAX_RETRIES = 4
DEFAULT_BACKOFF = 10  # seconds, used when the error message doesn't include a suggested wait
MAX_AUTO_WAIT = 90  # seconds - a per-minute (TPM) limit clears in single-digit seconds;
# a daily (TPD) quota can ask for 15-20+ minutes, which isn't worth silently
# blocking on. Past this threshold, fail fast with a clear error instead.

# Groq's message includes a suggested wait either as "try again in 5.14s" or,
# for daily quota exhaustion, "try again in 18m24.19s".
WAIT_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")


def _parse_wait_seconds(error_message: str) -> float | None:
    match = WAIT_RE.search(error_message)
    if not match:
        return None
    minutes, seconds = match.groups()
    return (int(minutes) * 60 if minutes else 0) + float(seconds)


def chat_completion_with_retry(client: Groq, **kwargs):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            last_error = e
            wait_s = _parse_wait_seconds(str(e))
            if wait_s is not None and wait_s > MAX_AUTO_WAIT:
                raise RuntimeError(
                    f"Groq rate limit requires a {wait_s:.0f}s wait (likely daily quota) - "
                    f"exceeds MAX_AUTO_WAIT={MAX_AUTO_WAIT}s, not auto-retrying. Original error: {e}"
                ) from e
            if wait_s is None:
                wait_s = DEFAULT_BACKOFF * (attempt + 1)
            else:
                wait_s += 1
            print(f"  [rate limited, waiting {wait_s:.1f}s before retry {attempt + 1}/{MAX_RETRIES}]")
            time.sleep(wait_s)
    raise last_error
