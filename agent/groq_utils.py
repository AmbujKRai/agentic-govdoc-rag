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

WAIT_RE = re.compile(r"try again in ([\d.]+)s")


def chat_completion_with_retry(client: Groq, **kwargs):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            last_error = e
            match = WAIT_RE.search(str(e))
            wait_s = float(match.group(1)) + 1 if match else DEFAULT_BACKOFF * (attempt + 1)
            print(f"  [rate limited, waiting {wait_s:.1f}s before retry {attempt + 1}/{MAX_RETRIES}]")
            time.sleep(wait_s)
    raise last_error
