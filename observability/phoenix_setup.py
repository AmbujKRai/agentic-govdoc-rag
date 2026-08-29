"""
Local, free, self-hosted tracing via Arize Phoenix - every Groq call (across
naive_rag.py, the agent graph, the router, and the eval harness's
faithfulness scorer, since they all funnel through agent/groq_utils.py) gets
captured automatically via openinference-instrumentation-groq, no manual
span code needed at each call site.

Usage:
    from observability.phoenix_setup import setup_tracing
    setup_tracing()
    # ...run naive_rag / agent queries as normal...
    # view traces at http://localhost:6006

Deliberately opt-in (not auto-imported by groq_utils.py) - the eval harness
runs many calls back-to-back and doesn't need a UI server spun up for that;
this is for interactive/demo runs where inspecting a trace is useful.
"""

import sys

# Phoenix's own startup code prints a unicode globe emoji, which crashes on
# Windows' default cp1252 console - same class of issue hit elsewhere in
# this project (see naive_rag.py, agent/graph.py).
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import phoenix as px
from openinference.instrumentation.groq import GroqInstrumentor
from phoenix.otel import register

_started = False


def setup_tracing(launch_ui: bool = True) -> None:
    global _started
    if _started:
        print("Tracing already active.")
        return

    if launch_ui:
        px.launch_app()
        print("Phoenix UI: http://localhost:6006")

    tracer_provider = register(project_name="govdoc-copilot", auto_instrument=False)
    GroqInstrumentor().instrument(tracer_provider=tracer_provider)
    _started = True
    print("Tracing active - every Groq call will now appear in Phoenix.")


if __name__ == "__main__":
    import time

    setup_tracing()
    print("\nPhoenix is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
