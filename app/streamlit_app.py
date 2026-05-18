"""
Streamlit chat UI for the Smart Retail Assistant.

Run with:
    streamlit run app/streamlit_app.py

Requires the ANTHROPIC_API_KEY env var (or a .env file in the project root).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make `app` importable when run via `streamlit run app/streamlit_app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

from app.agent import Agent, ToolCall

load_dotenv()

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Retail Assistant",
    page_icon="",
    layout="centered",
)

st.title("Smart Retail Assistant")
st.caption(
    "Ask about products, check an order, or start a return. "
    "Try: *“Where is order ORD-12345?”* or *“Do you have a wireless mouse?”*"
)


# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------

if not os.environ.get("OPENAI_API_KEY"):
    st.error(
        "🔑 **OPENAI_API_KEY is not set.** "
        "Add it to a `.env` file in the project root or export it in your "
        "shell, then restart the app."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "agent" not in st.session_state:
    st.session_state.agent = Agent()

# UI-facing message log. Distinct from agent.messages (which contains raw
# tool_use / tool_result blocks the user shouldn't see).
# Each entry: {"role": "user"|"assistant", "content": str, "tool_calls": [ToolCall]}
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Session")
    if st.button("🔄 New chat", use_container_width=True):
        st.session_state.agent.reset()
        st.session_state.display_messages = []
        st.rerun()

    st.divider()
    st.subheader("Model")
    st.code(st.session_state.agent.model, language=None)

    st.divider()
    st.subheader("Try these")
    st.markdown(
        "- *Do you sell a 4K monitor?*\n"
        "- *Where is order ORD-12345?*\n"
        "- *I want to return that order. It scratches easily.*\n"
        "- *I want to buy a wireless mouse, but only if my refund for "
        "order ORD-999 has been processed.*\n"
        "- *Return order ORD-999 — it's taking too long.* (should refuse)"
    )


# ---------------------------------------------------------------------------
# Render existing history
# ---------------------------------------------------------------------------


def _render_tool_calls(tool_calls: list[ToolCall]) -> None:
    """Show each tool call in a collapsible expander."""
    for call in tool_calls:
        label = f"`{call.name}({_short_args(call.arguments)})`"
        with st.expander(label, expanded=False):
            st.markdown("**Arguments**")
            st.code(json.dumps(call.arguments, indent=2), language="json")
            st.markdown("**Result**")
            st.code(
                json.dumps(call.result, indent=2, default=str), language="json"
            )


def _short_args(args: dict) -> str:
    """One-line argument summary for the expander label."""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 30:
            s = s[:27] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)


for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            _render_tool_calls(msg["tool_calls"])
        st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Chat input + agent turn
# ---------------------------------------------------------------------------

prompt = st.chat_input("Ask about a product or an order…")

if prompt:
    # Show the user message immediately.
    st.session_state.display_messages.append(
        {"role": "user", "content": prompt, "tool_calls": []}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent.
    with st.chat_message("assistant"):
        # Use a status to give live feedback while tools fire.
        with st.status("Thinking…", expanded=False) as status:
            calls_so_far: list[ToolCall] = []

            def _on_tool_call(call: ToolCall) -> None:
                calls_so_far.append(call)
                status.update(label=f"Calling `{call.name}`…")

            try:
                turn = st.session_state.agent.chat(
                    prompt, on_tool_call=_on_tool_call
                )
                status.update(label="Done", state="complete")
            except Exception as exc:  # noqa: BLE001
                status.update(label="Error", state="error")
                st.error(f"Something went wrong: {exc}")
                st.stop()

        if turn.tool_calls:
            _render_tool_calls(turn.tool_calls)

        st.markdown(turn.reply)

    st.session_state.display_messages.append(
        {
            "role": "assistant",
            "content": turn.reply,
            "tool_calls": turn.tool_calls,
        }
    )
