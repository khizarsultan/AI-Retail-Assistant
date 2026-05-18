"""
Agent orchestration loop for the Smart Retail Assistant.

This module is intentionally framework-free — no LangChain, no LlamaIndex —
so the tool-use loop is fully visible.

The flow is the standard OpenAI Chat Completions tool-calling loop:

    user message
        │
        ▼
    ┌───────────────────────────────────────────┐
    │ chat.completions.create(..., tools=[...]) │
    └───────────────────────────────────────────┘
        │
        ▼
    finish_reason == "tool_calls"? ──── no ──► return assistant text
        │ yes
        ▼
    for each tool_call:
        run it locally, build a {role: "tool", tool_call_id, content} msg
        │
        ▼
    append assistant message (with tool_calls) + tool result messages
        │
        ▼
    loop back

The full conversation (including tool_call and tool messages) is kept on
``Agent.messages``. That list IS the agent's memory — re-sending it on
every turn is how the model remembers earlier order ids, product searches,
etc. There is no separate "memory store".

Notes on OpenAI tool-use shape:
  - Tool calls come back on ``message.tool_calls`` (an array on the
    assistant message), not as content blocks.
  - Each tool result is a separate ``{"role": "tool", ...}`` message
    referencing the ``tool_call_id``.
  - Tool arguments arrive as a JSON *string* under
    ``tool_call.function.arguments`` and must be json.loads'd before use.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI

from app.prompts import SYSTEM_PROMPT
from app.tools import OPENAI_TOOL_SCHEMAS, dispatch_tool

log = logging.getLogger(__name__)

# Default model. Override with SMART_RETAIL_MODEL env var.
# gpt-4.1-mini is a strong balance of capability and cost for agentic tool use.
DEFAULT_MODEL = os.environ.get("SMART_RETAIL_MODEL", "gpt-4.1-mini")

# Safety cap on tool-use iterations per user turn. 10 is comfortably more than
# any legitimate request needs (the hardest scripted case is 3 calls).
MAX_TOOL_ITERATIONS = 10


@dataclass
class ToolCall:
    """Record of a single tool call, for UI display."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class AgentTurn:
    """The result of one user turn: assistant reply + all tool calls made."""

    reply: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class Agent:
    """Stateful conversational agent.

    One ``Agent`` instance corresponds to one user session. ``messages``
    accumulates the full transcript including tool_calls / tool messages
    so the model has the context to handle follow-ups.
    """

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
    ) -> None:
        self.client = client or OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        # The messages list is the agent's memory. We seed it with the
        # system prompt so it's sent on every API call.
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        on_tool_call: Callable[[ToolCall], None] | None = None,
    ) -> AgentTurn:
        """Process one user turn and return the assistant's reply.

        ``on_tool_call`` is an optional callback fired as each tool call
        completes — useful for streaming the trace into a UI.
        """
        self.messages.append({"role": "user", "content": user_message})

        tool_calls_this_turn: list[ToolCall] = []

        for iteration in range(MAX_TOOL_ITERATIONS):
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                tools=OPENAI_TOOL_SCHEMAS,
                messages=self.messages,
            )

            choice = response.choices[0]
            assistant_msg = choice.message

            # Persist the assistant's turn so the next API call has the full
            # context. We serialize it back to the dict shape the API expects
            # because the SDK returns a Pydantic model.
            self.messages.append(_assistant_msg_to_dict(assistant_msg))

            if choice.finish_reason != "tool_calls":
                # Model is done. Return the text content.
                log.debug(
                    "Turn complete after %d iterations, %d tool calls",
                    iteration + 1,
                    len(tool_calls_this_turn),
                )
                return AgentTurn(
                    reply=(assistant_msg.content or "").strip(),
                    tool_calls=tool_calls_this_turn,
                )

            # Model wants to call one or more tools. Run them all, then
            # send each result back as its own ``role: "tool"`` message.
            for tool_call in assistant_msg.tool_calls or []:
                name = tool_call.function.name
                # OpenAI returns arguments as a JSON string — parse it.
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    arguments = {}
                    result: dict[str, Any] = {
                        "error": f"Model produced invalid JSON arguments: {exc}"
                    }
                else:
                    log.info("Tool call: %s(%s)", name, arguments)
                    result = dispatch_tool(name, arguments)

                call_record = ToolCall(
                    name=name, arguments=arguments, result=result
                )
                tool_calls_this_turn.append(call_record)
                if on_tool_call is not None:
                    on_tool_call(call_record)

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": _stringify_result(result),
                    }
                )

        # Iteration cap hit — bail out gracefully.
        log.warning("Hit MAX_TOOL_ITERATIONS without a final answer")
        return AgentTurn(
            reply=(
                "Sorry — I got stuck working that out. Could you rephrase or "
                "break the request into smaller steps?"
            ),
            tool_calls=tool_calls_this_turn,
        )

    def reset(self) -> None:
        """Wipe conversation memory but keep the system prompt seeded."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assistant_msg_to_dict(msg: Any) -> dict[str, Any]:
    """Convert an OpenAI ChatCompletionMessage to the dict shape the API
    expects on subsequent calls.

    The SDK returns a Pydantic model, but on the way back in we need a
    plain dict with the right keys. ``content`` must be present even when
    null, and ``tool_calls`` (if any) must include each call's full shape.
    """
    out: dict[str, Any] = {
        "role": "assistant",
        "content": msg.content,
    }
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out


def _stringify_result(result: dict[str, Any]) -> str:
    """Serialize a tool result for the model.

    A JSON string is easier for the model to parse reliably than a Python
    repr — and it sidesteps any non-string keys.
    """
    return json.dumps(result, ensure_ascii=False, default=str)
