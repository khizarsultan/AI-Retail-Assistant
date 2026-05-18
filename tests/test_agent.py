"""
Tests for the agent orchestration loop.

These use a stub OpenAI client so they run offline and deterministically.
The stub replays a scripted sequence of model responses, letting us verify
that the loop correctly:

  * runs each tool the model asks for
  * feeds the tool result back into the next call as a ``role: "tool"`` msg
  * stops when ``finish_reason != "tool_calls"``
  * preserves message history across iterations (the "memory" requirement)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import tools
from app.agent import Agent


# ---------------------------------------------------------------------------
# Minimal stand-ins for the OpenAI chat.completions response shape.
# Just enough to satisfy the attribute access patterns in agent.py.
# ---------------------------------------------------------------------------


@dataclass
class StubFunction:
    name: str
    arguments: str  # OpenAI returns JSON-encoded string


@dataclass
class StubToolCall:
    id: str
    function: StubFunction
    type: str = "function"


@dataclass
class StubMessage:
    content: str | None = None
    tool_calls: list[StubToolCall] | None = None
    role: str = "assistant"


@dataclass
class StubChoice:
    message: StubMessage
    finish_reason: str
    index: int = 0


@dataclass
class StubResponse:
    choices: list[StubChoice]


def make_text_response(text: str) -> StubResponse:
    """Helper: a response with just a text reply (no tool calls)."""
    return StubResponse(
        choices=[
            StubChoice(
                message=StubMessage(content=text),
                finish_reason="stop",
            )
        ]
    )


def make_tool_call_response(
    name: str, arguments: dict[str, Any], call_id: str = "call_1"
) -> StubResponse:
    """Helper: a response asking the model to call one tool."""
    return StubResponse(
        choices=[
            StubChoice(
                message=StubMessage(
                    content=None,
                    tool_calls=[
                        StubToolCall(
                            id=call_id,
                            function=StubFunction(
                                name=name,
                                arguments=json.dumps(arguments),
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ]
    )


class StubCompletions:
    """Replays a fixed sequence of responses on .create().

    Captured kwargs are deep-copied so that the agent's subsequent mutation
    of its ``messages`` list doesn't change what tests see.
    """

    def __init__(self, scripted: list[StubResponse]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []  # captured kwargs per call

    def create(self, **kwargs: Any) -> StubResponse:
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            snapshot["messages"] = [{**m} for m in snapshot["messages"]]
        self.calls.append(snapshot)
        if not self._scripted:
            raise AssertionError(
                "Agent made more API calls than the test scripted."
            )
        return self._scripted.pop(0)


class StubChat:
    def __init__(self, completions: StubCompletions) -> None:
        self.completions = completions


class StubClient:
    """Mimics the OpenAI client surface used by agent.py: client.chat.completions.create()."""

    def __init__(self, scripted: list[StubResponse]) -> None:
        self.chat = StubChat(StubCompletions(scripted))

    @property
    def messages_log(self) -> StubCompletions:
        return self.chat.completions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentLoop:
    def setup_method(self) -> None:
        tools._RETURNS.clear()

    def test_simple_no_tool_response(self) -> None:
        """If the model replies without a tool call, the loop returns immediately."""
        client = StubClient([make_text_response("Hi there! How can I help?")])
        agent = Agent(client=client)  # type: ignore[arg-type]
        turn = agent.chat("Hello")

        assert turn.reply == "Hi there! How can I help?"
        assert turn.tool_calls == []
        assert len(client.messages_log.calls) == 1

    def test_single_tool_call(self) -> None:
        """One tool call, then a final text answer."""
        client = StubClient(
            [
                make_tool_call_response(
                    "get_order_details", {"order_id": "ORD-12345"}, "call_a"
                ),
                make_text_response(
                    "Your order ORD-12345 has been delivered on 2026-02-25."
                ),
            ]
        )
        agent = Agent(client=client)  # type: ignore[arg-type]
        turn = agent.chat("Where is order ORD-12345?")

        assert len(turn.tool_calls) == 1
        call = turn.tool_calls[0]
        assert call.name == "get_order_details"
        assert call.arguments == {"order_id": "ORD-12345"}
        assert call.result["found"] is True
        assert call.result["status"] == "delivered"
        assert "delivered" in turn.reply.lower()

        # The loop should have made exactly two API calls.
        assert len(client.messages_log.calls) == 2

        # The second API call's last message must be a tool result message.
        second_call_messages = client.messages_log.calls[1]["messages"]
        last_msg = second_call_messages[-1]
        assert last_msg["role"] == "tool"
        assert last_msg["tool_call_id"] == "call_a"

    def test_multi_step_reasoning(self) -> None:
        """The canonical multi-step prompt from the spec.

        "I want to buy a wireless mouse, but only if my refund for order
        #999 has been processed."

        Step 1: model calls get_order_details("ORD-999").
        Step 2: model calls search_catalog("wireless mouse").
        Step 3: model writes a final reply combining both.
        """
        client = StubClient(
            [
                make_tool_call_response(
                    "get_order_details", {"order_id": "ORD-999"}, "call_1"
                ),
                make_tool_call_response(
                    "search_catalog", {"query": "wireless mouse"}, "call_2"
                ),
                make_text_response(
                    "Order ORD-999 is still processing, so the refund "
                    "hasn't been issued yet. I found a Wireless Ergonomic "
                    "Mouse if you'd like to proceed anyway."
                ),
            ]
        )
        agent = Agent(client=client)  # type: ignore[arg-type]
        turn = agent.chat(
            "I want to buy a wireless mouse, but only if my refund for "
            "order ORD-999 has been processed."
        )

        assert [c.name for c in turn.tool_calls] == [
            "get_order_details",
            "search_catalog",
        ]
        assert turn.tool_calls[0].result["status"] == "processing"
        assert turn.tool_calls[1].result["count"] >= 1

    def test_memory_across_turns(self) -> None:
        """The order id from turn 1 must be in the message history at turn 2."""
        client = StubClient(
            [
                # Turn 1: look up order.
                make_tool_call_response(
                    "get_order_details", {"order_id": "ORD-12345"}, "call_1"
                ),
                make_text_response("Delivered on 2026-02-25."),
                # Turn 2: user says "return it" without the id. Model must
                # pull the id from memory and initiate the return.
                make_tool_call_response(
                    "initiate_return",
                    {"order_id": "ORD-12345", "reason": "Defective"},
                    "call_2",
                ),
                make_text_response("Return initiated."),
            ]
        )
        agent = Agent(client=client)  # type: ignore[arg-type]

        agent.chat("Where is order ORD-12345?")
        turn2 = agent.chat("Actually I want to return it. It's defective.")

        # The second turn's first API call (which is calls[2] overall, since
        # turn 1 used 2 calls) must contain ORD-12345 somewhere in history.
        turn2_first_call = client.messages_log.calls[2]
        history_text = json.dumps(turn2_first_call["messages"], default=str)
        assert "ORD-12345" in history_text

        # And the return must have succeeded.
        ret_call = next(c for c in turn2.tool_calls if c.name == "initiate_return")
        assert ret_call.result["success"] is True

    def test_iteration_cap_prevents_infinite_loop(self) -> None:
        """If the model keeps requesting tools, the loop bails out cleanly."""
        from app.agent import MAX_TOOL_ITERATIONS

        scripted = [
            make_tool_call_response(
                "search_catalog", {"query": "anything"}, f"call_{i}"
            )
            for i in range(MAX_TOOL_ITERATIONS + 5)
        ]
        client = StubClient(scripted)
        agent = Agent(client=client)  # type: ignore[arg-type]
        turn = agent.chat("Loop forever please")

        assert turn.reply  # non-empty
        assert len(turn.tool_calls) == MAX_TOOL_ITERATIONS

    def test_reset_clears_memory_but_keeps_system_prompt(self) -> None:
        client = StubClient([make_text_response("ok")])
        agent = Agent(client=client)  # type: ignore[arg-type]
        agent.chat("hello")
        assert len(agent.messages) > 1  # system + user + assistant
        agent.reset()
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"

    def test_handles_invalid_json_arguments(self) -> None:
        """If the model produces malformed JSON for tool args, we recover gracefully."""
        client = StubClient(
            [
                StubResponse(
                    choices=[
                        StubChoice(
                            message=StubMessage(
                                content=None,
                                tool_calls=[
                                    StubToolCall(
                                        id="bad_call",
                                        function=StubFunction(
                                            name="search_catalog",
                                            arguments="{not valid json",
                                        ),
                                    )
                                ],
                            ),
                            finish_reason="tool_calls",
                        )
                    ]
                ),
                make_text_response("Sorry, I couldn't process that."),
            ]
        )
        agent = Agent(client=client)  # type: ignore[arg-type]
        turn = agent.chat("find me something")

        assert len(turn.tool_calls) == 1
        assert "error" in turn.tool_calls[0].result
        assert turn.reply


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
