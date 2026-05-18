# Smart Retail Assistant

A conversational AI agent for an e-commerce store that can answer product questions, look up orders, and process return requests — built as a take-home assessment.

---

## What It Does

The assistant acts as a customer support agent. A user can:

- Ask about products (search by name, category, or description)
- Check the status of an order
- Request a return for a delivered order

The agent figures out on its own which action to take, calls the right tool, and responds in plain English. It also remembers context within the conversation — so if you ask about order #ORD-12345 and then say "I want to return it", it knows which order you mean.

---

## How I Approached This

The first decision was whether to use a framework like LangChain or build the agent loop directly. I went with the **native OpenAI tool-calling API** for a simple reason: the loop itself is the core of this assessment, and wrapping it in a framework would hide exactly what the evaluator wants to see.

The agent works like this:

1. User sends a message
2. The full conversation history + tool definitions are sent to the model
3. The model either writes a reply (done) or asks to call a tool
4. If it asks for a tool, we run it locally and send the result back
5. Repeat until the model writes a final reply

### The Three Tools

| Tool | What it does |
|---|---|
| `search_catalog` | Searches products by name and description. Scores results by relevance — name matches rank higher than description matches. |
| `get_order_details` | Looks up an order by ID and enriches it with product names and prices so the agent doesn't need a second call just to describe what was ordered. |
| `initiate_return` | Starts a return. Enforces the delivered-only rule **in code**, not just in the prompt. |

### The Delivered-Only Rule

The system prompt tells the model to check order status before initiating a return. But prompts can be ignored or misread. So the actual enforcement is inside `initiate_return()`:

```python
if order["status"] != "delivered":
    return {"success": False, "error": "Returns only allowed on delivered orders..."}
```

Even if the model makes a mistake, the tool itself will reject the request. This is the same pattern you'd use for any irreversible action in production: guard it in code, not just instructions.

### Memory

Memory is the `messages` list on the `Agent` object. Every API call sends the complete conversation history — user messages, assistant replies, tool calls, and tool results. There's no separate memory store. This works well within a session and is straightforward to reason about.

---

## Project Structure

```
smart_retail_assistant/
├── app/
│   ├── agent.py          # The tool-use loop
│   ├── tools.py          # Tool implementations + OpenAI schemas
│   ├── prompts.py        # System prompt
│   └── streamlit_app.py  # Chat UI
├── data/
│   ├── products.json     # Product catalog
│   └── orders.json       # Mock orders
├── tests/
│   ├── test_tools.py     # 21 unit tests for the tool layer
│   └── test_agent.py     # 7 tests for the agent loop (offline, stubbed)
├── requirements.txt
└── .env.example
```

---

## Running Locally

**1. Install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your OpenAI API key**
```bash
cp .env.example .env
# Edit .env and add your key
```

**3. Run the app**
```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

---

## Running Tests

```bash
pytest -v
```

28 tests, all offline — no API key needed. The agent tests use a stub OpenAI client that replays scripted model responses, so the orchestration logic is tested independently of the live model.

---

## Things to Try

| What you type | What happens |
|---|---|
| `Do you have a 4K monitor?` | Searches the catalog, returns real price and stock |
| `Where is order ORD-12345?` | Looks up the order, reports delivery date |
| `I want to return it` (follow-up) | Remembers the order from previous message, initiates return |
| `Return order ORD-999, it's late` | Refuses — ORD-999 is still processing, not delivered |
| `I want a wireless mouse but only if my refund for ORD-999 came through` | Checks the order first, then searches products — two tool calls, one response |

The UI shows a collapsible tool-call trace under each assistant reply so you can see exactly which tools were called and what they returned.

---

## Design Decisions Worth Noting

**No hallucinated data.** The system prompt explicitly instructs the model: prices and stock counts must come from `search_catalog`, order status must come from `get_order_details`. The model cannot quote data it didn't fetch.

**Graceful error handling at every layer.** If a tool receives bad arguments, it returns an error dict instead of raising an exception — so the model sees the error and explains it to the user rather than crashing. If the model somehow loops more than 10 times without finishing, the agent breaks out and returns a sensible fallback message.

**Tool schemas drive behavior.** Each tool's description is written to tell the model *when* to use it, not just *what* it does. For example, `initiate_return` explicitly states: "You MUST call `get_order_details` first" — this makes multi-step reasoning reliable without a rigid hardcoded control flow.

**Both Anthropic and OpenAI schemas are included.** `tools.py` exposes `TOOL_SCHEMAS` (Anthropic format) and auto-generates `OPENAI_TOOL_SCHEMAS` from it. Switching providers is a one-line change in `agent.py`.

---

## Tech Stack

- **Python 3.10+**
- **OpenAI API** — `gpt-4.1-mini` by default (override with `SMART_RETAIL_MODEL` env var)
- **Streamlit** — chat UI
- **python-dotenv** — environment variable management
- **pytest** — testing

---

## What I Would Add With More Time

- **Streaming responses** — send tokens as they arrive instead of waiting for the full reply
- **Semantic product search** — embeddings + vector similarity for natural queries like "something to reduce neck pain at my desk"
- **Conversation persistence** — save sessions to a database so they survive a page refresh
- **Eval harness** — scripted multi-turn test cases run against the live model to catch regressions in tool selection over time
- **LangGraph + proper database** — replace the in-memory state with a checkpointed graph and SQLite/Postgres for true production-grade persistence