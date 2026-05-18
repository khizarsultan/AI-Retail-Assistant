"""
Tools exposed to the Smart Retail Assistant agent.

Three tools are exposed:

1. ``search_catalog``     - read-only search over products.json
2. ``get_order_details``  - read-only lookup over orders.json
3. ``initiate_return``    - *mock* mutation guarded by a business rule
                            (returns are only allowed on ``delivered`` orders).

The agent does NOT enforce the business rule on its own. The rule is enforced
*inside* the tool, so a misbehaving / hallucinating LLM cannot bypass it.
The system prompt tells Claude to verify the order status first, but the
real guarantee lives here in code.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent.parent / "data"))


def _load_json(filename: str) -> list[dict[str, Any]]:
    path = _DATA_DIR / filename
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# Loaded once at import time. For a real app this would be a database call.
_PRODUCTS: list[dict[str, Any]] = _load_json("products.json")
_ORDERS: list[dict[str, Any]] = _load_json("orders.json")

# In-memory log of mock returns initiated this session.
# Indexed by order_id so a second return attempt on the same order is detected.
_RETURNS: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def search_catalog(query: str) -> dict[str, Any]:
    """Case-insensitive substring search across product name and description.

    Returns at most 10 matches, ranked by where the match occurred
    (name match > description match) and then by stock availability.
    """
    if not query or not query.strip():
        return {"query": query, "count": 0, "results": []}

    q = query.lower().strip()
    tokens = [t for t in q.split() if t]

    scored: list[tuple[int, dict[str, Any]]] = []
    for product in _PRODUCTS:
        name = product["name"].lower()
        desc = product["description"].lower()

        # Score: name token hits are worth more than description hits.
        score = 0
        for tok in tokens:
            if tok in name:
                score += 3
            if tok in desc:
                score += 1
        # Whole-phrase bonus.
        if q in name:
            score += 5
        elif q in desc:
            score += 2

        if score > 0:
            scored.append((score, product))

    # Sort: highest score first, then in-stock first.
    scored.sort(key=lambda item: (-item[0], -item[1]["stock_count"]))
    results = [p for _, p in scored[:10]]

    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


def get_order_details(order_id: str) -> dict[str, Any]:
    """Look up a single order by id and enrich it with product names.

    Returns ``{"found": False, ...}`` if the order does not exist, instead of
    raising. The agent should be able to tell the user politely.
    """
    if not order_id or not order_id.strip():
        return {"found": False, "error": "order_id is required."}

    oid = order_id.strip()
    for order in _ORDERS:
        if order["order_id"].lower() == oid.lower():
            # Enrich items with product info so the agent doesn't have to make
            # a second tool call just to know what the customer bought.
            enriched_items = []
            for item in order["items"]:
                product = next(
                    (p for p in _PRODUCTS if p["id"] == item["product_id"]), None
                )
                enriched_items.append(
                    {
                        "product_id": item["product_id"],
                        "product_name": product["name"] if product else "Unknown",
                        "unit_price": product["price"] if product else None,
                        "quantity": item["quantity"],
                    }
                )

            return {
                "found": True,
                "order_id": order["order_id"],
                "customer_name": order["customer_name"],
                "status": order["status"],
                "delivery_date": order.get("delivery_date"),
                "items": enriched_items,
            }

    return {
        "found": False,
        "order_id": order_id,
        "error": f"No order found with id '{order_id}'.",
    }


def initiate_return(order_id: str, reason: str) -> dict[str, Any]:
    """Mock-initiate a return for an order.

    Business rule (enforced here, NOT just in the prompt):
        Only ``delivered`` orders can be returned. ``processing`` and
        ``shipped`` orders must be cancelled instead, not returned.
    """
    if not order_id or not order_id.strip():
        return {"success": False, "error": "order_id is required."}
    if not reason or not reason.strip():
        return {"success": False, "error": "A return reason is required."}

    order = next(
        (o for o in _ORDERS if o["order_id"].lower() == order_id.strip().lower()),
        None,
    )

    if order is None:
        return {
            "success": False,
            "order_id": order_id,
            "error": f"No order found with id '{order_id}'.",
        }

    status = order["status"]
    if status != "delivered":
        return {
            "success": False,
            "order_id": order["order_id"],
            "current_status": status,
            "error": (
                f"Returns are only allowed for delivered orders. "
                f"Order {order['order_id']} is currently '{status}'. "
                "Please wait until it is delivered, or cancel it instead."
            ),
        }

    # Idempotency: don't open two RMAs for the same order.
    if order["order_id"] in _RETURNS:
        existing = _RETURNS[order["order_id"]]
        return {
            "success": False,
            "order_id": order["order_id"],
            "error": (
                f"A return ({existing['rma_id']}) has already been initiated "
                "for this order."
            ),
        }

    rma_id = f"RMA-{uuid.uuid4().hex[:8].upper()}"
    record = {
        "rma_id": rma_id,
        "order_id": order["order_id"],
        "reason": reason.strip(),
        "status": "return_initiated",
    }
    _RETURNS[order["order_id"]] = record
    log.info("Return initiated: %s", record)

    return {"success": True, **record}


# ---------------------------------------------------------------------------
# Tool dispatch table + JSON schemas for Anthropic tool use
# ---------------------------------------------------------------------------

# Maps tool name -> Python callable. The agent loop uses this to invoke tools
# requested by the model.
TOOL_REGISTRY = {
    "search_catalog": search_catalog,
    "get_order_details": get_order_details,
    "initiate_return": initiate_return,
}


# Schemas follow Anthropic's tool-use spec: each tool has a name, a
# description (which the model reads to decide when to call it), and an
# input_schema (JSON schema for the arguments).
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_catalog",
        "description": (
            "Search the product catalog by free-text query. Returns matching "
            "products with id, name, description, price, and stock_count. "
            "Use this whenever the user asks about products, availability, "
            "or prices. Do NOT invent product details — only use what this "
            "tool returns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Free-text search query, e.g. 'wireless mouse', "
                        "'4k monitor', 'usb-c hub'."
                    ),
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_order_details",
        "description": (
            "Look up a single order by its order_id (e.g. 'ORD-12345'). "
            "Returns the order status (one of 'processing', 'shipped', "
            "'delivered'), customer name, items, and delivery date. Use this "
            "before quoting any order status to the user, and ALWAYS use it "
            "before attempting a return."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order id, e.g. 'ORD-12345'.",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "initiate_return",
        "description": (
            "Initiate a return for an order. This is a state-changing action. "
            "You MUST call get_order_details first and confirm the order "
            "status is 'delivered' before calling this tool. If the order is "
            "'processing' or 'shipped', refuse the return and explain why — "
            "do NOT call this tool. The tool itself will also reject "
            "non-delivered orders as a safety net."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order id to return.",
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Customer-provided reason for the return. Ask the "
                        "user for one if they haven't given it."
                    ),
                },
            },
            "required": ["order_id", "reason"],
        },
    },
]


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Invoke a tool by name. Used by the agent loop."""
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}
    try:
        return TOOL_REGISTRY[name](**arguments)
    except TypeError as exc:
        # Wrong arguments from the model — return as a tool error so the
        # model can recover instead of crashing the loop.
        return {"error": f"Invalid arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - we want to surface any failure
        log.exception("Tool %s raised", name)
        return {"error": f"Internal error in {name}: {exc}"}


# ---------------------------------------------------------------------------
# OpenAI-format schemas (chat.completions tool-calling spec)
# ---------------------------------------------------------------------------
# OpenAI wraps every function as {"type": "function", "function": {...}} and
# uses ``parameters`` instead of ``input_schema``. The actual JSON schema body
# is identical to Anthropic's, so we convert on the fly rather than duplicate.

OPENAI_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOL_SCHEMAS
]
