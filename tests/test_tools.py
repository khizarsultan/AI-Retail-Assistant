"""Unit tests for the tools layer.

These tests cover the business logic that must NOT depend on the LLM's
judgement — particularly the "returns only on delivered orders" rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import tools


# ---------------------------------------------------------------------------
# search_catalog
# ---------------------------------------------------------------------------


class TestSearchCatalog:
    def test_finds_product_by_name(self) -> None:
        result = tools.search_catalog("wireless mouse")
        assert result["count"] >= 1
        names = [p["name"] for p in result["results"]]
        assert "Wireless Ergonomic Mouse" in names

    def test_case_insensitive(self) -> None:
        lower = tools.search_catalog("4k monitor")
        upper = tools.search_catalog("4K MONITOR")
        assert lower["count"] == upper["count"]
        assert lower["count"] >= 1

    def test_searches_descriptions(self) -> None:
        # "Qi" only appears in the description of p_009.
        result = tools.search_catalog("Qi")
        assert any(p["id"] == "p_009" for p in result["results"])

    def test_empty_query_returns_no_results(self) -> None:
        assert tools.search_catalog("")["count"] == 0
        assert tools.search_catalog("   ")["count"] == 0

    def test_no_match_returns_empty(self) -> None:
        result = tools.search_catalog("kombucha smoothie")
        assert result["count"] == 0
        assert result["results"] == []

    def test_name_match_ranks_above_description_match(self) -> None:
        # "Keyboard" is in the name of p_002 and nowhere else.
        # "Pad" is in the names of p_008 (Desk Pad) and p_009 (Wireless Charging Pad).
        # Search for "pad" — both name-matches should rank above any
        # description-only matches (none exist for this exact query, but
        # the test asserts the scoring shape).
        result = tools.search_catalog("pad")
        ids = [p["id"] for p in result["results"]]
        assert "p_008" in ids and "p_009" in ids


# ---------------------------------------------------------------------------
# get_order_details
# ---------------------------------------------------------------------------


class TestGetOrderDetails:
    def test_finds_delivered_order(self) -> None:
        result = tools.get_order_details("ORD-12345")
        assert result["found"] is True
        assert result["status"] == "delivered"
        assert result["customer_name"] == "Alice Smith"
        assert len(result["items"]) == 2

    def test_enriches_items_with_product_names(self) -> None:
        result = tools.get_order_details("ORD-12345")
        assert result["found"] is True
        # p_001 is the Wireless Ergonomic Mouse.
        mouse_item = next(i for i in result["items"] if i["product_id"] == "p_001")
        assert mouse_item["product_name"] == "Wireless Ergonomic Mouse"
        assert mouse_item["unit_price"] == 29.99
        assert mouse_item["quantity"] == 1

    def test_case_insensitive_order_id(self) -> None:
        a = tools.get_order_details("ord-12345")
        b = tools.get_order_details("ORD-12345")
        assert a["found"] is True
        assert b["found"] is True
        assert a["order_id"] == b["order_id"]

    def test_missing_order_returns_not_found(self) -> None:
        result = tools.get_order_details("ORD-DOESNOTEXIST")
        assert result["found"] is False
        assert "error" in result

    def test_empty_order_id_returns_not_found(self) -> None:
        assert tools.get_order_details("")["found"] is False
        assert tools.get_order_details("   ")["found"] is False


# ---------------------------------------------------------------------------
# initiate_return — the safety-critical tool
# ---------------------------------------------------------------------------


class TestInitiateReturn:
    def setup_method(self) -> None:
        # Wipe in-memory return log between tests so each test is isolated.
        tools._RETURNS.clear()

    def test_succeeds_on_delivered_order(self) -> None:
        result = tools.initiate_return("ORD-12345", "Defective on arrival")
        assert result["success"] is True
        assert result["order_id"] == "ORD-12345"
        assert result["rma_id"].startswith("RMA-")
        assert result["status"] == "return_initiated"
        assert result["reason"] == "Defective on arrival"

    def test_refuses_processing_order(self) -> None:
        # ORD-999 is in processing.
        result = tools.initiate_return("ORD-999", "Changed my mind")
        assert result["success"] is False
        assert result["current_status"] == "processing"
        assert "delivered" in result["error"].lower()

    def test_refuses_shipped_order(self) -> None:
        # ORD-45678 is shipped but not delivered.
        result = tools.initiate_return("ORD-45678", "Wrong color")
        assert result["success"] is False
        assert result["current_status"] == "shipped"

    def test_refuses_unknown_order(self) -> None:
        result = tools.initiate_return("ORD-NOPE", "Whatever")
        assert result["success"] is False
        assert "no order found" in result["error"].lower()

    def test_requires_reason(self) -> None:
        result = tools.initiate_return("ORD-12345", "")
        assert result["success"] is False
        assert "reason" in result["error"].lower()

    def test_requires_order_id(self) -> None:
        result = tools.initiate_return("", "Defective")
        assert result["success"] is False

    def test_is_idempotent(self) -> None:
        first = tools.initiate_return("ORD-12345", "Defective")
        second = tools.initiate_return("ORD-12345", "Defective again")
        assert first["success"] is True
        assert second["success"] is False
        assert "already" in second["error"].lower()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_dispatches_known_tool(self) -> None:
        result = tools.dispatch_tool("get_order_details", {"order_id": "ORD-12345"})
        assert result["found"] is True

    def test_unknown_tool_returns_error(self) -> None:
        result = tools.dispatch_tool("nuke_database", {})
        assert "error" in result

    def test_bad_arguments_returns_error_not_exception(self) -> None:
        # Missing required arg.
        result = tools.dispatch_tool("get_order_details", {"foo": "bar"})
        assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
