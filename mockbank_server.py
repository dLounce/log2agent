"""
MockBank MCP server — exposes business tools for the RfP/loan automation domain.
Run standalone: python mockbank_server.py
Inspect live:   npx @modelcontextprotocol/inspector python mockbank_server.py
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MockBank")


@mcp.tool()
def submit_request(case_id: str, amount: float) -> dict:
    """Submit a request for payment into the system. Returns a submission receipt."""
    return {"case_id": case_id, "status": "submitted", "amount": amount, "ticket": f"SUB-{case_id}"}


@mcp.tool()
def check_administration(case_id: str, amount: float) -> dict:
    """Administrative check on a payment request. Returns whether it passes admin rules."""
    passes = amount <= 5000
    return {"case_id": case_id, "admin_check": "pass" if passes else "flagged", "threshold": 5000}


@mcp.tool()
def request_supervisor_approval(case_id: str, amount: float) -> dict:
    """Route a request to a supervisor for final approval. Returns routing confirmation."""
    return {"case_id": case_id, "routed_to": "supervisor", "status": "awaiting_approval"}


@mcp.tool()
def execute_payment(case_id: str, amount: float) -> dict:
    """Execute the actual payment transfer. Irreversible financial action."""
    return {"case_id": case_id, "status": "payment_executed", "amount": amount, "txn": f"TXN-{case_id}"}


@mcp.tool()
def confirm_payment_handled(case_id: str) -> dict:
    """Confirm a payment has been fully processed and settled."""
    return {"case_id": case_id, "status": "payment_handled", "settled": True}


@mcp.tool()
def notify_employee(case_id: str, message: str) -> dict:
    """Send a notification to the requesting employee (e.g. rejection reason)."""
    return {"case_id": case_id, "notified": True, "channel": "email"}


@mcp.tool()
def check_credit_score(case_id: str) -> dict:
    """Retrieve the credit score for a loan applicant (loan process)."""
    return {"case_id": case_id, "credit_score": 720, "band": "good"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
