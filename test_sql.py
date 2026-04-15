import os
import json
import sys
from dotenv import load_dotenv
load_dotenv(override=True)  
try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2-binary is required. Install it with:")
    print("  pip install psycopg2-binary")
    sys.exit(1)


# ── Config ──────────────────────────────────────────────────────────────────
DB_URL = os.getenv("ADMIN_DB_URL", "postgresql://rag_readonly:change_me@localhost:5432/agentic_rag_db")

# ── Test queries (all SELECT — safe for read-only role) ─────────────────────
TESTS = [
    {
        "name": "1. List all tables in the database",
        "sql": (
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name;"
        ),
    },
    {
        "name": "2. business_segments — all segments",
        "sql": "SELECT id, segment_name FROM business_segments ORDER BY id;",
    },
    {
        "name": "3. consolidated_financials — distinct metrics available",
        "sql": (
            "SELECT DISTINCT metric_name, metric_category "
            "FROM consolidated_financials ORDER BY metric_category, metric_name;"
        ),
    },
    {
        "name": "4. consolidated_financials — RIL Gross Revenue across all periods",
        "sql": (
            "SELECT period, fiscal_year, value_in_cr, yoy_change_pct, unit "
            "FROM consolidated_financials "
            "WHERE metric_name = 'Gross Revenue' ORDER BY period;"
        ),
    },
    {
        "name": "5. consolidated_financials — EBITDA and PAT for 2Q FY25",
        "sql": (
            "SELECT metric_name, value_in_cr, yoy_change_pct, unit "
            "FROM consolidated_financials "
            "WHERE period = '2Q FY25' AND fiscal_year = 'FY25' "
            "AND metric_name IN ('EBITDA', 'Profit After Tax', 'EBITDA Margin') "
            "ORDER BY metric_name;"
        ),
    },
    {
        "name": "6. jpl_financials — Jio PAT trend (all periods)",
        "sql": (
            "SELECT period, value_in_cr, yoy_change_pct "
            "FROM jpl_financials "
            "WHERE metric_name = 'Profit After Tax' ORDER BY period;"
        ),
    },
    {
        "name": "7. jpl_financials — Jio EBITDA Margin comparison Q2 FY25 vs Q2 FY24",
        "sql": (
            "SELECT period, value_in_cr, unit "
            "FROM jpl_financials "
            "WHERE metric_name = 'EBITDA Margin' AND period LIKE '2Q %' "
            "ORDER BY period;"
        ),
    },
    {
        "name": "8. jpl_operations — all operational KPIs for 2Q FY25",
        "sql": (
            "SELECT metric_name, value, unit, yoy_change_pct "
            "FROM jpl_operations "
            "WHERE period = '2Q FY25' ORDER BY metric_name;"
        ),
    },
    {
        "name": "9. jpl_operations — ARPU trend",
        "sql": (
            "SELECT period, value, yoy_change_pct "
            "FROM jpl_operations "
            "WHERE metric_name = 'ARPU' ORDER BY period;"
        ),
    },
    {
        "name": "10. consolidated_financials — Balance sheet metrics for 2Q FY25",
        "sql": (
            "SELECT metric_name, value_in_cr, unit "
            "FROM consolidated_financials "
            "WHERE period = '2Q FY25' AND metric_category = 'balance_sheet' "
            "ORDER BY metric_name;"
        ),
    },
    {
        "name": "11. Read-only enforcement — try INSERT (should FAIL)",
        "sql": "INSERT INTO business_segments (segment_name) VALUES ('hacked');",
        "expect_error": True,
    },
    {
        "name": "12. Read-only enforcement — try DELETE (should FAIL)",
        "sql": "DELETE FROM consolidated_financials WHERE id = 1;",
        "expect_error": True,
    },
]


def run_query(cur, sql: str) -> tuple:
    """Execute query, return (columns, rows) or raise."""
    cur.execute(sql)
    if cur.description:
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return columns, rows
    return [], []


def print_table(columns, rows, max_width=30):
    """Print a formatted table to console."""
    if not columns:
        print("    (no results)")
        return

    # Truncate long column names / values
    def trunc(val, w):
        s = str(val) if val is not None else "NULL"
        return s[:w].ljust(w)

    # Header
    header = " | ".join(trunc(c, max_width) for c in columns)
    separator = "-+-".join("-" * max_width for _ in columns)
    print(f"    {header}")
    print(f"    {separator}")

    # Rows (cap at 20 for readability)
    for row in rows[:20]:
        line = " | ".join(trunc(v, max_width) for v in row)
        print(f"    {line}")

    if len(rows) > 20:
        print(f"    ... and {len(rows) - 20} more rows")

    print(f"    ({len(rows)} rows)")


def main():
    print()
    print("=" * 70)
    print("  agentic_rag_db — Standalone SQL Test")
    print("=" * 70)
    print(f"  Connecting to: {DB_URL}")
    print()

    # ── Connect ─────────────────────────────────────────────────────────────
    try:
        conn = psycopg2.connect(DB_URL)
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        print("  Connection: OK")
    except Exception as e:
        print(f"  Connection FAILED: {e}")
        print()
        print("  Make sure:")
        print("    1. PostgreSQL is running")
        print("    2. agentic_rag_db database exists (run seed_db.py)")
        print("    3. rag_readonly role exists with correct password")
        print("    4. DATABASE_URL env var is set correctly")
        sys.exit(1)

    cur = conn.cursor()
    passed = 0
    failed = 0

    # ── Run tests ───────────────────────────────────────────────────────────
    for test in TESTS:
        name = test["name"]
        sql = test["sql"]
        expect_error = test.get("expect_error", False)

        print(f"  [{name}]")
        print(f"    SQL: {sql[:120]}{'...' if len(sql) > 120 else ''}")

        try:
            columns, rows = run_query(cur, sql)

            if expect_error:
                print("    FAIL: Expected an error but query succeeded!")
                failed += 1
            else:
                print_table(columns, rows)
                passed += 1

        except psycopg2.Error as e:
            if expect_error:
                print(f"    PASS: Correctly blocked — {e.pgerror.split('\\n')[0] if e.pgerror else str(e)}")
                passed += 1
            else:
                print(f"    FAIL: {e.pgerror.strip() if e.pgerror else str(e)}")
                failed += 1

        print()

    cur.close()
    conn.close()

    # ── Summary ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  Results: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    print("=" * 70)

    if failed == 0:
        print("  All tests passed! Database is working correctly.")
        print("  Your agent should be able to route financial queries to SQL.")
    else:
        print("  Some tests failed. Check the errors above.")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
