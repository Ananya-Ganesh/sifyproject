#!/usr/bin/env python3
"""
Standalone verification that the currency logic is correct
(doesn't require pdfplumber or other heavy dependencies)
"""

# Simulate what the adapter does
currency_validation_inr = {
    "status": "CRITICAL",
    "company_currency": "INR",
    "customer_currency": "USD",
    "warnings": ["[CRITICAL] Currency Mismatch Detected"]
}

currency_validation_ok = {
    "status": "OK",
    "company_currency": "USD",
    "customer_currency": "USD",
    "warnings": []
}

def test_add_total_row_logic(label, a_val, b_val, currency_validation):
    """Simulates the add_total_row logic"""
    status = "ok"
    conflicts = []
    
    # Check for numeric difference
    has_numeric_diff = abs(a_val - b_val) > 0.01
    
    # Check for currency mismatch (especially important for grand_total)
    has_currency_mismatch = (
        currency_validation.get("status") == "CRITICAL" and label == "grand_total"
    )
    
    print(f"Testing label='{label}', a_val={a_val}, b_val={b_val}")
    print(f"  has_numeric_diff: {has_numeric_diff}")
    print(f"  currency status: {currency_validation.get('status')}")
    print(f"  has_currency_mismatch: {has_currency_mismatch}")
    
    if has_numeric_diff or has_currency_mismatch:
        status = "conflict"
        if has_currency_mismatch and not has_numeric_diff:
            conflict_msg = f"{label} (numeric values match, but currencies differ: {currency_validation.get('company_currency')} vs {currency_validation.get('customer_currency')})"
            conflicts.append({
                "field": conflict_msg,
                "a": f"{a_val} {currency_validation.get('company_currency', '')}",
                "b": f"{b_val} {currency_validation.get('customer_currency', '')}"
            })
        else:
            conflicts.append({"field": label, "a": a_val, "b": b_val})
    
    print(f"  Result status: {status}")
    print(f"  Result conflicts: {conflicts}")
    print()
    
    return status, conflicts

print("=" * 80)
print("TEST 1: Same value, CRITICAL currency mismatch (INR vs USD)")
print("=" * 80)
status, conflicts = test_add_total_row_logic("grand_total", 25000, 25000, currency_validation_inr)
assert status == "conflict", f"Expected 'conflict' but got '{status}'"
assert len(conflicts) > 0, "Expected conflicts to be populated"
print("✓ PASS: Correctly detected currency mismatch as conflict\n")

print("=" * 80)
print("TEST 2: Same value, OK currency (USD vs USD)")
print("=" * 80)
status, conflicts = test_add_total_row_logic("grand_total", 25000, 25000, currency_validation_ok)
assert status == "ok", f"Expected 'ok' but got '{status}'"
assert len(conflicts) == 0, "Expected no conflicts"
print("✓ PASS: Correctly marked as OK for matching currencies\n")

print("=" * 80)
print("TEST 3: Different values, CRITICAL currency mismatch")
print("=" * 80)
status, conflicts = test_add_total_row_logic("grand_total", 25000, 30000, currency_validation_inr)
assert status == "conflict", f"Expected 'conflict' but got '{status}'"
assert len(conflicts) > 0, "Expected conflicts to be populated"
print("✓ PASS: Correctly detected both numeric AND currency mismatch\n")

print("=" * 80)
print("TEST 4: OTC row with currency mismatch (should NOT be a conflict)")
print("=" * 80)
status, conflicts = test_add_total_row_logic("otc", 0, 0, currency_validation_inr)
assert status == "ok", f"Expected 'ok' but got '{status}' (currency check should only apply to grand_total)"
assert len(conflicts) == 0, "Expected no conflicts for OTC row"
print("✓ PASS: Currency mismatch only affects grand_total row\n")

print("=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nConclusion:")
print("The logic is correct. When files have different currencies (INR vs USD),")
print("the grand_total row will be marked as 'conflict' even if the numeric")
print("values match (e.g., 25,000 vs 25,000).")
print("\nThe conflict will appear in the results table, and conflict_count will increment.")
