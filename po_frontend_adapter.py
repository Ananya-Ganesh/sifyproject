from __future__ import annotations

import os
from typing import Dict, Any, Optional, List

from po_auditor import DocumentExtractor as AuditorExtractor, POComparator, LineItem, FileExtractor
from po_engine import compare_files as engine_compare_files
from po_currency_validator import validate_currencies_before_comparison


def _lineitem_to_frontend(li: Optional[LineItem]) -> Dict[str, Any]:
    if li is None:
        return {}
    return {
        "raw_description": li.name,
        "normalized_description": li.name,
        "quantity": li.quantity,
        "unit_price": li.unit_price,
        "total_price": li.total,
    }


def compare_for_frontend(path_a: str, path_b: str) -> Dict[str, Any]:
    """
    Wrapper that intelligently chooses the best engine based on file type:
    - Excel files (.xlsx) use po_engine.py (better parsing)
    - PDFs and other formats use po_auditor.py (proven reliable)
    
    Returns response shaped like the original PDF-only API for the React UI.
    """
    # Detect file type
    ext_a = os.path.splitext(path_a)[1].lower()
    ext_b = os.path.splitext(path_b)[1].lower()
    
    # If both are Excel files, use po_engine for better parsing
    if ext_a == ".xlsx" and ext_b == ".xlsx":
        return _compare_with_engine(path_a, path_b)
    
    # Otherwise, use po_auditor (works better for PDFs)
    return _compare_with_auditor(path_a, path_b)


def _compare_with_engine(path_a: str, path_b: str) -> Dict[str, Any]:
    """Use po_engine for Excel file comparison with currency validation."""
    # Extract raw text (and tables) for currency validation
    file_extractor = FileExtractor()
    text_a, tables_a = file_extractor.extract(path_a)
    text_b, tables_b = file_extractor.extract(path_b)

    # Include table cell text in the validation corpus so currency symbols
    # present inside DOCX or XLSX tables are detected as well.
    def _merge_text_and_tables(text, tables):
        parts = [text or ""]
        if tables:
            parts.append("\n".join([" ".join([str(c) for c in row]) for row in tables]))
        return "\n".join(parts)

    full_text_a = _merge_text_and_tables(text_a, tables_a)
    full_text_b = _merge_text_and_tables(text_b, tables_b)

    # Validate currencies BEFORE comparison
    currency_validation = validate_currencies_before_comparison(
        full_text_a, full_text_b,
        company_po_name=os.path.basename(path_a),
        customer_po_name=os.path.basename(path_b)
    )
    
    result = engine_compare_files(path_a, path_b)
    
    # If engine reports an error (order mismatch), return early with warning
    if result.get("status") == "error":
        return {
            "summary": {
                "total_items_a": 0,
                "total_items_b": 0,
                "matched_items": 0,
                "conflict_count": 0,
                "order_similarity": result.get("product_overlap", 0),
                "otc_company": 0,
                "otc_customer": 0,
                "arc_company": 0,
                "arc_customer": 0,
                "grand_company": 0,
                "grand_customer": 0,
                "warning": None,
                "currency_validation": currency_validation,
            },
            "results": [],
        }
    
    # Map the engine's results to frontend format
    results = []
    for r in result.get("results", []):
        item_a = r.get("item_a")
        item_b = r.get("item_b")
        # Skip synthetic total rows (otc, arc, grand_total) to avoid duplication
        item_a_name = item_a.get("name", "").lower() if item_a else ""
        if item_a_name in ["otc", "arc", "grand_total"]:
            continue
        
        results.append({
            "status": r.get("status"),
            "similarity": r.get("similarity", 0),
            "item_a": {
                "raw_description": item_a.get("name", "") if item_a else "",
                "normalized_description": item_a.get("name", "") if item_a else "",
                "quantity": item_a.get("qty", 0) if item_a else 0,
                "unit_price": item_a.get("unit_price", 0.0) if item_a else 0.0,
                "total_price": item_a.get("total", 0.0) if item_a else 0.0,
            } if item_a else {},
            "item_b": {
                "raw_description": item_b.get("name", "") if item_b else "",
                "normalized_description": item_b.get("name", "") if item_b else "",
                "quantity": item_b.get("qty", 0) if item_b else 0,
                "unit_price": item_b.get("unit_price", 0.0) if item_b else 0.0,
                "total_price": item_b.get("total", 0.0) if item_b else 0.0,
            } if item_b else {},
            "conflicts": r.get("conflicts", []),
        })
    
    # Extract totals from parsed data
    parsed = result.get("parsed", {})
    company_parsed = parsed.get("company_parsed", {})
    customer_parsed = parsed.get("customer_parsed", {})
    otc_a = company_parsed.get("otc", 0.0)
    otc_b = customer_parsed.get("otc", 0.0)
    arc_a = company_parsed.get("arc", 0.0)
    arc_b = customer_parsed.get("arc", 0.0)
    grand_a = company_parsed.get("grand_total", 0.0)
    grand_b = customer_parsed.get("grand_total", 0.0)
    
    conflict_count = len([r for r in results if r["status"] == "conflict"])
    
    def add_total_row(label: str, a_val: float, b_val: float):
        nonlocal conflict_count
        status = "ok"
        conflicts: List[Dict[str, Any]] = []
        if abs(a_val - b_val) > 0.01:
            status = "conflict"
            conflict_count += 1
            conflicts.append({"field": label, "a": a_val, "b": b_val})
        results.append(
            {
                "status": status,
                "similarity": 1.0,
                "item_a": {
                    "raw_description": label,
                    "normalized_description": label,
                    "quantity": 0,
                    "unit_price": 0.0,
                    "total_price": a_val,
                },
                "item_b": {
                    "raw_description": label,
                    "normalized_description": label,
                    "quantity": 0,
                    "unit_price": 0.0,
                    "total_price": b_val,
                },
                "conflicts": conflicts,
            }
        )
    
    add_total_row("otc", otc_a, otc_b)
    add_total_row("arc", arc_a, arc_b)
    add_total_row("grand_total", grand_a, grand_b)
    
    summary = {
        "total_items_a": len([li for li in company_parsed.get("line_items", []) if li.get("name", "").lower() not in ["otc", "arc", "grand_total"]]),
        "total_items_b": len([li for li in customer_parsed.get("line_items", []) if li.get("name", "").lower() not in ["otc", "arc", "grand_total"]]),
        "matched_items": sum(1 for r in results if r["status"] != "missing_in_b" and r.get("item_a", {}).get("raw_description") not in ["otc", "arc", "grand_total"]),
        "conflict_count": conflict_count,
        "order_similarity": result.get("summary", {}).get("product_overlap", result.get("product_overlap", 0)),
        "otc_company": otc_a,
        "otc_customer": otc_b,
        "arc_company": arc_a,
        "arc_customer": arc_b,
        "grand_company": grand_a,
        "grand_customer": grand_b,
        "warning": None,
        "currency_validation": currency_validation,
    }

    return {"summary": summary, "results": results}


def _compare_with_auditor(path_a: str, path_b: str) -> Dict[str, Any]:
    """Use po_auditor for PDF and other file formats."""
    print(f"\n[DEBUG _compare_with_auditor] Starting comparison")
    print(f"  File A: {os.path.basename(path_a)}")
    print(f"  File B: {os.path.basename(path_b)}")
    
    # Extract raw text (and tables) for currency validation
    file_extractor = FileExtractor()
    text_a, tables_a = file_extractor.extract(path_a)
    text_b, tables_b = file_extractor.extract(path_b)

    def _merge_text_and_tables(text, tables):
        parts = [text or ""]
        if tables:
            parts.append("\n".join([" ".join([str(c) for c in row]) for row in tables]))
        return "\n".join(parts)

    full_text_a = _merge_text_and_tables(text_a, tables_a)
    full_text_b = _merge_text_and_tables(text_b, tables_b)

    print(f"  Text A length: {len(full_text_a) if full_text_a else 0}")
    print(f"  Text B length: {len(full_text_b) if full_text_b else 0}")
    print(f"  Text A (first 300 chars): {full_text_a[:300] if full_text_a else 'NONE'}")
    print(f"  Text B (first 300 chars): {full_text_b[:300] if full_text_b else 'NONE'}")

    # Validate currencies BEFORE comparison
    currency_validation = validate_currencies_before_comparison(
        full_text_a, full_text_b,
        company_po_name=os.path.basename(path_a),
        customer_po_name=os.path.basename(path_b)
    )
    
    # DEBUG: Log currency validation result
    print(f"\n[DEBUG] Currency Validation Result:")
    print(f"  Status: {currency_validation.get('status')}")
    print(f"  Company Currency: {currency_validation.get('company_currency')}")
    print(f"  Customer Currency: {currency_validation.get('customer_currency')}")
    print(f"  Warnings: {currency_validation.get('warnings')}")
    print()
    
    extractor = AuditorExtractor()
    po_a = extractor.extract_po(path_a, order_id="COMPANY_PO")
    po_b = extractor.extract_po(path_b, order_id="CUSTOMER_PO")

    # Filter out OTC/ARC from line items if they exist as separate line items,
    # to avoid double-counting. We'll add them as synthetic rows below.
    def is_charge_type(li: LineItem) -> bool:
        name_lower = li.name.lower().strip()
        return name_lower in ["otc", "arc", "one time charge", "annual recurring charge"]

    product_items_a = [li for li in po_a.line_items if not is_charge_type(li)]
    product_items_b = [li for li in po_b.line_items if not is_charge_type(li)]

    comparator = POComparator()
    matches, product_overlap, _ = comparator._match_line_items(
        product_items_a, product_items_b
    )

    results: List[Dict[str, Any]] = []
    conflict_count = 0

    for la, lb, sim in matches:
        status = "ok"
        conflicts: List[Dict[str, Any]] = []

        if lb is None:
            status = "missing_in_b"
        else:
            line_conf = comparator._compare_line_items(la, lb, sim)
            if line_conf:
                status = "conflict"
                conflict_count += 1
                # Flatten issues into simple field/a/b structure
                for issue in line_conf.get("issues", []):
                    conflicts.append(
                        {
                            "field": issue.get("field"),
                            "a": issue.get("company"),
                            "b": issue.get("customer"),
                        }
                    )

        results.append(
            {
                "status": status,
                "similarity": round(sim, 3),
                "item_a": _lineitem_to_frontend(la),
                "item_b": _lineitem_to_frontend(lb),
                "conflicts": conflicts,
            }
        )

    # Check if OTC/ARC already exist as line items
    otc_in_items_a = any(li.name.lower().strip() == "otc" for li in po_a.line_items)
    otc_in_items_b = any(li.name.lower().strip() == "otc" for li in po_b.line_items)
    arc_in_items_a = any(li.name.lower().strip() == "arc" for li in po_a.line_items)
    arc_in_items_b = any(li.name.lower().strip() == "arc" for li in po_b.line_items)

    # Add synthetic rows for OTC, ARC, and Grand Total so the frontend
    # explicitly shows these financial checks.
    def add_total_row(label: str, a_val: float, b_val: float):
        nonlocal conflict_count
        status = "ok"
        conflicts: List[Dict[str, Any]] = []
        
        # Check for numeric difference
        has_numeric_diff = abs(a_val - b_val) > 0.01
        
        # Check for currency mismatch (only when validator explicitly finds different currencies)
        has_currency_mismatch = (
            currency_validation.get("status") == "CRITICAL" and label == "grand_total"
        )
        
        # DEBUG
        if label == "grand_total":
            print(f"[DEBUG] Grand Total Check:")
            print(f"  currency_validation.status = {currency_validation.get('status')}")
            print(f"  has_currency_mismatch = {has_currency_mismatch}")
            print(f"  has_numeric_diff = {has_numeric_diff}")
            print(f"  a_val = {a_val}, b_val = {b_val}")
        
        if has_numeric_diff or has_currency_mismatch:
            status = "conflict"
            conflict_count += 1
            print(f"  [CONFLICT] Incremented conflict_count to {conflict_count} (label={label})")
            if has_currency_mismatch and not has_numeric_diff:
                # Currency issue without numeric diff
                if currency_validation.get("status") == "CRITICAL":
                    # Different currencies detected
                    conflict_msg = f"{label} (numeric values match, but currencies differ: {currency_validation.get('company_currency')} vs {currency_validation.get('customer_currency')})"
                    conflicts.append({
                        "field": conflict_msg,
                        "a": f"{a_val} {currency_validation.get('company_currency', '')}",
                        "b": f"{b_val} {currency_validation.get('customer_currency', '')}"
                    })
                else:
                    # Currency validation failed (WARNING) - can't verify
                    conflict_msg = f"{label} (currency symbols not detected - cannot verify currency consistency)"
                    conflicts.append({
                        "field": conflict_msg,
                        "a": a_val,
                        "b": b_val
                    })
            else:
                conflicts.append({"field": label, "a": a_val, "b": b_val})
        
        results.append(
            {
                "status": status,
                "similarity": 1.0,
                "item_a": {
                    "raw_description": label,
                    "normalized_description": label,
                    "quantity": 0,
                    "unit_price": 0.0,
                    "total_price": a_val,
                },
                "item_b": {
                    "raw_description": label,
                    "normalized_description": label,
                    "quantity": 0,
                    "unit_price": 0.0,
                    "total_price": b_val,
                },
                "conflicts": conflicts,
            }
        )

    # Only add synthetic OTC/ARC if they're not already in line items
    if not (otc_in_items_a and otc_in_items_b):
        add_total_row("otc", po_a.otc, po_b.otc)
    if not (arc_in_items_a and arc_in_items_b):
        add_total_row("arc", po_a.arc, po_b.arc)
    
    # For grand total, use sum of PRODUCT line items (excluding OTC/ARC if they're separate)
    # plus OTC and ARC. This gives the true total payable.
    grand_sum_a = sum(li.total for li in product_items_a) + po_a.otc + po_a.arc
    grand_sum_b = sum(li.total for li in product_items_b) + po_b.otc + po_b.arc
    add_total_row("grand_total", grand_sum_a, grand_sum_b)

    warning: Optional[str] = None
    # Only show the "different orders" warning when we are quite sure they are
    # unrelated: there are at least TWO items on BOTH sides and ZERO good
    # product matches (overlap ~ 0). This avoids warning on valid same-order
    # POs with formatting/value differences.
    min_items = min(len(po_a.line_items), len(po_b.line_items))
    if min_items >= 2 and product_overlap <= 0.05:
        warning = (
            "Error: The uploaded company PO and customer PO appear to belong to "
            "different orders. Please upload POs for the same order to run a "
            "reliable comparison."
        )
    
    # Add currency mismatch warning only when a DIFFERENT currency was detected
    # (status == 'CRITICAL'). If currencies are not detected (status == 'WARNING')
    # we intentionally do NOT display a top-level currency warning to the UI.
    # The grand_total row will still be flagged as a conflict when verification
    # isn't possible, but we avoid an extra warning banner unless currencies
    # are explicitly different.
    if currency_validation.get("status") == "CRITICAL":
        # Build a concise banner message (avoid filenames or extra text)
        company_code = currency_validation.get("company_currency", "unknown")
        customer_code = currency_validation.get("customer_currency", "unknown")
        currency_warning = (
            f"⚠️ CRITICAL: Currency mismatch detected: {company_code} vs {customer_code}. "
            "Price comparison may be invalid."
        )
        if warning:
            warning += " " + currency_warning
        else:
            warning = currency_warning

    summary = {
        "total_items_a": len(product_items_a),
        "total_items_b": len(product_items_b),
        "matched_items": sum(1 for r in results if r["status"] != "missing_in_b" and r.get("item_a", {}).get("raw_description") not in ["otc", "arc", "grand_total"]),
        "conflict_count": conflict_count,
        "order_similarity": product_overlap,
        "otc_company": po_a.otc,
        "otc_customer": po_b.otc,
        "arc_company": po_a.arc,
        "arc_customer": po_b.arc,
        "grand_company": grand_sum_a,
        "grand_customer": grand_sum_b,
        "warning": warning,
        "currency_validation": currency_validation,
    }
    
    print(f"\n[AUDIT RESULT] conflict_count={conflict_count}, currency_status={currency_validation.get('status')}")
    print(f"  Results rows: {len(results)}")
    for i, r in enumerate(results):
        if r.get('item_a', {}).get('raw_description') in ['grand_total', 'otc', 'arc']:
            print(f"    Row {i}: {r.get('item_a', {}).get('raw_description')} -> status={r.get('status')}")

    return {"summary": summary, "results": results}


