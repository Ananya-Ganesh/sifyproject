#!/usr/bin/env python3
"""
Quick test of currency validation in the adapter
"""
import json
from po_frontend_adapter import compare_for_frontend

# Test with the DOCX files you showed (with ₹ and $ currencies)
company_file = "Company_P_toreCH.docx"  # Update with actual path if needed
customer_file = "Customer_P_rderCH.docx"  # Update with actual path if needed

try:
    result = compare_for_frontend(company_file, customer_file)
    
    print("=" * 80)
    print("COMPARISON RESULT")
    print("=" * 80)
    print(json.dumps(result["summary"], indent=2))
    
    print("\n" + "=" * 80)
    print("CURRENCY VALIDATION DETAILS")
    print("=" * 80)
    currency_val = result["summary"].get("currency_validation", {})
    print(f"Status: {currency_val.get('status')}")
    print(f"Company Currency: {currency_val.get('company_currency')}")
    print(f"Customer Currency: {currency_val.get('customer_currency')}")
    print(f"Warnings: {currency_val.get('warnings', [])}")
    
    print("\n" + "=" * 80)
    print("WARNING MESSAGE")
    print("=" * 80)
    print(result["summary"].get("warning", "No warning"))
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
