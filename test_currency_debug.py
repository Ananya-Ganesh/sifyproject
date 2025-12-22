#!/usr/bin/env python3
"""Debug currency validation"""
import sys
sys.path.insert(0, r'c:\Users\Administrator\Desktop\PO')

from po_currency_validator import validate_currencies_before_comparison
from po_auditor import FileExtractor

# Test with your DOCX files
company_file = r"C:\Users\Administrator\Desktop\PO\Company_P_toreCH.docx"
customer_file = r"C:\Users\Administrator\Desktop\PO\Customer_P_rderCH.docx"

try:
    # Extract text
    extractor = FileExtractor()
    text_a, _ = extractor.extract(company_file)
    text_b, _ = extractor.extract(customer_file)
    
    print("=" * 80)
    print("COMPANY PO TEXT (first 500 chars):")
    print("=" * 80)
    print(text_a[:500])
    
    print("\n" + "=" * 80)
    print("CUSTOMER PO TEXT (first 500 chars):")
    print("=" * 80)
    print(text_b[:500])
    
    # Validate currencies
    result = validate_currencies_before_comparison(text_a, text_b, company_file, customer_file)
    
    print("\n" + "=" * 80)
    print("CURRENCY VALIDATION RESULT:")
    print("=" * 80)
    import json
    print(json.dumps(result, indent=2))
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
