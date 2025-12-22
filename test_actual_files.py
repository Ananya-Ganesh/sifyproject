#!/usr/bin/env python3
"""Test the actual files with the backend"""
import sys
import os
sys.path.insert(0, r'c:\Users\Administrator\Desktop\PO')
os.chdir(r'c:\Users\Administrator\Desktop\PO')

from po_frontend_adapter import compare_for_frontend

file1 = r'c:\Users\Administrator\Downloads\Company_PO_BookstoreCH.docx'
file2 = r'c:\Users\Administrator\Downloads\Customer_Purchase_OrderCH.docx'

print("Testing with actual files...")
result = compare_for_frontend(file1, file2)

print("\nRESULT SUMMARY:")
print(f"  Conflicts: {result['summary']['conflict_count']}")
print(f"  Order Similarity: {result['summary']['order_similarity']}")
print(f"  Grand Company: {result['summary']['grand_company']}")
print(f"  Grand Customer: {result['summary']['grand_customer']}")
print(f"  Warning: {result['summary'].get('warning')}")
print(f"\nCurrency Validation:")
cv = result['summary'].get('currency_validation', {})
print(f"  Status: {cv.get('status')}")
print(f"  Company: {cv.get('company_currency')}")
print(f"  Customer: {cv.get('customer_currency')}")
print(f"  Warnings: {cv.get('warnings')}")
