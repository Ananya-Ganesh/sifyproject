#!/usr/bin/env python
"""Direct test of the adapter output"""
import sys
import os
import json
sys.path.insert(0, r'c:\Users\Administrator\Desktop\PO')
os.chdir(r'c:\Users\Administrator\Desktop\PO')

from po_frontend_adapter import compare_for_frontend
import glob

# Find the DOCX files
docx_files = sorted(glob.glob(r'*.docx'))
print(f"Found files: {docx_files}")

if len(docx_files) >= 2:
    file1 = docx_files[0]
    file2 = docx_files[1]
    
    print(f"\nComparing:")
    print(f"  {file1}")
    print(f"  {file2}")
    print()
    
    result = compare_for_frontend(file1, file2)
    
    print("SUMMARY:")
    print(json.dumps(result.get('summary', {}), indent=2, default=str))
    
    print("\nRESULTS (showing just status and conflicts):")
    for i, r in enumerate(result.get('results', []), 1):
        print(f"  Row {i}: status={r.get('status')}, conflicts={r.get('conflicts')}")
