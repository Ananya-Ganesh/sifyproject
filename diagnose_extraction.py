#!/usr/bin/env python3
"""Diagnostic script to show exactly what's being extracted from DOCX files"""
import sys
import os
sys.path.insert(0, r'c:\Users\Administrator\Desktop\PO')
os.chdir(r'c:\Users\Administrator\Desktop\PO')

from po_auditor import FileExtractor

file1 = r'c:\Users\Administrator\Downloads\Company_PO_BookstoreCH.docx'
file2 = r'c:\Users\Administrator\Downloads\Customer_Purchase_OrderCH.docx'

extractor = FileExtractor()

print("=" * 80)
print("FILE 1 EXTRACTION")
print("=" * 80)
text1, tables1 = extractor.extract(file1)
print(f"Extracted text length: {len(text1)}")
print(f"Number of tables: {len(tables1)}")
print(f"\nFULL TEXT:\n{text1}")
print(f"\nTABLES: {tables1}")

print("\n" + "=" * 80)
print("FILE 2 EXTRACTION")
print("=" * 80)
text2, tables2 = extractor.extract(file2)
print(f"Extracted text length: {len(text2)}")
print(f"Number of tables: {len(tables2)}")
print(f"\nFULL TEXT:\n{text2}")
print(f"\nTABLES: {tables2}")

print("\n" + "=" * 80)
print("CHARACTER ANALYSIS")
print("=" * 80)
print(f"File 1 contains '$': {'$' in text1}")
print(f"File 1 contains '₹': {chr(0x20b9) in text1}")
print(f"File 2 contains '$': {'$' in text2}")
print(f"File 2 contains '₹': {chr(0x20b9) in text2}")

# Look for patterns like "25000" or numbers
import re
numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text1 + text2)
print(f"\nNumbers found in files: {set(numbers)}")
