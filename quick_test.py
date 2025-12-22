import sys
import os
sys.path.insert(0, r'c:\Users\Administrator\Desktop\PO')
os.chdir(r'c:\Users\Administrator\Desktop\PO')

# Test the currency validation directly
from po_currency_validator import CurrencyValidator

validator = CurrencyValidator()

# Test with sample text containing different currencies
company_text = "rays science book ₹25,000 Grand Total ₹25,000"
customer_text = "rays science $25,000 Grand Total $25,000"

print("=" * 80)
print("TEST 1: Rupee vs Dollar")
print("=" * 80)
print(f"Company text: {company_text}")
print(f"Customer text: {customer_text}")
print()

result = validator.validate_currency_match(company_text, customer_text, "Company.docx", "Customer.docx")
print(f"Result status: {result.get('status')}")
print(f"Result: {result}")
print()

# Now test with the actual files
print("=" * 80)
print("TEST 2: Actual DOCX files")
print("=" * 80)

from po_auditor import FileExtractor

extractor = FileExtractor()

# List files to find the actual ones
import glob
docx_files = glob.glob(r'c:\Users\Administrator\Desktop\PO\*.docx')
print(f"Found DOCX files: {docx_files}")
print()

if len(docx_files) >= 2:
    file1 = docx_files[0]
    file2 = docx_files[1]
    
    text1, _ = extractor.extract(file1)
    text2, _ = extractor.extract(file2)
    
    print(f"File 1: {file1}")
    print(f"Text (first 300 chars): {text1[:300]}")
    print()
    print(f"File 2: {file2}")
    print(f"Text (first 300 chars): {text2[:300]}")
    print()
    
    result = validator.validate_currency_match(text1, text2, os.path.basename(file1), os.path.basename(file2))
    print(f"Validation result: {result}")
