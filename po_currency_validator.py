"""
Currency Detection and Validation Module

Detects currency symbols/codes in PO documents and validates currency consistency
before price comparison. Provides warnings for currency mismatches.
"""

import re
from typing import Dict, Tuple, Optional, List


class CurrencyValidator:
    """Detects and validates currencies in PO documents."""
    
    # Currency symbol and code mappings
    CURRENCY_SYMBOLS = {
        '$': 'USD',
        '₹': 'INR',
        'Rs': 'INR',
        'Rs.': 'INR',
        '£': 'GBP',
        '€': 'EUR',
        '¥': 'JPY',
    }
    
    CURRENCY_CODES = {
        'USD': '$',
        'INR': '₹',
        'GBP': '£',
        'EUR': '€',
        'JPY': '¥',
    }
    
    def __init__(self):
        pass
    
    def extract_currency_from_text(self, text: str, focus_section: str = "Grand Total") -> Optional[Tuple[str, str]]:
        """
        Extract currency symbol and code from text, prioritizing sections near Grand Total.
        
        Args:
            text: Full document text
            focus_section: Section keyword to search near (default: "Grand Total")
        
        Returns:
            Tuple of (symbol, code) or None if no currency detected.
            Example: ('$', 'USD') or ('₹', 'INR')
        """
        if not text:
            return None
        
        # Debug: check what characters are in the text
        has_symbols = any(s in text for s in self.CURRENCY_SYMBOLS.keys())
        if has_symbols:
            print(f"[CURRENCY] Found currency symbols in text")
            for symbol in self.CURRENCY_SYMBOLS.keys():
                if symbol in text:
                    print(f"  Found: {symbol} ({self.CURRENCY_SYMBOLS[symbol]})")
        
        # Try to find currency near the focus section first
        lines = text.split('\n')
        focus_context = []
        
        for i, line in enumerate(lines):
            if focus_section.lower() in line.lower():
                # Grab lines around the focus section (5 lines before and after)
                start = max(0, i - 5)
                end = min(len(lines), i + 6)
                focus_context = lines[start:end]
                break
        
        # If focus section found, search there first
        if focus_context:
            context_text = '\n'.join(focus_context)
            currency = self._detect_currency(context_text)
            if currency:
                return currency
        
        # Fallback: search entire text
        return self._detect_currency(text)
    
    def _detect_currency(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Internal method to detect currency from text.
        
        Returns:
            Tuple of (symbol, code) or None
        """
        if not text:
            return None
            
        # Check for currency symbols (this is the primary detection method)
        for symbol, code in self.CURRENCY_SYMBOLS.items():
            if symbol in text:
                return (symbol, code)
        
        # Check for currency codes (case-insensitive, word boundary)
        text_upper = text.upper()
        for code in self.CURRENCY_CODES.keys():
            # Use word boundaries to avoid partial matches
            if re.search(r'\b' + code + r'\b', text_upper):
                symbol = self.CURRENCY_CODES[code]
                return (symbol, code)
        
        return None
    
    def validate_currency_match(
        self,
        company_po_text: str,
        customer_po_text: str,
        company_po_name: str = "Company PO",
        customer_po_name: str = "Customer PO"
    ) -> Dict[str, any]:
        """
        Validate currency consistency between two PO documents.
        
        Args:
            company_po_text: Full text content of company PO
            customer_po_text: Full text content of customer PO
            company_po_name: Name/path of company PO (for reporting)
            customer_po_name: Name/path of customer PO (for reporting)
        
        Returns:
            Dict with keys:
            - 'is_valid': bool (True if currencies match or can be reconciled)
            - 'company_currency': str or None (code like 'USD')
            - 'customer_currency': str or None (code like 'INR')
            - 'warnings': List[str] (empty if no issues, otherwise warning messages)
            - 'status': str ('OK', 'WARNING', 'CRITICAL')
        """
        result = {
            'is_valid': True,
            'company_currency': None,
            'customer_currency': None,
            'warnings': [],
            'status': 'OK',
        }
        
        # Extract currencies from both documents
        company_currency = self.extract_currency_from_text(company_po_text)
        customer_currency = self.extract_currency_from_text(customer_po_text)
        
        company_code = company_currency[1] if company_currency else None
        customer_code = customer_currency[1] if customer_currency else None
        
        result['company_currency'] = company_code
        result['customer_currency'] = customer_code
        
        # Case 1: Both currencies detected
        if company_code and customer_code:
            if company_code == customer_code:
                # Currencies match - all good
                result['is_valid'] = True
                result['status'] = 'OK'
                result['warnings'] = []
            else:
                # Currency mismatch - CRITICAL
                result['is_valid'] = False
                result['status'] = 'CRITICAL'
                warning_msg = (
                    f"[CRITICAL] Currency Mismatch Detected: "
                    f"{company_po_name} uses {company_code}, "
                    f"but {customer_po_name} uses {customer_code}. "
                    f"Price comparison may be invalid."
                )
                result['warnings'].append(warning_msg)
        
        # Case 2: Only one currency detected
        elif company_code and not customer_code:
            result['is_valid'] = True
            result['status'] = 'WARNING'
            warning_msg = (
                f"[INFO] Unverified Currency: "
                f"{customer_po_name} has no explicit currency marker. "
                f"Assuming {company_code} (from {company_po_name})."
            )
            result['warnings'].append(warning_msg)
            result['customer_currency'] = company_code  # Default to company currency
        
        elif customer_code and not company_code:
            result['is_valid'] = True
            result['status'] = 'WARNING'
            warning_msg = (
                f"[INFO] Unverified Currency: "
                f"{company_po_name} has no explicit currency marker. "
                f"Assuming {customer_code} (from {customer_po_name})."
            )
            result['warnings'].append(warning_msg)
            result['company_currency'] = customer_code  # Default to customer currency
        
        # Case 3: No currency detected in either document
        else:
            result['is_valid'] = True
            result['status'] = 'WARNING'
            warning_msg = (
                f"[INFO] No Currency Detected: "
                f"Neither {company_po_name} nor {customer_po_name} have explicit currency markers. "
                f"Assuming same currency for comparison."
            )
            result['warnings'].append(warning_msg)
        
        return result


def validate_currencies_before_comparison(
    company_po_text: str,
    customer_po_text: str,
    company_po_name: str = "Company PO",
    customer_po_name: str = "Customer PO"
) -> Dict[str, any]:
    """
    Convenience function to validate currencies before comparison.
    
    Call this before running the main comparison logic.
    
    Args:
        company_po_text: Raw text from company PO
        customer_po_text: Raw text from customer PO
        company_po_name: Label for company PO
        customer_po_name: Label for customer PO
    
    Returns:
        Validation result dict (see CurrencyValidator.validate_currency_match)
    """
    validator = CurrencyValidator()
    return validator.validate_currency_match(
        company_po_text,
        customer_po_text,
        company_po_name,
        customer_po_name
    )
