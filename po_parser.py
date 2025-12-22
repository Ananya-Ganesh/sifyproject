from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

import pdfplumber
import re

try:
    from word2number import w2n
except ImportError:
    w2n = None


@dataclass
class POItem:
    """Represents a single line item in a PO."""

    line_id: str
    raw_description: str
    normalized_description: str
    quantity: Optional[float]
    unit_price: Optional[float]
    total_price: Optional[float]
    charge_type: str  # "one_time" or "recurring" or "unknown"
    recurrence: Optional[str]  # e.g. "annual", "monthly", None


# Pattern to match numbers - handles both comma-separated thousands and decimals
# Matches: "25,000", "25000", "25.50", "100", etc.
# IMPORTANT: Uses word boundaries and does NOT match spaces between numbers
# This ensures "200 10 2000" extracts as [200, 10, 2000], not [200102000]
PRICE_PATTERN = re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\b|\b\d+\.\d{1,2}\b')

# Common number word patterns (e.g., "twenty five thousand", "one hundred")
NUMBER_WORD_PATTERN = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|lakh|crore)\b",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _normalize_date(date_str: str) -> Optional[str]:
    """
    Normalize various date formats to a standard format (YYYY-MM-DD).
    Handles formats like:
    - "15-12-2025", "15/12/2025", "15.12.2025" -> "2025-12-15"
    - "15-Dec-2025", "15 December 2025" -> "2025-12-15"
    - "15.12.25" -> "2025-12-15" (assumes 20xx for 2-digit years)
    Returns None if not a valid date.
    """
    date_str = date_str.strip()
    
    # Month name mapping
    month_names = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }
    
    # Try different date formats
    formats = [
        # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
        (r'(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})', lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3)))),
        # DD-MM-YY, DD/MM/YY, DD.MM.YY (assume 20xx)
        (r'(\d{1,2})[-/.](\d{1,2})[-/.](\d{2})', lambda m: (int(m.group(1)), int(m.group(2)), 2000 + int(m.group(3)))),
        # DD-Mon-YYYY, DD Mon YYYY, DD-Month-YYYY
        (r'(\d{1,2})[- ]([a-z]+)[- ](\d{4})', lambda m: (int(m.group(1)), month_names.get(m.group(2).lower(), 0), int(m.group(3)))),
        # DD-Mon-YY, DD Mon YY
        (r'(\d{1,2})[- ]([a-z]+)[- ](\d{2})', lambda m: (int(m.group(1)), month_names.get(m.group(2).lower(), 0), 2000 + int(m.group(3)))),
    ]
    
    for pattern, parser in formats:
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            try:
                day, month, year = parser(match)
                if 1 <= month <= 12 and 1 <= day <= 31 and year >= 1900:
                    # Validate the date
                    try:
                        dt = datetime(year, month, day)
                        return dt.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
            except (ValueError, AttributeError):
                continue
    
    return None


def _extract_and_normalize_dates(text: str) -> str:
    """
    Find dates in text and normalize them to standard format.
    This helps matching recognize that "15-12-2025" and "15-Dec-2025" are the same.
    """
    # Find date patterns and replace with normalized format
    date_patterns = [
        r'\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}',  # DD-MM-YYYY, DD/MM/YYYY, etc.
        r'\d{1,2}[- ][a-z]+[- ]\d{2,4}',      # DD-Mon-YYYY, DD Mon YYYY
    ]
    
    result = text
    for pattern in date_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            date_str = match.group(0)
            normalized = _normalize_date(date_str)
            if normalized:
                # Replace with normalized format
                result = result.replace(date_str, normalized)
    
    return result


def _words_to_number(text: str) -> Optional[float]:
    """
    Convert number words to numeric value.
    Examples: "twenty five thousand" -> 25000, "one hundred" -> 100
    Handles phrases like "Twenty Five Thousand Rupees Only" -> 25000
    """
    if not w2n:
        return None
    
    text_lower = text.lower().strip()
    # Remove common suffixes like "rupees only", "only", etc. that don't affect the number
    text_lower = re.sub(r"\b(rupees?|only|rs\.?|and)\b", "", text_lower, flags=re.IGNORECASE).strip()
    # Clean up extra spaces
    text_lower = re.sub(r"\s+", " ", text_lower).strip()
    
    # Try to extract number word phrases (e.g., "twenty five thousand")
    # Common patterns for Indian numbering
    patterns = [
        r"twenty\s*five\s*thousand",
        r"twenty\s*five\s*lakh",
        r"twenty\s*thousand",
        r"thirty\s*thousand",
        r"forty\s*thousand",
        r"fifty\s*thousand",
        r"sixty\s*thousand",
        r"seventy\s*thousand",
        r"eighty\s*thousand",
        r"ninety\s*thousand",
        r"one\s*hundred\s*thousand",
        r"two\s*hundred\s*thousand",
        r"three\s*hundred\s*thousand",
        r"four\s*hundred\s*thousand",
        r"five\s*hundred\s*thousand",
        r"one\s*lakh",
        r"two\s*lakh",
        r"three\s*lakh",
        r"four\s*lakh",
        r"five\s*lakh",
        r"ten\s*lakh",
        r"twenty\s*lakh",
        r"fifty\s*lakh",
        r"one\s*crore",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            phrase = match.group(0)
            try:
                num = w2n.word_to_num(phrase)
                return float(num)
            except (ValueError, AttributeError):
                continue
    
    # Fallback: try converting the whole cleaned text
    try:
        num = w2n.word_to_num(text_lower)
        return float(num)
    except (ValueError, AttributeError):
        pass
    
    return None


def _normalize_numbers_in_text(text: str) -> str:
    """
    Replace number words with their numeric equivalents in text for better matching.
    Example: "Twenty Five Thousand Rupees Only" -> "25000 Rupees Only"
    This helps semantic matching recognize that "25,000" and "twenty five thousand" are the same.
    """
    num_value = _words_to_number(text)
    if num_value is not None:
        # Find the number word phrase and replace it with the numeric value
        text_lower = text.lower()
        patterns = [
            (r"twenty\s*five\s*thousand", "25000"),
            (r"twenty\s*five\s*lakh", "2500000"),
            (r"twenty\s*thousand", "20000"),
            (r"thirty\s*thousand", "30000"),
            (r"forty\s*thousand", "40000"),
            (r"fifty\s*thousand", "50000"),
            (r"sixty\s*thousand", "60000"),
            (r"seventy\s*thousand", "70000"),
            (r"eighty\s*thousand", "80000"),
            (r"ninety\s*thousand", "90000"),
            (r"one\s*hundred\s*thousand", "100000"),
            (r"two\s*hundred\s*thousand", "200000"),
            (r"three\s*hundred\s*thousand", "300000"),
            (r"four\s*hundred\s*thousand", "400000"),
            (r"five\s*hundred\s*thousand", "500000"),
            (r"one\s*lakh", "100000"),
            (r"two\s*lakh", "200000"),
            (r"three\s*lakh", "300000"),
            (r"four\s*lakh", "400000"),
            (r"five\s*lakh", "500000"),
            (r"ten\s*lakh", "1000000"),
            (r"twenty\s*lakh", "2000000"),
            (r"fifty\s*lakh", "5000000"),
            (r"one\s*crore", "10000000"),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                break
    
    return text


def _guess_charge_type(line: str) -> str:
    l = line.lower()
    if any(k in l for k in ["one time", "one-time", "setup"]):
        return "one_time"
    if any(k in l for k in ["annual", "yearly", "per year"]):
        return "recurring"
    if any(k in l for k in ["monthly", "per month", "quarterly"]):
        return "recurring"
    return "unknown"


def _guess_recurrence(line: str) -> Optional[str]:
    l = line.lower()
    if any(k in l for k in ["annual", "yearly", "per year"]):
        return "annual"
    if any(k in l for k in ["monthly", "per month"]):
        return "monthly"
    if "quarterly" in l:
        return "quarterly"
    return None


def _extract_numbers(line: str) -> List[float]:
    """
    Return a list of numeric values found in the line (in order).
    Extracts both digit numbers (e.g., "25,000") and word numbers (e.g., "twenty five thousand").
    Properly handles commas as thousands separators (not decimal separators).
    IMPORTANT: Extracts numbers in the order they appear, properly separated.
    """
    nums: List[float] = []
    
    # Split line by whitespace to get individual tokens
    # This ensures "200 10 2000" extracts as [200, 10, 2000], not concatenated
    tokens = line.split()
    
    for token in tokens:
        # Remove any non-numeric characters except comma and dot
        # This handles cases like "200,", "10.", etc.
        clean_token = re.sub(r'[^\d,.]', '', token)
        
        if not clean_token or not re.match(r'^\d', clean_token):
            continue
        
        # Handle comma: determine if thousands separator or decimal
        val = clean_token
        if "," in val:
            parts = val.split(",")
            if len(parts) == 2:
                if len(parts[1]) == 3:
                    # Thousands separator: "25,000" -> "25000"
                    val = val.replace(",", "")
                elif len(parts[1]) <= 2:
                    # Decimal separator: "25,50" -> "25.50"
                    val = val.replace(",", ".")
                else:
                    val = val.replace(",", "")
            else:
                # Multiple commas: "1,25,000" -> remove all
                val = val.replace(",", "")
        
        try:
            num_val = float(val)
            nums.append(num_val)
        except ValueError:
            continue
    
    # Also try regex pattern as fallback for numbers with spaces (shouldn't be needed but just in case)
    if not nums:
        number_matches = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\b|\b\d+\.\d{1,2}\b', line)
        for match in number_matches:
            val = match.strip().replace(" ", "").replace("\t", "")
            if "," in val:
                parts = val.split(",")
                if len(parts) == 2 and len(parts[1]) == 3:
                    val = val.replace(",", "")
                elif len(parts) == 2 and len(parts[1]) <= 2:
                    val = val.replace(",", ".")
                else:
                    val = val.replace(",", "")
            try:
                num_val = float(val)
                nums.append(num_val)
            except ValueError:
                continue
    
    # Then, try to extract number words (e.g., "twenty five thousand")
    word_num = _words_to_number(line)
    if word_num is not None:
        # Check if this number word value is already represented in the digit numbers
        is_duplicate = any(abs(n - word_num) < 0.01 for n in nums)
        if not is_duplicate:
            nums.append(word_num)
    
    return nums


def extract_items_from_pdf(path: str) -> List[POItem]:
    """
    Comprehensive extraction from PDF:
    - First tries to extract from tables (if PDF has table structure)
    - Falls back to line-by-line text extraction
    - Handles both digit numbers and number words
    """
    items: List[POItem] = []

    with pdfplumber.open(path) as pdf:
        line_counter = 1
        
        # First, try to extract from tables (many POs are in table format)
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    # Skip header rows (usually first row)
                    for row_idx, row in enumerate(table):
                        if row_idx == 0:
                            continue  # Skip header
                        
                        # Combine row cells into a single line
                        row_text = " ".join(str(cell) if cell else "" for cell in row)
                        row_text = _clean_text(row_text)
                        
                        if not row_text or len(row_text.strip()) < 5:
                            continue
                        
                        # Process this table row as a line item
                        item = _process_line_as_item(row_text, line_counter)
                        if item:
                            items.append(item)
                            line_counter += 1
        
        # If no tables found, fall back to line-by-line text extraction
        if not items:
            for page in pdf.pages:
                text = page.extract_text() or ""
                for raw_line in text.splitlines():
                    line = _clean_text(raw_line)
                    if not line:
                        continue
                    
                    item = _process_line_as_item(line, line_counter)
                    if item:
                        items.append(item)
                        line_counter += 1
    
    return items


def _process_line_as_item(line: str, line_counter: int) -> Optional[POItem]:
    """
    Process a single line and extract it as a PO item if it contains product/price info.
    Returns POItem if valid, None otherwise.
    """

    # Allow metadata lines (PO Number, Date) but skip totals
    line_lower = line.lower()
    if any(keyword in line_lower for keyword in ['total:', 'grand total:', 'subtotal:', 'sum:']):
        return None
    
    # For metadata lines (PO Number, Date), extract them but with special handling
    is_metadata = any(keyword in line_lower for keyword in ['date:', 'order date:', 'po no:', 'po number:', 'order ref:', 'reference:'])
    
    # Skip very short lines (likely headers/footers)
    if len(line.strip()) < 5:
        return None
    
    # Extract numbers (both digits and words)
    nums = _extract_numbers(line)
    
    # Also check if line contains number words even if no digit numbers found
    has_number_words = _words_to_number(line) is not None
    
    # A line item should have at least one number (digit or word)
    # OR contain product-like keywords
    # OR be a metadata line (PO Number, Date)
    product_keywords = ['book', 'pen', 'pencil', 'eraser', 'box', 'sharpener', 'highlighter', 'marker', 'notebook', 'drawing', 'item', 'product', 'description', 'science', 'rays']
    has_product_keyword = any(kw in line_lower for kw in product_keywords)
    
    # Allow metadata lines even without numbers (they might just have text)
    if not is_metadata and not nums and not has_number_words and not has_product_keyword:
        return None
    
    # For metadata lines without numbers, create empty number list (they'll be matched by text)
    if is_metadata and not nums:
        nums = []
    
    # If no digit numbers but has number words, create a placeholder
    if not nums and has_number_words:
        word_num = _words_to_number(line)
        if word_num:
            nums = [word_num]

    # Remove serial numbers: if first number is a single digit (1-9) and there are 4+ numbers,
    # it's likely a serial number, not quantity
    # But skip this for metadata lines
    if not is_metadata and len(nums) >= 4 and nums[0] < 10 and nums[0] == int(nums[0]):
        nums = nums[1:]

    # Heuristic for assigning numbers based on typical PO format: "Product Qty Unit Total"
    # For metadata lines, don't assign prices
    quantity = None
    unit_price = None
    total_price = None

    if is_metadata:
        # Metadata lines don't have prices - leave them as None
        pass
    elif len(nums) == 1:
        unit_price = nums[0]
    elif len(nums) == 2:
        unit_price = nums[0]
        total_price = nums[1]
    elif len(nums) == 3:
        quantity = nums[0]
        unit_price = nums[1]
        total_price = nums[2]
    elif len(nums) >= 4:
        quantity = nums[-3]
        unit_price = nums[-2]
        total_price = nums[-1]

    charge_type = _guess_charge_type(line)
    recurrence = _guess_recurrence(line)

    # Normalize description for better semantic matching
    normalized_desc = line.lower()
    normalized_desc = re.sub(r'^\d+\s+', '', normalized_desc)
    normalized_desc = _extract_and_normalize_dates(normalized_desc)
    normalized_desc = _normalize_numbers_in_text(normalized_desc)
    normalized_desc = re.sub(r'[^\w\s-]', ' ', normalized_desc)
    normalized_desc = ' '.join(normalized_desc.split())

    return POItem(
        line_id=f"L{line_counter}",
        raw_description=line,
        normalized_description=normalized_desc,
        quantity=quantity,
        unit_price=unit_price,
        total_price=total_price,
        charge_type=charge_type,
        recurrence=recurrence,
    )


