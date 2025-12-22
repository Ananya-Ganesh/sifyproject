## PO Comparison AI Tool (Python, PDF)

This is a minimal prototype of an AI-assisted purchase order (PO) comparison tool.
It accepts **two PDF POs**, extracts line items, uses **semantic similarity** to align
items between the two documents (even with different wording), and then flags
**price and charge-type conflicts**.

### Features

- Upload two PDF POs (company vs customer).
- Extract basic line items from each PDF.
- Use `sentence-transformers` embeddings to detect when two descriptions refer
  to the **same product**, even with slightly different wording.
- Detect:
  - Unit price mismatches.
  - One-time vs recurring charge conflicts.

> Note: The extraction logic is intentionally simple and intended as a starting
> point. Real POs vary a lot in format; you will likely customize `po_parser.py`
> for your own layouts.

### Installation

1. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows PowerShell
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the API

Start the FastAPI server with:

```bash
uvicorn app:app --reload
```

By default, it listens on `http://127.0.0.1:8000`.

### API Usage

**Endpoint:** `POST /compare-pos`

**Body (multipart/form-data):**

- `po_a`: PDF file (company PO).
- `po_b`: PDF file (customer PO).

You can test it with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/compare-pos" ^
  -F "po_a=@path\to\company_po.pdf" ^
  -F "po_b=@path\to\customer_po.pdf"
```

The response will contain:

- Extracted items from each PO.
- Match similarity scores.
- A list of conflicts per matched line (price differences, charge type, etc.).

### Next Steps / Customization

- Improve `po_parser.py` to match your exact PO layouts and tables.
- Add your own mapping of product name aliases (e.g., `"natraj pencils"` vs
  `"nat pencils"`).
- Add a small web UI for uploading PDFs and viewing differences in a table.



