import pandas as pd
import json
from po_engine import extract_raw_text, parse_to_unified_schema

pa = r'C:\Users\Administrator\Downloads\Company_PO_Excel_0723_003361.xlsx'
pb = r'C:\Users\Administrator\Downloads\Customer_PO_Excel_0723_003361.xlsx'

def load_tables(path):
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception:
        xls = pd.ExcelFile(path)
        sheets = {n: xls.parse(n) for n in xls.sheet_names}
    tables = []
    for name, df in sheets.items():
        if df is None or df.empty:
            continue
        headers = [str(h) for h in df.columns]
        tables.append(headers)
        for _, row in df.fillna("").iterrows():
            tables.append([str(v) if v is not None else "" for v in row.tolist()])
    return tables

for path in (pa, pb):
    print('---', path)
    txt = extract_raw_text(path)
    print('RAW_TEXT sample:')
    print('\n'.join(txt.splitlines()[:20]))
    tables = load_tables(path)
    print('First 10 table rows:')
    for r in tables[:10]:
        print(r)
    po = parse_to_unified_schema(txt, tables=tables, order_id=path)
    print('Parsed PO:')
    print(json.dumps({'order_id':po.order_id,'line_items':[li.__dict__ for li in po.line_items],'otc':po.otc,'arc':po.arc,'grand_total':po.grand_total}, indent=2))
