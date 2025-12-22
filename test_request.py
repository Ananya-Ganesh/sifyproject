import json
import requests


def main():
    # Update these paths if your PDFs move to a different folder
    po_a_path = r"C:\PO sample\Company_PO_Bookstore.pdf"
    po_b_path = r"C:\PO sample\Customer_Purchase_Order.pdf"

    files = {
        "po_a": open(po_a_path, "rb"),
        "po_b": open(po_b_path, "rb"),
    }

    try:
        resp = requests.post("http://127.0.0.1:8000/compare-pos", files=files)
    finally:
        for f in files.values():
            f.close()

    print("HTTP status code:", resp.status_code)

    if resp.status_code != 200:
        print("Server returned an error.")
        print(resp.text)
        return

    data = json.loads(resp.text)

    summary = data.get("summary", {})
    conflict_count = summary.get("conflict_count", 0)
    total_a = summary.get("total_items_a", 0)
    total_b = summary.get("total_items_b", 0)

    print()
    print("===== SIMPLE RESULT =====")
    print(f"Items in company PO:  {total_a}")
    print(f"Items in customer PO: {total_b}")
    print(f"Conflicts found:      {conflict_count}")

    print()
    print("Each line item status (company vs customer):")
    for i, r in enumerate(data.get("results", []), start=1):
        status = r.get("status")
        sim = r.get("similarity")
        item_a = r.get("item_a", {})
        item_b = r.get("item_b", {})
        conflicts = r.get("conflicts", [])
        
        line_id = item_a.get("line_id") or f"Line {i}"
        product_desc_a = item_a.get("raw_description") or item_a.get("normalized_description") or "Unknown product"
        product_desc_b = ""
        if item_b:
            product_desc_b = item_b.get("raw_description") or item_b.get("normalized_description") or "Unknown product"
        else:
            product_desc_b = "(no match)"

        print(f"{line_id}: status = {status}, similarity = {sim}")
        print(f"  Company product:   {product_desc_a[:80]}...")
        print(f"  Customer product:  {product_desc_b[:80]}...")
        print(f"  Company qty:       {item_a.get('quantity')}")
        print(f"  Customer qty:      {item_b.get('quantity') if item_b else None}")
        print(f"  Company unit:      {item_a.get('unit_price')}")
        print(f"  Customer unit:     {item_b.get('unit_price') if item_b else None}")
        print(f"  Company total:     {item_a.get('total_price')}")
        print(f"  Customer total:    {item_b.get('total_price') if item_b else None}")
        
        # If there are conflicts, show them clearly
        if conflicts:
            print(f"    ⚠️  CONFLICTS DETECTED:")
            for conflict in conflicts:
                field = conflict.get("field", "unknown")
                val_a = conflict.get("a")
                val_b = conflict.get("b")
                
                if field == "unit_price":
                    print(f"      • PRICE MISMATCH:")
                    print(f"        Company PO:  ${val_a}")
                    print(f"        Customer PO: ${val_b}")
                    print(f"        Difference:  ${abs(val_a - val_b):.2f}")
                elif field == "charge_type":
                    print(f"      • CHARGE TYPE MISMATCH:")
                    print(f"        Company PO:  {val_a}")
                    print(f"        Customer PO: {val_b}")
                elif field == "recurrence":
                    print(f"      • RECURRENCE MISMATCH:")
                    print(f"        Company PO:  {val_a}")
                    print(f"        Customer PO: {val_b}")
                else:
                    print(f"      • {field.upper()} MISMATCH:")
                    print(f"        Company PO:  {val_a}")
                    print(f"        Customer PO: {val_b}")
        
        # If item is missing in PO B
        if status == "missing_in_b":
            print(f"    ⚠️  ITEM NOT FOUND in Customer PO")
        
        print()  # Empty line between items


if __name__ == "__main__":
    main()



