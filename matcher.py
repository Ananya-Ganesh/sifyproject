from __future__ import annotations

from dataclasses import asdict
from typing import List, Dict, Any, Optional

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from po_parser import POItem


_vectorizer: Optional[TfidfVectorizer] = None


def _build_vectorizer(corpus: List[str]) -> TfidfVectorizer:
    global _vectorizer
    _vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    _vectorizer.fit(corpus)
    return _vectorizer


def match_items(
    items_a: List[POItem],
    items_b: List[POItem],
    threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    """
    For each item in A, find the best match in B using TF-IDF cosine similarity
    plus a fuzzy score. No heavy Torch/transformer dependencies.
    Returns a list of dicts: {"item_a", "item_b", "similarity"}.
    """
    if not items_a:
        return []

    descs_a = [it.normalized_description for it in items_a]
    descs_b = [it.normalized_description for it in items_b] if items_b else []

    results: List[Dict[str, Any]] = []

    if not descs_b:
        for it in items_a:
            results.append({"item_a": it, "item_b": None, "similarity": 0.0})
        return results

    # Build a shared TF-IDF space on all descriptions
    corpus = descs_a + descs_b
    vectorizer = _build_vectorizer(corpus)

    tfidf_a = vectorizer.transform(descs_a)
    tfidf_b = vectorizer.transform(descs_b)

    cos_matrix = cosine_similarity(tfidf_a, tfidf_b)

    for idx_a, row in enumerate(cos_matrix):
        idx_b = int(row.argmax())
        cos_sim = float(row[idx_b])

        # Combine cosine similarity with multiple fuzzy scores for better semantic matching
        # token_set_ratio: handles word order variations ("pencil hb" vs "hb pencil")
        # partial_ratio: handles shortforms ("long size scale" vs "long scale")
        fuzzy_token = fuzz.token_set_ratio(descs_a[idx_a], descs_b[idx_b]) / 100.0
        fuzzy_partial = fuzz.partial_ratio(descs_a[idx_a], descs_b[idx_b]) / 100.0
        fuzzy_ratio = fuzz.ratio(descs_a[idx_a], descs_b[idx_b]) / 100.0
        
        # Weighted combination: token_set is best for word order, partial for shortforms
        fuzzy_score = (fuzzy_token * 0.5 + fuzzy_partial * 0.3 + fuzzy_ratio * 0.2)
        combined = (cos_sim * 0.4 + fuzzy_score * 0.6)  # Give more weight to fuzzy matching

        item_b = items_b[idx_b] if combined >= threshold else None
        results.append(
            {
                "item_a": items_a[idx_a],
                "item_b": item_b,
                "similarity": round(combined, 3),
            }
        )

    return results


def compare_prices(
    item_a: POItem,
    item_b: Optional[POItem],
    similarity: float,
    price_tolerance: float = 0.01,
) -> Dict[str, Any]:
    """
    Compare two matched items for price/total and charge-type conflicts.
    """
    if item_b is None:
        return {
            "status": "missing_in_b",
            "similarity": similarity,
            "conflicts": [],
        }

    conflicts = []
    
    # Check if this is a metadata line (PO Number, Date, etc.) - these don't have prices
    desc_a_lower = item_a.raw_description.lower()
    desc_b_lower = item_b.raw_description.lower() if item_b else ""
    is_metadata = any(kw in desc_a_lower or kw in desc_b_lower for kw in ['po no:', 'po number:', 'date:', 'order date:', 'order ref:', 'reference:'])
    
    # For metadata lines, only check if they match semantically (no price conflicts)
    if is_metadata:
        # Metadata lines are OK if they match semantically (similarity check)
        # No price conflicts for metadata
        return {
            "status": "ok" if similarity > 0.4 else "conflict",
            "similarity": similarity,
            "conflicts": [],
            "item_a": asdict(item_a),
            "item_b": asdict(item_b),
        }

    # Quantity conflict
    if (
        item_a.quantity is not None
        and item_b.quantity is not None
        and abs(item_a.quantity - item_b.quantity) > price_tolerance
    ):
        conflicts.append(
            {
                "field": "quantity",
                "a": item_a.quantity,
                "b": item_b.quantity,
            }
        )

    # Unit price conflict
    if (
        item_a.unit_price is not None
        and item_b.unit_price is not None
        and abs(item_a.unit_price - item_b.unit_price) > price_tolerance
    ):
        conflicts.append(
            {
                "field": "unit_price",
                "a": item_a.unit_price,
                "b": item_b.unit_price,
            }
        )

    # Total price conflict (if available)
    if (
        item_a.total_price is not None
        and item_b.total_price is not None
        and abs(item_a.total_price - item_b.total_price) > price_tolerance
    ):
        conflicts.append(
            {
                "field": "total_price",
                "a": item_a.total_price,
                "b": item_b.total_price,
            }
        )

    if item_a.charge_type != item_b.charge_type:
        conflicts.append(
            {
                "field": "charge_type",
                "a": item_a.charge_type,
                "b": item_b.charge_type,
            }
        )

    if item_a.recurrence != item_b.recurrence:
        conflicts.append(
            {
                "field": "recurrence",
                "a": item_a.recurrence,
                "b": item_b.recurrence,
            }
        )

    return {
        "status": "conflict" if conflicts else "ok",
        "similarity": similarity,
        "conflicts": conflicts,
        "item_a": asdict(item_a),
        "item_b": asdict(item_b),
    }


