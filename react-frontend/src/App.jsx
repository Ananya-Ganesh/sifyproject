import { useState } from "react";
import "./App.css";

function App() {
  const [poA, setPoA] = useState(null);
  const [poB, setPoB] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [summary, setSummary] = useState(null);
  const [results, setResults] = useState([]);
  const [warning, setWarning] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!poA || !poB) {
      setError("Please select both files.");
      return;
    }
    setError(null);
    setLoading(true);

    const form = new FormData();
    form.append("po_a", poA);
    form.append("po_b", poB);

    try {
      const resp = await fetch("/compare-pos", {
        method: "POST",
        body: form,
      });
      if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
      const data = await resp.json();
      setSummary(data.summary || {});
      setResults(data.results || []);
      setWarning((data.summary && data.summary.warning) || null);
    } catch (err) {
      setError(err.message || "Failed to fetch.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="shell">
      <h1>PO Comparison</h1>
      <p className="sub">
        Upload company and customer PDFs to compare items, prices, and charge types.
      </p>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="row">
            <div className="col">
              <label className="label">Company PO (PDF)</label>
              <input
                type="file"
                accept=".pdf,.png,.jpg,.jpeg,.docx,.xlsx"
                onChange={(e) => setPoA(e.target.files?.[0] || null)}
              />
            </div>
            <div className="col">
              <label className="label">Customer PO (PDF)</label>
              <input
                type="file"
                accept=".pdf,.png,.jpg,.jpeg,.docx,.xlsx"
                onChange={(e) => setPoB(e.target.files?.[0] || null)}
              />
            </div>
            <div style={{ display: "flex", alignItems: "flex-end" }}>
              <button type="submit" disabled={loading}>
                {loading ? "Comparing..." : "Compare"}
              </button>
            </div>
          </div>
        </form>
      </div>

      {error && <div className="error">{error}</div>}
      {warning && <div className="warning">{warning}</div>}

      {summary && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="summary-grid">
            <div className="pill">
              <b>Items (Company)</b>
              {summary.total_items_a ?? 0}
            </div>
            <div className="pill">
              <b>Items (Customer)</b>
              {summary.total_items_b ?? 0}
            </div>
            <div className="pill">
              <b>Matched</b>
              {summary.matched_items ?? 0}
            </div>
            <div className="pill">
              <b>Conflicts</b>
              <span
                className="badge"
                style={{
                  background:
                    (summary.conflict_count ?? 0) > 0 ? "#2b0f13" : "#0f2b1c",
                  color:
                    (summary.conflict_count ?? 0) > 0 ? "#ffb2b2" : "#96f2c7",
                }}
              >
                {summary.conflict_count ?? 0}
              </span>
            </div>
            <div className="pill">
              <b>Order similarity</b>
              {(summary.order_similarity !== undefined
                ? summary.order_similarity.toFixed(2)
                : "n/a")}
            </div>
          </div>
        </div>
      )}

      {results.length > 0 && (
        <div className="card" style={{ marginTop: 12 }}>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Status</th>
                <th>Similarity</th>
                <th>Company PO item</th>
                <th>Customer PO item</th>
                <th>Conflicts</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, idx) => {
                const itemA = r.item_a || {};
                const itemB = r.item_b || {};
                const conflicts = r.conflicts || [];
                const status = r.status;
                const cls =
                  status === "conflict"
                    ? "conflict"
                    : status === "missing_in_b"
                    ? "missing"
                    : "ok";
                const dot =
                  status === "conflict"
                    ? "status-conflict"
                    : status === "missing_in_b"
                    ? "status-missing"
                    : "status-ok";
                const descA =
                  itemA.raw_description ||
                  itemA.normalized_description ||
                  "(no description)";
                const descB =
                  itemB.raw_description ||
                  itemB.normalized_description ||
                  "(no match)";
                const conflictsHtml = conflicts.length
                  ? conflicts.map((c, i) => (
                      <li key={i}>
                        <strong>{c.field}</strong>: {c.a} vs {c.b}
                      </li>
                    ))
                  : status === "missing_in_b"
                  ? "Not found in customer PO"
                  : "None";

                return (
                  <tr key={idx}>
                    <td>{idx + 1}</td>
                    <td className={cls}>
                      <span className={`status-dot ${dot}`} />
                      {status}
                    </td>
                    <td>{r.similarity ?? ""}</td>
                    <td>{descA}</td>
                    <td>{descB}</td>
                    <td>
                      {Array.isArray(conflictsHtml) ? (
                        <ul>{conflictsHtml}</ul>
                      ) : (
                        conflictsHtml
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default App;
