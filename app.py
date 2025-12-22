from __future__ import annotations

import os
import tempfile
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from po_frontend_adapter import compare_for_frontend


app = FastAPI(title="PO Comparison AI Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1] or ".pdf"
    print(f"[APP] Upload filename: {upload.filename}, suffix: {suffix}")
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as tmp:
        tmp.write(upload.file.read())
    return path


@app.post("/compare-pos")
async def compare_pos(
    po_a: UploadFile = File(...),
    po_b: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Accepts two PO files (PDF / image / DOCX / XLSX), runs the PO auditor,
    and returns the comparison result.
    """
    path_a = _save_upload_to_temp(po_a)
    path_b = _save_upload_to_temp(po_b)

    try:
        result = compare_for_frontend(path_a, path_b)
        return result
    finally:
        for p in (path_a, path_b):
            try:
                os.remove(p)
            except OSError:
                pass


# Serve frontend files
frontend_dist = os.path.join(os.path.dirname(__file__), "react-frontend", "dist")

if os.path.exists(frontend_dist):
    # Mount assets folder
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    # Serve index.html at root
    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(frontend_dist, "index.html"))

    # Serve other static files (like vite.svg) if they exist in dist
    @app.get("/{filename}")
    async def serve_root_files(filename: str):
        file_path = os.path.join(frontend_dist, filename)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return {"message": "Not Found"}
else:
    @app.get("/")
    async def root():
        return {
            "message": "PO comparison AI is running (Backend Only). Frontend build ('dist') not found."
        }


