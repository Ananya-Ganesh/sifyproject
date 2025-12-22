from __future__ import annotations

import os
import tempfile
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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


# Serve React Frontend
# Check if the dist directory exists (it should in production)
if os.path.exists("dist"):
    app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        # Allow API routes to pass through if defined above
        # But here we only have /compare-pos which is POST.
        # So we just need to catch GET requests that aren't API.
        
        # If the file exists in dist, serve it (e.g. vite.svg)
        file_path = os.path.join("dist", full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
            
        # Otherwise serve index.html for client-side routing
        return FileResponse("dist/index.html")
else:
    @app.get("/")
    async def root():
        return {
            "message": (
                "PO comparison AI is running (Backend Only). "
                "Frontend build ('dist') not found."
            )
        }


