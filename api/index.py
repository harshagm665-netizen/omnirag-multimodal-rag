import sys
import os
import traceback

# Add root directory to python path so 'src' package can be found
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

# Top-level app assignment — required by Vercel's static analyzer
app = FastAPI(
    title="Multimodal RAG API",
    description="Enterprise API for processing PDFs, Images, and Tables with ColPali & Qdrant.",
    version="1.0.0",
)

_import_error = None
try:
    from src.api.main import app as _real_app
    app = _real_app
except Exception:
    _import_error = traceback.format_exc()

if _import_error:
    @app.get("/{path:path}")
    async def fallback(path: str = ""):
        return PlainTextResponse(
            f"Failed to import src.api.main\n\nTraceback:\n{_import_error}",
            status_code=500,
        )
