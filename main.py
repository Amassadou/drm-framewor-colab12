"""
DRM Framework - Minimal FastAPI entry point
"""

from fastapi import FastAPI
from datetime import datetime

app = FastAPI(
    title="DRM Framework",
    description="Minimal API for DRM watermarking experiments",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {
        "project": "DRM Framework",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/info")
async def info():
    return {
        "framework": "DRM Framework",
        "watermark": "FWHT + SIFT + BCH",
        "hash": "MobileNetV3 + SHA-256",
        "storage": "IPFS",
        "blockchain": "Polygon",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
