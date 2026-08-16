import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

# This demo backend is separate from the graded KLA submission (eval.py,
# scripts/, src/) — it exists only to drive the judge-facing UI in
# demo/frontend, per PRD.md §4's non-goal on web UI/interactive demos.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.restoration_net import RestorationNet  # noqa: E402

CHECKPOINT = REPO_ROOT / "checkpoints" / "best.pth"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Local-only demo tool — Vite picks whatever port is free, so allow any
    # localhost origin rather than hardcoding one.
    allow_origin_regex=r"http://localhost:\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RestorationNet().to(device)

checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded {CHECKPOINT.name} on {device}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": str(device)}


@app.post("/restore")
async def restore(file: UploadFile = File(...)) -> StreamingResponse:
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("L")

    # The trained model expects the KLA dataset's specific degraded-input
    # distribution (speckle noise, values roughly in [-0.28, 2.16]) — an
    # arbitrary uploaded photo has none of that. Normalizing to [0, 1] is
    # the only universal choice available for a demo tool and will not
    # match eval.py's preprocessing; this endpoint is for the pitch demo,
    # not a claim about submission-grade accuracy on arbitrary images.
    array = np.asarray(image, dtype=np.float32) / 255.0

    tensor = (
        torch.from_numpy(array)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        output = model(tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    restored = (
        output[0, 0].cpu().numpy().clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)

    out_image = Image.fromarray(restored, mode="L")

    buf = io.BytesIO()
    out_image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Inference-Ms": str((t1 - t0) * 1000.0),
            "X-Output-Width": str(out_image.width),
            "X-Output-Height": str(out_image.height),
        },
    )
