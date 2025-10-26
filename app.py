# app.py
"""
Polyvore Image Similarity Search (Backend)
- FAISS + metadata + id_map (+ vectors.npy optional for exact cosine re-ranking)
- Model: ResNet50 backbone (nn.Sequential -> backbone.0..7) + proj(2048->512)
- Transform khớp pipeline extract: Resize(224,224) + Normalize(ImageNet)

Endpoints:
  GET  /healthz, /readyz, /version, /items/{item_id}
  POST /search               (query_vector or query_item)
  POST /search_image         (upload image)
  POST /similar/{item_id}    (neighbors of existing item)
"""

import io
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
import torchvision.transforms as T
from torchvision import models

# ---------------------------- Config ----------------------------
DATA_DIR   = os.getenv("DATA_DIR", "data")
MODELS_DIR = os.getenv("MODELS_DIR", "models")
INDEX_PATH = os.getenv("INDEX_PATH", os.path.join(DATA_DIR, "faiss_index.index"))
IDMAP_PATH = os.getenv("IDMAP_PATH", os.path.join(DATA_DIR, "id_map.json"))
META_PATH  = os.getenv("META_PATH",  os.path.join(DATA_DIR, "items_metadata_joined_fixed.json"))
VEC_PATH   = os.getenv("VEC_PATH",   os.path.join(DATA_DIR, "vectors.npy"))
WEIGHTS    = os.getenv("MODEL_WEIGHTS", os.path.join(MODELS_DIR, "resnet50_proj512_best.pt"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

EMBED_DIM = 512
AUTO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = os.getenv("DEVICE", AUTO_DEVICE)

# ---------------------------- Logging ---------------------------
logger = logging.getLogger("polyvore-backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

# ---------------------------- App init --------------------------
app = FastAPI(title="Polyvore Similarity Backend", version="1.1.1")

# CORS + GZip
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ORIGINS == ["*"] else CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Timing middleware
@app.middleware("http")
async def add_timing(request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    ms = (time.time() - t0) * 1000
    logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path, resp.status_code, ms)
    return resp

# Serve local images (optional)
IMAGES_DIR = os.path.join(DATA_DIR, "images")
if os.path.isdir(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
    logger.info("Serving /images from %s", IMAGES_DIR)
else:
    logger.info("No local images dir at %s (skipping static serve)", IMAGES_DIR)

# -------------------- Load FAISS + metadata ---------------------
def _require(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

logger.info("Bootstrapping resources...")
for p in (INDEX_PATH, IDMAP_PATH, META_PATH):
    _require(p)

logger.info("Loading FAISS index: %s", INDEX_PATH)
_index = faiss.read_index(INDEX_PATH)

with open(IDMAP_PATH, "r", encoding="utf-8") as f:
    _id_map: Dict[int, str] = {int(k): v for k, v in json.load(f).items()}
_inv_map: Dict[str, int] = {v: k for k, v in _id_map.items()}

with open(META_PATH, "r", encoding="utf-8") as f:
    _meta: Dict[str, dict] = json.load(f)

def _inner_index(ix):
    try:
        return ix.index
    except Exception:
        return ix

_inner = _inner_index(_index)
_METRIC_IS_IP = ("ip" in type(_inner).__name__.lower())
logger.info("FAISS inner=%s metric=%s", type(_inner).__name__, "IP" if _METRIC_IS_IP else "L2")

_vectors: Optional[np.ndarray] = None
if os.path.exists(VEC_PATH):
    try:
        _vectors = np.load(VEC_PATH, mmap_mode="r")
        logger.info("vectors.npy loaded: shape=%s dtype=%s", getattr(_vectors, "shape", None), getattr(_vectors, "dtype", None))
    except Exception as e:
        logger.warning("Failed to load vectors.npy: %s", e)
        _vectors = None

# -------------------------- Utils --------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)

def _faiss_score_to_cosine(raw_distance: float) -> float:
    """
    Convert FAISS distance to cosine-like score (higher is better) assuming vectors are L2-normalized.
    - For L2 metric FAISS returns squared L2 distance d = ||a-b||^2.
      Cosine = 1 - d/2
    - For Inner Product, raw is already dot-product ~ cosine (if normalized).
    """
    return float(raw_distance) if _METRIC_IS_IP else float(1.0 - raw_distance / 2.0)

def _ensure_vec_1d(vec: np.ndarray):
    if vec.ndim != 1:
        raise HTTPException(status_code=400, detail="query_vector must be 1D")

# ------------------ Model: backbone.0..7 + proj ------------------
class FineTuneModel(nn.Module):
    """
    ResNet50 backbone wrapped in nn.Sequential(*children()[:-1]) so keys are:
      backbone.0 (conv1), backbone.1 (bn1), backbone.2 (relu), backbone.3 (maxpool),
      backbone.4 (layer1), backbone.5 (layer2), backbone.6 (layer3), backbone.7 (layer4)
    + proj: Linear(2048 -> 512) then L2-normalize.
    """
    def __init__(self, out_dim=EMBED_DIM):
        super().__init__()
        resnet = models.resnet50(weights=None)  # chuẩn mới (thay pretrained=False)
        modules = list(resnet.children())[:-1]  # đến avgpool
        self.backbone = nn.Sequential(*modules)
        self.proj = nn.Linear(2048, out_dim)

    def forward(self, x):
        x = self.backbone(x)           # (B,2048,1,1)
        x = x.view(x.size(0), -1)      # (B,2048)
        emb = self.proj(x)             # (B,512)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

def load_model(weights_path: str, device: str) -> nn.Module:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    logger.info("Loading model weights: %s", weights_path)

    model = FineTuneModel(out_dim=EMBED_DIM).to(device).eval()
    ckpt = torch.load(weights_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise RuntimeError("Unknown checkpoint type")

    # Giữ backbone.* và proj.*; bỏ classifier.*
    filtered = {k: v for k, v in state.items()
                if k.startswith("backbone.") or k.startswith("proj.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:    logger.warning("Missing keys: %s", missing)        # thường là []
    if unexpected: logger.warning("Unexpected keys: %s", unexpected)  # thường là []
    return model

# Transform KHỚP lúc extract
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

@torch.inference_mode()
def image_to_vec(img: Image.Image, model: nn.Module) -> np.ndarray:
    img = img.convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    emb = model(x)
    return emb[0].detach().cpu().numpy().astype(np.float32)

# ---------------------- Schemas ----------------------
class SearchRequest(BaseModel):
    query_item: Optional[str] = None
    query_vector: Optional[List[float]] = None
    topk: int = 10
    candidates: Optional[int] = None
    filter_main: Optional[str] = None
    filter_sub: Optional[str] = None

class SearchHit(BaseModel):
    item_id: str
    score: float
    title: Optional[str] = None
    main_category: Optional[str] = None
    sub_category: Optional[str] = None
    image_path: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchHit]

# ------------------- Core Search ----------------------
def _faiss_topk(query_vec: np.ndarray, k: int, k_candidates: int) -> Tuple[np.ndarray, np.ndarray]:
    q = _normalize(query_vec).reshape(1, -1).astype(np.float32)
    D, I = _index.search(q, k_candidates)
    return D[0], I[0]

def _post_filter_and_format(D: np.ndarray, I: np.ndarray,
                            topk: int,
                            filter_main: Optional[str],
                            filter_sub: Optional[str]) -> List[SearchHit]:
    out: List[SearchHit] = []
    for raw, iid in zip(D, I):
        if iid < 0: continue
        item_id = _id_map.get(int(iid))
        if not item_id: continue
        meta = _meta.get(str(item_id), {})
        if filter_main and meta.get("main_category") != filter_main: continue
        if filter_sub and meta.get("subcategory") != filter_sub: continue
        out.append(SearchHit(
            item_id=str(item_id),
            score=_faiss_score_to_cosine(float(raw)),
            title=meta.get("title"),
            main_category=meta.get("main_category"),
            sub_category=meta.get("subcategory"),
            image_path=meta.get("image_path"),
        ))
        if len(out) >= topk:
            break
    return out

def _rerank_with_exact_cosine(query_vec: np.ndarray, ids: np.ndarray, topk: int) -> List[int]:
    """
    Re-rank exact cosine using vectors.npy if available.
    Returns list of int ids (internal) sorted by cosine desc, length<=topk.
    """
    if _vectors is None:
        return ids[ids >= 0][:topk].tolist()
    q = _normalize(query_vec.astype(np.float32))
    cand = ids[ids >= 0]
    if cand.size == 0:
        return []
    mat = _vectors[cand]  # (C, 512)
    denom = np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8, None)
    mat_norm = mat / denom
    cos = (mat_norm @ q.reshape(-1, 1)).ravel()
    order = np.argsort(-cos)
    return cand[order][:topk].tolist()

# -------------------- App & Routes ---------------------
_model: nn.Module = load_model(WEIGHTS, DEVICE).eval()
app_state = {
    "device": DEVICE,
    "embed_dim": EMBED_DIM,
    "has_vectors_npy": _vectors is not None,
    "metric": "IP" if _METRIC_IS_IP else "L2",
}

@app.get("/healthz")
def healthz():
    return {"status": "ok", "items": len(_id_map)}

@app.get("/readyz")
def readyz():
    return {
        "index": os.path.exists(INDEX_PATH),
        "idmap": os.path.exists(IDMAP_PATH),
        "metadata": os.path.exists(META_PATH),
        "model": os.path.exists(WEIGHTS),
    }

@app.get("/version")
def version():
    return {
        "app": app.version or "1.1.1",
        "device": app_state["device"],
        "embed_dim": app_state["embed_dim"],
        "metric": app_state["metric"],
        "has_vectors_npy": app_state["has_vectors_npy"],
        "weights": os.path.basename(WEIGHTS),
    }

@app.get("/items/{item_id}", response_model=Dict)
def get_item(item_id: str):
    meta = _meta.get(str(item_id))
    if not meta:
        raise HTTPException(status_code=404, detail="item not found")
    return meta

@app.post("/similar/{item_id}", response_model=SearchResponse)
def similar(item_id: str,
            topk: int = Query(10, ge=1),
            filter_main: Optional[str] = None,
            filter_sub: Optional[str] = None,
            candidates: Optional[int] = None,
            rerank: bool = Query(False, description="Exact cosine re-rank using vectors.npy if available")):
    if item_id not in _inv_map:
        raise HTTPException(status_code=404, detail="item not found")
    int_id = _inv_map[item_id]
    try:
        v = _index.reconstruct(int(int_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reconstruct vector: {e}")
    vec = np.asarray(v, dtype=np.float32)

    k = max(1, int(topk))
    k_candidates = int(candidates) if candidates else (k * 10 if (filter_main or filter_sub or rerank) else k)
    D, I = _faiss_topk(vec, k, k_candidates)

    if rerank:
        re_ids = _rerank_with_exact_cosine(vec, I, topk=k)
        if _vectors is not None and len(re_ids) > 0:
            # dựng D/I giả theo thứ tự rerank để tái dùng formatter
            qn = _normalize(vec)
            mat = _vectors[re_ids]
            denom = np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8, None)
            mat_norm = mat / denom
            cos = (mat_norm @ qn.reshape(-1, 1)).ravel()
            return SearchResponse(results=_post_filter_and_format(cos, np.array(re_ids), k, filter_main, filter_sub))
        # fallback: filter D/I theo ids đã chọn
        mask = np.isin(I, np.array(re_ids))
        return SearchResponse(results=_post_filter_and_format(D[mask], I[mask], k, filter_main, filter_sub))

    return SearchResponse(results=_post_filter_and_format(D, I, k, filter_main, filter_sub))

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest,
           rerank: bool = Query(False, description="Exact cosine re-rank using vectors.npy if available")):
    if not req.query_item and req.query_vector is None:
        raise HTTPException(status_code=400, detail="Provide query_item or query_vector")

    if req.query_item:
        if req.query_item not in _inv_map:
            raise HTTPException(status_code=404, detail="query_item not found")
        int_id = _inv_map[req.query_item]
        try:
            v = _index.reconstruct(int(int_id))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to reconstruct vector: {e}")
        query = np.asarray(v, dtype=np.float32)
    else:
        query = np.asarray(req.query_vector, dtype=np.float32)
        _ensure_vec_1d(query)

    k = max(1, int(req.topk))
    k_candidates = int(req.candidates) if req.candidates else (k * 10 if (req.filter_main or req.filter_sub or rerank) else k)
    D, I = _faiss_topk(query, k, k_candidates)

    if rerank:
        re_ids = _rerank_with_exact_cosine(query, I, topk=k)
        if _vectors is not None and len(re_ids) > 0:
            qn = _normalize(query)
            mat = _vectors[re_ids]
            denom = np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8, None)
            mat_norm = mat / denom
            cos = (mat_norm @ qn.reshape(-1, 1)).ravel()
            return SearchResponse(results=_post_filter_and_format(cos, np.array(re_ids), k, req.filter_main, req.filter_sub))
        mask = np.isin(I, np.array(re_ids))
        return SearchResponse(results=_post_filter_and_format(D[mask], I[mask], k, req.filter_main, req.filter_sub))

    return SearchResponse(results=_post_filter_and_format(D, I, k, req.filter_main, req.filter_sub))

@app.post("/search_image", response_model=SearchResponse)
async def search_image(
    file: UploadFile = File(...),
    topk: int = Query(10, ge=1),
    filter_main: Optional[str] = None,
    filter_sub: Optional[str] = None,
    candidates: Optional[int] = None,
    rerank: bool = Query(False, description="Exact cosine re-rank using vectors.npy if available")
):
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    vec = image_to_vec(img, _model)

    k = max(1, int(topk))
    k_candidates = int(candidates) if candidates else (k * 10 if (filter_main or filter_sub or rerank) else k)
    D, I = _faiss_topk(vec, k, k_candidates)

    if rerank:
        re_ids = _rerank_with_exact_cosine(vec, I, topk=k)
        if _vectors is not None and len(re_ids) > 0:
            qn = _normalize(vec)
            mat = _vectors[re_ids]
            denom = np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8, None)
            mat_norm = mat / denom
            cos = (mat_norm @ qn.reshape(-1, 1)).ravel()
            return SearchResponse(results=_post_filter_and_format(cos, np.array(re_ids), k, filter_main, filter_sub))
        mask = np.isin(I, np.array(re_ids))
        return SearchResponse(results=_post_filter_and_format(D[mask], I[mask], k, filter_main, filter_sub))

    return SearchResponse(results=_post_filter_and_format(D, I, k, filter_main, filter_sub))



if __name__ == "__main__":
    import uvicorn
    print(">>> __main__ block reached, starting uvicorn ...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

