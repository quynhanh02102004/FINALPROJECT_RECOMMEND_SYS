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
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

# --- Paths cho SIMILARITY search ---
SIMILARITY_DATA_DIR = os.path.join(DATA_DIR, "similarity")
INDEX_PATH = os.path.join(SIMILARITY_DATA_DIR, "faiss_index.index")
IDMAP_PATH = os.path.join(SIMILARITY_DATA_DIR, "id_map.json")
VEC_PATH   = os.path.join(SIMILARITY_DATA_DIR, "vectors.npy")

# --- Paths cho COMPATIBILITY search ---
COMPAT_DATA_DIR = os.path.join(DATA_DIR, "compatibility")
COMPAT_INDEX_PATH = os.path.join(COMPAT_DATA_DIR, "faiss_index_feature2.index")
COMPAT_IDMAP_PATH = os.path.join(COMPAT_DATA_DIR, "id_map_feature2.json")
COMPAT_VEC_PATH = os.path.join(COMPAT_DATA_DIR, "vectors_feature2.npy")

# --- Path dùng chung và model ---
META_PATH  = os.path.join(DATA_DIR, "items_metadata_joined_fixed.json")
WEIGHTS    = os.path.join(MODELS_DIR, "resnet50_proj512_best.pt")

EMBED_DIM = 512
AUTO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = os.getenv("DEVICE", AUTO_DEVICE)

# ---------------------------- Logging ---------------------------
logger = logging.getLogger("polyvore-backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

# ---------------------------- App init --------------------------
app = FastAPI(title="Polyvore Smart Compatibility Backend", version="3.2.0-final")
app.add_middleware(CORSMiddleware, allow_origins=["*"] if CORS_ORIGINS == ["*"] else CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1024)

@app.middleware("http")
async def add_timing(request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    ms = (time.time() - t0) * 1000
    logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path, resp.status_code, ms)
    return resp

IMAGES_DIR = os.path.join(DATA_DIR, "images")
if os.path.isdir(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# ===================================================================
# =========== LOGIC DEFINITIONS FOR SMART COMPATIBILITY =============
# ===================================================================
SUMMER_BEACH_SUBCATS = {'swimsuit', 'one-piece swimsuit', 'swimsuit bottom', 'swimsuit top', 'coverup', 'male swim shorts', 'flip-flops', 'male flip-flops', 'sandal', 'sandals', 'kimono', 'shorts'}
WINTER_COLD_SUBCATS = {'winter hat', 'gloves', 'male gloves', 'scarf', 'male scarves', 'turtleneck sweater', 'sweater', 'male sweater', 'parka', 'coat', 'boots', 'booties', 'heeled boots', 'over-the-knee boots', 'flat boots'}
FORMAL_OFFICE_SUBCATS = {'blazer', 'male formal jacket', 'male suit jacket', 'male suit pants', 'male suit', 'heels', 'pump', 'male formal shoes', 'male loafers', 'male shirt', 'blouse', 'tie', 'bowtie', 'cufflings'}

SUMMER_BEACH_WHITELIST = {'TOPS', 'BOTTOMS', 'BAGS', 'JEWELLERY', 'SHOES', 'ACCESSORIES', 'SUNGLASSES', 'HATS', 'ALL-BODY'}
WINTER_WHITELIST = {'TOPS', 'BOTTOMS', 'BAGS', 'JEWELLERY', 'SHOES', 'ACCESSORIES', 'OUTERWEAR', 'HATS'}
FORMAL_OFFICE_WHITELIST = {'TOPS', 'BOTTOMS', 'BAGS', 'JEWELLERY', 'SHOES', 'ACCESSORIES', 'OUTERWEAR'}
GENERAL_WHITELIST = {'TOPS', 'BOTTOMS', 'BAGS', 'JEWELLERY', 'SHOES', 'ACCESSORIES', 'OUTERWEAR', 'ALL-BODY'}

INCOMPATIBLE_PAIRS = {
    'ALL-BODY': {'TOPS', 'BOTTOMS', 'OUTERWEAR'},
    'TOPS': {'ALL-BODY'},
    'BOTTOMS': {'ALL-BODY'},
    'OUTERWEAR': {'ALL-BODY'}
}

# -------------------- Load Resources ---------------------
def _require(path: str):
    if not os.path.exists(path): raise FileNotFoundError(f"Missing required file: {path}")

_require(META_PATH)
with open(META_PATH, "r", encoding="utf-8") as f: _meta: Dict[str, dict] = json.load(f)
logger.info("Metadata loaded with %d items", len(_meta))

_require(INDEX_PATH); _require(IDMAP_PATH)
_index = faiss.read_index(INDEX_PATH)
with open(IDMAP_PATH, "r") as f: _id_map: Dict[int, str] = {int(k): v for k, v in json.load(f).items()}
_inv_map: Dict[str, int] = {v: k for k, v in _id_map.items()}
_vectors: Optional[np.ndarray] = np.load(VEC_PATH, mmap_mode="r") if os.path.exists(VEC_PATH) else None
logger.info("Similarity resources loaded. Has vectors.npy: %s", _vectors is not None)

_compat_index, _compat_id_map, _compat_inv_map, _compat_vectors = (None, {}, {}, None)
if os.path.exists(COMPAT_INDEX_PATH):
    try:
        _compat_index = faiss.read_index(COMPAT_INDEX_PATH)
        with open(COMPAT_IDMAP_PATH, "r") as f: _compat_id_map = {int(k): v for k, v in json.load(f).items()}
        _compat_inv_map = {v: k for k, v in _compat_id_map.items()}
        _compat_vectors = np.load(COMPAT_VEC_PATH)
        logger.info("Compatibility resources loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load compatibility resources: %s", e)

# ------------------ Model Definition and Loading ------------------
class FineTuneModel(nn.Module):
    def __init__(self, out_dim=EMBED_DIM):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(2048, out_dim)
    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        return F.normalize(self.proj(x), p=2, dim=1)

_model: nn.Module = None
if os.path.exists(WEIGHTS):
    try:
        def load_model(weights_path, device):
            model = FineTuneModel(out_dim=EMBED_DIM).to(device).eval()
            ckpt = torch.load(weights_path, map_location=device)
            state = ckpt.get("state_dict", ckpt)
            filtered = {k: v for k, v in state.items() if k.startswith(("backbone.", "proj."))}
            model.load_state_dict(filtered, strict=False)
            return model
        _model = load_model(WEIGHTS, DEVICE)
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")

preprocess = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

@torch.inference_mode()
def image_to_vec(img: Image.Image) -> np.ndarray:
    if not _model: raise HTTPException(503, "Image search disabled: model not loaded.")
    img_rgb = img.convert("RGB")
    x = preprocess(img_rgb).unsqueeze(0).to(DEVICE)
    return _model(x)[0].cpu().numpy().astype(np.float32)

# ---------------------- API Schemas ----------------------
class SearchHit(BaseModel):
    item_id: str; score: float; title: Optional[str] = None
    main_category: Optional[str] = None; sub_category: Optional[str] = None
    image_path: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchHit]

# ------------------- Core Search Logic with Re-ranking ----------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v); return v if n == 0 else (v / n)

def _calculate_exact_cosine(query_vec: np.ndarray, result_indices: List[int], vector_matrix: np.ndarray) -> np.ndarray:
    if not result_indices: return np.array([])
    q_norm = _normalize(query_vec)
    candidate_vectors = vector_matrix[result_indices]
    norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
    candidate_vectors_norm = candidate_vectors / np.where(norms == 0, 1e-8, norms)
    return (candidate_vectors_norm @ q_norm.reshape(-1, 1)).ravel()

def _format_results(item_ids: List[str], scores: List[float]) -> List[SearchHit]:
    results = []
    for item_id, score in zip(item_ids, scores):
        meta = _meta.get(item_id, {})
        results.append(SearchHit(
            item_id=item_id, score=float(score), title=meta.get("title"),
            main_category=meta.get("main_category"), sub_category=meta.get("subcategory"),
            image_path=meta.get("image_path")
        ))
    return results

# -------------------- API Endpoints ---------------------
@app.get("/healthz")
def healthz(): return {"status": "ok", "similarity_items": len(_id_map), "compatibility_items": len(_compat_id_map)}

# === ENDPOINT "THÔNG MINH" NHẤT CHO GỢI Ý PHỐI ĐỒ ===
@app.post("/compatible/{item_id}", response_model=SearchResponse, summary="Get smart, context-aware compatible items")
def get_compatible_items_smart_filter(
    item_id: str, topk: int = Query(5, ge=1, le=10), candidates: int = Query(50000, ge=100),
):
    if not _compat_index: raise HTTPException(503, "Compatibility feature is not available.")
    if item_id not in _compat_inv_map: raise HTTPException(404, "Item not found in database.")

    q_meta = _meta.get(item_id, {})
    query_main_cat = q_meta.get("main_category", "").strip().upper()
    query_sub_cat = q_meta.get("subcategory", "").strip()

    category_whitelist = GENERAL_WHITELIST
    forbidden_subcat_group = set()
    context = "GENERAL"
    if query_sub_cat in SUMMER_BEACH_SUBCATS:
        category_whitelist = SUMMER_BEACH_WHITELIST; forbidden_subcat_group = WINTER_COLD_SUBCATS; context = "SUMMER/BEACH"
    elif query_sub_cat in WINTER_COLD_SUBCATS:
        category_whitelist = WINTER_WHITELIST; forbidden_subcat_group = SUMMER_BEACH_SUBCATS; context = "WINTER/COLD"
    elif query_sub_cat in FORMAL_OFFICE_SUBCATS:
        category_whitelist = FORMAL_OFFICE_WHITELIST; context = "FORMAL/OFFICE"
    
    logger.info(f"Item {item_id} context: {context}. Activating filter rules.")
    blacklisted_categories = INCOMPATIBLE_PAIRS.get(query_main_cat, set())

    q_index = _compat_inv_map[item_id]
    query_vec = _compat_vectors[q_index].astype('float32').reshape(1, -1)
    _, I = _compat_index.search(query_vec, candidates)

    filtered_candidates = []
    selected_categories = {query_main_cat}
    
    for iid in I[0]:
        if iid < 0 or iid == q_index: continue
        
        result_item_id = _compat_id_map.get(int(iid))
        if not result_item_id: continue

        meta = _meta.get(result_item_id, {})
        main_cat = meta.get("main_category", "").strip().upper()
        sub_cat = meta.get("subcategory", "").strip()

        if (not main_cat or
            main_cat not in category_whitelist or
            main_cat in selected_categories or
            main_cat in blacklisted_categories or
            sub_cat in forbidden_subcat_group):
            continue
        
        filtered_candidates.append(int(iid))
        selected_categories.add(main_cat)
    
    if not filtered_candidates: return SearchResponse(results=[])

    scores = _calculate_exact_cosine(query_vec.flatten(), filtered_candidates, _compat_vectors)
    sorted_reranked_indices = np.array(filtered_candidates)[np.argsort(-scores)]
    final_indices = sorted_reranked_indices[:topk].tolist()
    
    final_item_ids = [_compat_id_map[i] for i in final_indices]
    final_scores = _calculate_exact_cosine(query_vec.flatten(), final_indices, _compat_vectors)
    
    return SearchResponse(results=_format_results(final_item_ids, final_scores.tolist()))

# === CÁC ENDPOINT CŨ CHO TÌM KIẾM TƯƠNG TỰ (với re-ranking) ===
def _perform_similarity_search(query_vec: np.ndarray, topk: int, rerank: bool, candidates: Optional[int] = None) -> List[SearchHit]:
    k = max(1, topk); k_candidates = candidates or (k * 20 if rerank else k)
    D, I = _index.search(_normalize(query_vec).reshape(1, -1).astype('float32'), k_candidates)
    
    result_indices = I[0][I[0] >= 0].tolist()
    if not result_indices: return []

    if rerank and _vectors is not None:
        scores = _calculate_exact_cosine(query_vec, result_indices, _vectors)
        sorted_reranked_indices = np.array(result_indices)[np.argsort(-scores)]
        final_indices = sorted_reranked_indices[:k].tolist()
        final_scores = _calculate_exact_cosine(query_vec, final_indices, _vectors).tolist()
    else:
        final_indices = result_indices[:k]; final_distances = D[0][:k]
        final_scores = [1.0 - (d / 2.0) for d in final_distances]
    
    final_item_ids = [_id_map[i] for i in final_indices]
    return _format_results(final_item_ids, final_scores)

@app.post("/similar/{item_id}", response_model=SearchResponse, summary="Find visually similar items")
def similar(item_id: str, topk: int = Query(10, ge=1), rerank: bool = Query(True), candidates: Optional[int] = None):
    if item_id not in _inv_map: raise HTTPException(404, "Item not found")
    vec = _vectors[_inv_map[item_id]] if _vectors is not None else _index.reconstruct(_inv_map[item_id])
    return SearchResponse(results=_perform_similarity_search(vec, topk, rerank, candidates))

@app.post("/search_image", response_model=SearchResponse, summary="Find similar items by uploading an image")
async def search_image(file: UploadFile = File(...), topk: int = Query(10, ge=1), rerank: bool = Query(True), candidates: Optional[int] = None):
    content = await file.read(); img = Image.open(io.BytesIO(content))
    vec = image_to_vec(img)
    return SearchResponse(results=_perform_similarity_search(vec, topk, rerank, candidates))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)