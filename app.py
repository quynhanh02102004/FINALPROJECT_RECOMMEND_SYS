# CÁC THƯ VIỆN CẦN THIẾT
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware # <--- IMPORT THÊM DÒNG NÀY
import uvicorn
import torch
import faiss
from PIL import Image
import numpy as np
import pandas as pd
import io
import torchvision.transforms as transforms

# KHỞI TẠO ỨNG DỤNG FASTAPI
app = FastAPI(title='YAME Clone API')

# === PHẦN CODE MỚI: CẤU HÌNH CORS ===
# CORS (Cross-Origin Resource Sharing) cho phép frontend trên Vercel
# gọi được API của backend đang chạy trên Render.
origins = [
    "http://localhost:3000",                  # Dành cho lúc bạn phát triển ở local
    "https://l.facebook.com/l.php?u=https%3A%2F%2Fyame-clone-animated-o7tp.vercel.app%2F%3Ffbclid%3DIwZXh0bgNhZW0CMTAAYnJpZBExVUVhSDVSbDV0OXU2c0lwSgEesDItWOLIdlAsCQMjqqDoWTuq3RacV2U1MM20iMwe1pskTQ8GarhbgnlcxUk_aem_98TF2lIqZey8Jmm6RCTWWw&h=AT2aPfLVOE_bBu46xqFIswyqgEgbi3dpBE_Gr4BPeHz2tFqobmszComvuGQwUmCjBw2qDPzRhnJq5MDSfZRHpMgnICPvPyLgzZK3XaYmAMQ2rmj9z0WBg9e-2CsSUIA" # URL của frontend trên Vercel
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các method (GET, POST, etc.)
    allow_headers=["*"],  # Cho phép tất cả các header
)
# ============================================


# --- TẢI MODEL VÀ DỮ LIỆU ---
# Phần này sẽ chạy một lần duy nhất khi server khởi động.
# Trên Render (gói Free), việc này có thể mất một chút thời gian.

print("Starting server and loading models...")

# Xác định thiết bị (sẽ luôn là 'cpu' trên Render)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Tải model PyTorch đã được huấn luyện
model = torch.load('data/model.pkl', map_location=device)
model.eval()
print("Model loaded successfully.")

# Tải mảng features đã được trích xuất
features_arr = np.load('data/features_arr.npy')
print("Features array loaded successfully.")

# Tải dữ liệu sản phẩm
df = pd.read_csv('data/product_data_with_images.csv')
print("Product data loaded successfully.")

# Tải index của Faiss để tìm kiếm nhanh
index = faiss.read_index('data/faiss_index.idx')
print("Faiss index loaded successfully.")

# Định nghĩa các bước chuyển đổi hình ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("Image transformer created.")
print("--- Server is ready to accept requests ---")
# ------------------------------------


# --- CÁC HÀM HỖ TRỢ ---
def get_vector(image):
    """Hàm trích xuất vector đặc trưng từ một hình ảnh."""
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        vector = model.extract_features(image_tensor).cpu().numpy().flatten()
    return vector
# -------------------------


# --- CÁC API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"message": "Welcome to YAME Clone Recommendation API!"}

@app.post("/api/search_by_image")
async def search_by_image(file: UploadFile = File(...)):
    """API nhận một file ảnh, tìm kiếm 10 sản phẩm tương tự nhất."""
    # Đọc nội dung file ảnh từ request
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Trích xuất vector đặc trưng từ ảnh
    query_vector = get_vector(image)
    query_vector = np.expand_dims(query_vector, axis=0) # Reshape for Faiss
    
    # Tìm kiếm trong Faiss index
    k = 10  # Số lượng sản phẩm tương tự cần tìm
    distances, indices = index.search(query_vector, k)
    
    # Lấy thông tin sản phẩm từ các index tìm được
    results_df = df.iloc[indices[0]]
    results = results_df.to_dict(orient='records')
    
    return {"results": results}

@app.get("/api/recommend/{product_id}")
async def recommend(product_id: int):
    """API nhận ID sản phẩm, gợi ý 10 sản phẩm tương tự."""
    try:
        # Tìm vị trí (index) của sản phẩm trong DataFrame
        product_index = df.index[df['product_id'] == product_id].tolist()[0]
        
        # Lấy vector đặc trưng của sản phẩm đó
        query_vector = features_arr[product_index]
        query_vector = np.expand_dims(query_vector, axis=0)
        
        # Tìm kiếm trong Faiss, k=11 để bao gồm cả chính nó
        k = 11
        distances, indices = index.search(query_vector, k)
        
        # Lấy thông tin sản phẩm và loại bỏ sản phẩm gốc khỏi kết quả
        similar_indices = [idx for idx in indices[0] if idx != product_index]
        results_df = df.iloc[similar_indices[:10]] # Lấy 10 sản phẩm
        results = results_df.to_dict(orient='records')
        
        return {"results": results}
    except IndexError:
        return {"error": "Product ID not found"}, 404
# ----------------------------


# === PHẦN CODE CẦN VÔ HIỆU HÓA (COMMENT LẠI) ===
# Đoạn code này chỉ dùng để chạy server ở local.
# Trên Render, họ sẽ dùng Start Command để chạy server, nên ta không cần đoạn này nữa.
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)
# =================================================