"""
Hybrid Content-Based Recommender System
---------------------------------------
1. Categorical Feature Engineering: TF-IDF & One-Hot Encoding + Average Pooling.
2. Text Feature Engineering:
   - Aggregates 'highlights', 'drawbacks', 'practical_tips' per location.
   - Deep Sentence Embedding using 'all-MiniLM-L6-v2' (384-dim).
   - Dimensionality Reduction with PCA (down to 20-dim).
3. Hybrid Matrix & Clustering:
   - Horizontal Stacking (Categorical + Text PCA).
   - KMeans Clustering on Hybrid Profiles.
4. Evaluation & Inference:
   - Simulates user query combining free text & structured preferences.
   - Top-K Cosine Similarity retrieval.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from fastembed import TextEmbedding
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import os

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class HybridLocationRecommender:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        pca_components: int = 20,
        n_clusters: Optional[int] = None,
    ):
        """
        Khởi tạo hệ thống gợi ý lai (Hybrid Recommender).
        """
        self.embedding_model_name = embedding_model_name
        self.pca_components = pca_components
        self.n_clusters = n_clusters

        # Models & Transformers
        self.sentence_model = TextEmbedding(model_name=f"sentence-transformers/{embedding_model_name}")
        self.pca = None
        self.kmeans = None
        self.tfidf_vectorizers: Dict[str, TfidfVectorizer] = {}

        # Dữ liệu & Ma trận lưu trữ
        self.location_ids: List[str] = []
        self.location_names: List[str] = []
        self.location_id_to_name: Dict[str, str] = {}
        self.categorical_matrix: Optional[np.ndarray] = None
        self.text_pca_matrix: Optional[np.ndarray] = None
        self.final_hybrid_matrix: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None

        # Cấu hình các trường đặc trưng có cấu trúc
        self.array_fields = ["activities", "target_audience", "vibes", "suitable_for", "amenities"]
        self.field_weights = {
            "activities": 1.5,
            "target_audience": 1.5,
            "vibes": 1.2,
            "suitable_for": 1.0,
            "amenities": 0.8,
        }

    # =========================================================================
    # 1. LOAD & TIỀN XỬ LÝ DỮ LIỆU
    # =========================================================================
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Đọc và xử lý file JSON chứa các review đã enrich."""
        logger.info(f"Đang đọc dữ liệu từ: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        
        # Đảm bảo có location_id và location_name
        if "location_id" not in df.columns:
            if "location_name" in df.columns:
                df["location_id"] = df["location_name"]
            else:
                raise ValueError("Không tìm thấy trường 'location_id' hoặc 'location_name'.")

        # Map locationType từ file gốc để lọc restaurant
        try:
            # File gốc chứa locationType
            with open(filepath.replace("llm_enriched_reviews.json", "reviews_data.json"), "r", encoding="utf-8-sig") as f:
                raw_data = json.load(f)
            loc_type_map = {item.get("locationName", ""): item.get("locationType", "") for item in raw_data}
            df["locationType"] = df["location_name"].map(loc_type_map)
            
            original_len = len(df)
            df = df[~df["locationType"].astype(str).str.lower().str.contains("restaurant", na=False)]
            logger.info(f"Đã lọc bỏ {original_len - len(df)} reviews thuộc về restaurant.")
        except Exception as e:
            logger.warning(f"Không thể map locationType: {e}")

        # Xử lý các giá trị missing / null an toàn
        text_cols = ["highlights", "drawbacks", "practical_tips", "extracted_spaces", "extracted_times"]
        for col in text_cols:
            if col not in df.columns:
                df[col] = [[] for _ in range(len(df))]
            else:
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

        for col in self.array_fields:
            if col not in df.columns:
                df[col] = [[] for _ in range(len(df))]
            else:
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

        logger.info(f"Đã nạp {len(df)} reviews cho {df['location_id'].nunique()} địa điểm.")
        return df

    # =========================================================================
    # 2. XÂY DỰNG CATEGORICAL VECTOR MATRIX
    # =========================================================================
    def _build_categorical_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Mã hóa các trường danh mục bằng TF-IDF, áp trọng số và tính trung bình theo địa điểm."""
        logger.info("Đang tạo đặc trưng có cấu trúc (Categorical / Tag features)...")
        field_matrices = []

        for field in self.array_fields:
            # Chuyển list các thẻ thành chuỗi token cách nhau bởi dấu cách
            joined_docs = df[field].apply(lambda lst: " ".join([str(item).replace(" ", "_") for item in lst]))
            
            vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=True)
            try:
                mat = vec.fit_transform(joined_docs).toarray()
                self.tfidf_vectorizers[field] = vec

                # Nhân trọng số cho từng trường
                weight = self.field_weights.get(field, 1.0)
                field_matrices.append(mat * weight)
            except ValueError:
                logger.warning(f"Trường '{field}' trống hoàn toàn, bỏ qua không dùng cho vector.")
                continue

        # Ghép tất cả các feature categorical của từng review
        review_cat_matrix = np.hstack(field_matrices)
        df_cat = pd.DataFrame(review_cat_matrix, index=df["location_id"])

        # Average Pooling theo từng địa điểm
        location_grouped = df_cat.groupby(level=0).mean()
        unique_locs = location_grouped.index.tolist()
        loc_cat_matrix = location_grouped.to_numpy()

        # Chuẩn hóa L2
        loc_cat_matrix = normalize(loc_cat_matrix, norm="l2")
        return loc_cat_matrix, unique_locs

    # =========================================================================
    # 3. TEXT FEATURE ENGINEERING (SENTENCE EMBEDDING + PCA)
    # =========================================================================
    def _build_text_pca_matrix(self, df: pd.DataFrame, location_order: List[str]) -> np.ndarray:
        """
        Ghép 'highlights', 'drawbacks', 'practical_tips' thành Location Document,
        sau đó embed bằng SentenceTransformer và giảm chiều bằng PCA.
        """
        logger.info("Đang tổng hợp Location Document từ highlights, drawbacks, practical_tips...")
        location_docs = []

        for loc_id in location_order:
            sub = df[df["location_id"] == loc_id]
            all_text_snippets = []

            for _, row in sub.iterrows():
                # Lấy text an toàn từ các trường mở
                for field in ["highlights", "drawbacks", "practical_tips", "extracted_spaces", "extracted_times"]:
                    val = row.get(field, [])
                    if isinstance(val, list):
                        all_text_snippets.extend([str(item).strip() for item in val if str(item).strip()])
                    elif isinstance(val, str) and val.strip():
                        all_text_snippets.append(val.strip())

            # Ghép thành 1 văn bản toàn diện cho địa điểm
            if all_text_snippets:
                combined_doc = ". ".join(all_text_snippets)
            else:
                # Fallback nếu địa điểm chưa có review text mở
                loc_name = sub["location_name"].iloc[0] if "location_name" in sub.columns else loc_id
                combined_doc = f"Popular destination: {loc_name}"

            location_docs.append(combined_doc)

        logger.info(f"Đang sinh vector nhúng (Embedding) bằng '{self.embedding_model_name}' (384-dim)...")
        # Embedding 384 chiều
        embeddings_384 = np.vstack(list(self.sentence_model.embed(
            location_docs,
            batch_size=16
        )))

        # Giảm chiều bằng PCA xuống n_components=20 (tránh lấn át categorical)
        n_samples = len(location_docs)
        actual_components = min(self.pca_components, n_samples - 1 if n_samples > 1 else 1)
        
        logger.info(f"Áp dụng PCA giảm chiều từ 384 -> {actual_components} dimensions...")
        self.pca = PCA(n_components=actual_components, random_state=42)
        text_pca = self.pca.fit_transform(embeddings_384)

        # Chuẩn hóa L2 cho ma trận text PCA
        text_pca = normalize(text_pca, norm="l2")
        return text_pca

    # =========================================================================
    # 4. FIT PIPELINE (HYBRID REPRESENTATION & KMEANS CLUSTERING)
    # =========================================================================
    def fit(self, df_or_filepath: Any):
        """Huấn luyện toàn bộ pipeline từ DataFrame hoặc file JSON."""
        if isinstance(df_or_filepath, str):
            df = self.load_data(df_or_filepath)
        else:
            df = df_or_filepath

        # Lưu ánh xạ ID -> Name
        if "location_name" in df.columns:
            self.location_id_to_name = (
                df.groupby("location_id")["location_name"].first().to_dict()
            )
        else:
            self.location_id_to_name = {loc: str(loc) for loc in df["location_id"].unique()}

        # 1. Categorical Feature Matrix
        self.categorical_matrix, self.location_ids = self._build_categorical_matrix(df)
        self.location_names = [self.location_id_to_name.get(lid, str(lid)) for lid in self.location_ids]

        # 2. Text Feature Matrix qua SentenceTransformer + PCA
        self.text_pca_matrix = self._build_text_pca_matrix(df, self.location_ids)

        # 3. Hybrid Representation (np.hstack)
        logger.info("Ghép nối (np.hstack) Categorical Matrix và Text PCA Matrix...")
        self.final_hybrid_matrix = np.hstack([self.categorical_matrix, self.text_pca_matrix])
        self.final_hybrid_matrix = normalize(self.final_hybrid_matrix, norm="l2")

        logger.info(f"Final Location Matrix Shape: {self.final_hybrid_matrix.shape} "
                    f"({len(self.location_ids)} địa điểm x {self.final_hybrid_matrix.shape[1]} đặc trưng)")

        # 4. KMeans Clustering
        num_locs = len(self.location_ids)
        if self.n_clusters is None:
            # Tự động chọn số cụm hợp lý (ví dụ: căn bậc 2 số địa điểm, tối đa 8, tối thiểu 2)
            self.n_clusters = max(2, min(8, int(np.sqrt(num_locs))))

        logger.info(f"Huấn luyện KMeans Clustering với k={self.n_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.final_hybrid_matrix)

        logger.info("Hoàn tất huấn luyện mô hình Hybrid Recommender!")

    # =========================================================================
    # 5. INFERENCE / EVALUATION FUNCTION
    # =========================================================================
    def recommend(
        self,
        query_text: str = "",
        user_preferences: Optional[Dict[str, List[str]]] = None,
        top_k: int = 5,
        text_weight: float = 1.0,
        cat_weight: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Dự đoán và xếp hạng địa điểm dựa trên câu truy vấn tự do và sở thích danh mục.
        """
        if self.final_hybrid_matrix is None:
            raise RuntimeError("Mô hình chưa được huấn luyện! Hãy gọi fit() trước.")

        user_preferences = user_preferences or {}

        # 1. Transform Categorical Query
        cat_parts = []
        for field in self.array_fields:
            if field not in self.tfidf_vectorizers:
                continue
            tags = user_preferences.get(field, [])
            joined = " ".join([str(t).replace(" ", "_") for t in tags])
            vec = self.tfidf_vectorizers[field]
            mat = vec.transform([joined]).toarray()
            weight = self.field_weights.get(field, 1.0)
            cat_parts.append(mat * weight)

        user_cat_vector = np.hstack(cat_parts)
        if np.linalg.norm(user_cat_vector) > 0:
            user_cat_vector = normalize(user_cat_vector, norm="l2")

        # 2. Transform Text Query (SentenceTransformer -> PCA)
        if query_text.strip():
            raw_embed = np.vstack(list(self.sentence_model.embed([query_text])))
            user_text_pca = self.pca.transform(raw_embed)
            user_text_pca = normalize(user_text_pca, norm="l2")
        else:
            user_text_pca = np.zeros((1, self.pca.n_components_))

        # 3. Tạo User Hybrid Vector với trọng số điều chỉnh
        user_hybrid_vector = np.hstack([user_cat_vector * cat_weight, user_text_pca * text_weight])
        if np.linalg.norm(user_hybrid_vector) > 0:
            user_hybrid_vector = normalize(user_hybrid_vector, norm="l2")

        # 4. Tính Cosine Similarity
        scores = cosine_similarity(user_hybrid_vector, self.final_hybrid_matrix)[0]

        # 5. Lấy Top-K kết quả
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            loc_id = self.location_ids[idx]
            results.append({
                "rank": rank,
                "location_id": loc_id,
                "location_name": self.location_names[idx],
                "similarity_score": round(float(scores[idx]), 4),
                "cluster_id": int(self.cluster_labels[idx]),
            })

        return results


    # =========================================================================
    # 6. EVALUATE HIT RATE
    # =========================================================================
    def evaluate_hit_rate(self, df: pd.DataFrame, ks: List[int] = [3, 5, 7, 9]):
        logger.info("Bắt đầu chia tập Train (80%) / Test (20%) ngẫu nhiên theo từng địa điểm...")
        train_dfs = []
        test_dfs = []
        for loc_id, group in df.groupby("location_id"):
            group = group.sample(frac=1.0)
            n_train = max(1, int(len(group) * 0.8))
            train_dfs.append(group.iloc[:n_train])
            test_dfs.append(group.iloc[n_train:])
            
        train_df = pd.concat(train_dfs)
        test_df = pd.concat(test_dfs)
        logger.info(f"Train set: {len(train_df)} reviews. Test set: {len(test_df)} reviews.")
        
        # Build Location Profiles from Train set ONLY
        logger.info("Huấn luyện mô hình Hybrid bằng tập Train...")
        self.fit(train_df)
        
        # Evaluate on Test set
        hits = {k: 0 for k in ks}
        total = len(test_df)
        
        logger.info("Đang đánh giá Hit Rate trên tập Test...")
        for _, row in test_df.iterrows():
            true_loc_id = row["location_id"]
            
            # Xây dựng User Preferences từ review
            user_prefs = {}
            for field in self.array_fields:
                val = row.get(field, [])
                if isinstance(val, list):
                    user_prefs[field] = val
            
            # Xây dựng Query text từ các text mở
            query_parts = []
            for field in ["highlights", "drawbacks", "practical_tips", "extracted_spaces", "extracted_times"]:
                val = row.get(field, [])
                if isinstance(val, list):
                    query_parts.extend([str(x).strip() for x in val if str(x).strip()])
                elif isinstance(val, str) and val.strip():
                    query_parts.append(val.strip())
            query_text = ". ".join(query_parts)
            
            # Giả lập gợi ý với User Vector
            recs = self.recommend(query_text=query_text, user_preferences=user_prefs, top_k=max(ks))
            
            # Kiểm tra Hits
            top_k_locs = [r["location_id"] for r in recs]
            for k in ks:
                if true_loc_id in top_k_locs[:k]:
                    hits[k] += 1
                    
        print("\n" + "=" * 50)
        print("ĐÁNH GIÁ CHỈ SỐ HIT RATE (HYBRID MODEL):")
        print("=" * 50)
        for k in ks:
            hit_rate = (hits[k] / total) * 100
            print(f"Hit Rate @ {k}: {hit_rate:.2f}% (Đúng {hits[k]}/{total} reviews)")
        print("=" * 50)

# =========================================================================
# CHẠY THỬ NGHIỆM TRÊN DỮ LIỆU THỰC TẾ
# =========================================================================
if __name__ == "__main__":
    DATA_PATH = r"D:\Workspace\projects\recommender_project\llm_enriched_reviews.json"
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found!")
        exit(1)

    recommender = HybridLocationRecommender(
        embedding_model_name="all-MiniLM-L6-v2",
        pca_components=20,
        n_clusters=6,
    )
    
    # Nạp toàn bộ data
    df_all = recommender.load_data(DATA_PATH)
    
    # Chạy Evaluation Hit Rate
    recommender.evaluate_hit_rate(df_all, ks=[3, 5, 7, 9])
