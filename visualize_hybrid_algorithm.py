import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os

from hybrid_recommender import HybridLocationRecommender

def visualize_clusters(data_path, output_image_path):
    # Khởi tạo mô hình
    recommender = HybridLocationRecommender(
        embedding_model_name="all-MiniLM-L6-v2",
        pca_components=20,
        n_clusters=6,
    )
    
    # Fit mô hình để lấy final_hybrid_matrix (68D) và cluster_labels
    print("Đang huấn luyện mô hình để lấy vector đặc trưng...")
    recommender.fit(data_path)
    
    # Giảm chiều xuống 2D để vẽ biểu đồ
    print("Đang giảm chiều bằng PCA xuống 2D để vẽ biểu đồ...")
    pca_2d = PCA(n_components=2, random_state=42)
    hybrid_matrix_2d = pca_2d.fit_transform(recommender.final_hybrid_matrix)
    
    # Chuẩn bị DataFrame cho Seaborn
    plot_df = pd.DataFrame({
        'PC1': hybrid_matrix_2d[:, 0],
        'PC2': hybrid_matrix_2d[:, 1],
        'Cluster': [f"Cluster {c}" for c in recommender.cluster_labels],
        'Location': recommender.location_names
    })
    
    # Vẽ biểu đồ
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    
    # Tạo scatter plot
    scatter = sns.scatterplot(
        data=plot_df,
        x='PC1',
        y='PC2',
        hue='Cluster',
        palette='Set2',
        s=150,
        edgecolor='k',
        alpha=0.8
    )
    
    # Thêm text cho từng điểm
    for i in range(len(plot_df)):
        plt.text(
            plot_df['PC1'][i] + 0.01, 
            plot_df['PC2'][i] + 0.01, 
            plot_df['Location'][i], 
            fontsize=9,
            color='darkblue',
            alpha=0.7
        )
    
    plt.title('2D Visualization of Hybrid Location Profiles (Text Embedding + Categorical)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(f'Principal Component 1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    plt.legend(title="K-Means Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Lưu biểu đồ
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ tại: {output_image_path}")

if __name__ == "__main__":
    DATA_PATH = r"D:\Workspace\projects\recommender_project\llm_enriched_reviews.json"
    OUTPUT_PATH = r"C:\Users\dung1\.gemini\antigravity-ide\brain\722d267c-4a51-4f20-97f9-6b7a47828b07\scratch\hybrid_algorithm_visualization.png"
    
    visualize_clusters(DATA_PATH, OUTPUT_PATH)
