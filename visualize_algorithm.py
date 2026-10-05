import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random

# Import functions from our existing recommender script
from content_based_recommender import load_data, custom_train_test_split, encode_features, build_location_profiles_and_cluster
import warnings
warnings.filterwarnings('ignore')

def visualize():
    print("Loading data for visualization...")
    df = load_data()
    train_df, test_df = custom_train_test_split(df, test_ratio=0.2)
    train_encoded, test_encoded, feature_names = encode_features(train_df, test_df)
    location_profiles = build_location_profiles_and_cluster(train_encoded, feature_names)
    
    # We drop 'cluster' column to just get the raw high-dimensional vectors
    profile_features = location_profiles.drop(columns=['cluster'], errors='ignore')
    
    # Let's select Top 5 locations with the most test samples to make the plot clean
    top_5_locations = test_encoded['location_id'].value_counts().head(5).index.tolist()
    
    # Filter test users and location profiles to only these top 5
    test_subset = test_encoded[test_encoded['location_id'].isin(top_5_locations)]
    profile_subset = profile_features.loc[top_5_locations]
    
    # Combine data for PCA to ensure they share the same 2D transformation space
    # 1. Location profiles (5 rows)
    # 2. Test user vectors (N rows)
    all_vectors = np.vstack([profile_subset.values, test_subset[feature_names].values])
    
    # Reduce the 96-dimensional vectors to 2 dimensions using PCA
    pca = PCA(n_components=2, random_state=42)
    reduced_vectors = pca.fit_transform(all_vectors)
    
    # Split back
    reduced_profiles = reduced_vectors[:5]
    reduced_users = reduced_vectors[5:]
    
    # Setup plot
    plt.figure(figsize=(14, 9))
    
    # 5 distinct colors for our 5 selected locations
    colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231']
    
    # Plot user vectors
    for idx, loc_id in enumerate(top_5_locations):
        # Find users who actually belong to this location
        user_indices = np.where(test_subset['location_id'] == loc_id)[0]
        user_points = reduced_users[user_indices]
        
        # Plot Users (Small dots)
        plt.scatter(user_points[:, 0], user_points[:, 1], 
                    color=colors[idx], alpha=0.6, s=40, label=f"User Reviews ({loc_id[:8]}...)")
                    
        # Plot the Location Profile (Giant Star)
        plt.scatter(reduced_profiles[idx, 0], reduced_profiles[idx, 1], 
                    color=colors[idx], marker='*', s=800, edgecolors='black', 
                    label=f"Location Profile ({loc_id[:8]}...)")
                    
        # Draw lines from a few users to their profile to visualize "distance/similarity" computation
        for i in range(min(5, len(user_points))):
            plt.plot([reduced_profiles[idx, 0], user_points[i, 0]], 
                     [reduced_profiles[idx, 1], user_points[i, 1]], 
                     color=colors[idx], linestyle='--', alpha=0.3)

    plt.title("Thuật toán Content-Based Filtering: Không gian Vector (Giảm chiều PCA xuống 2D)\nMỗi Review (Chấm tròn) sẽ được gợi ý Địa điểm (Ngôi sao) có khoảng cách hình học gần nhất", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel(f"Chiều PCA 1 (Giải thích {pca.explained_variance_ratio_[0]:.1%} độ bao phủ dữ liệu)", fontsize=12)
    plt.ylabel(f"Chiều PCA 2 (Giải thích {pca.explained_variance_ratio_[1]:.1%} độ bao phủ dữ liệu)", fontsize=12)
    
    # Put a legend to the right of the current axis
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Chú thích", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig('algorithm_visualization.png', dpi=300)
    print("Success: Graph saved as algorithm_visualization.png!")

if __name__ == "__main__":
    visualize()
