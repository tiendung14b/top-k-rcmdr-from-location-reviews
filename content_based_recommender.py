import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

def parse_json_file(filepath):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
                if isinstance(content, list):
                    data.extend(content)
                else:
                    data.append(content)
            except json.JSONDecodeError:
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    except Exception:
        pass
    return data

def load_data():
    all_data = []
    file1_path = 'reviews_data.json'
    if os.path.exists(file1_path):
        all_data.extend(parse_json_file(file1_path))
    file2_path = 'llm_enriched_reviews.json'
    if os.path.exists(file2_path):
        all_data.extend(parse_json_file(file2_path))

    df = pd.DataFrame(all_data)
    if df.empty:
        return df

    array_cols = ['target_audience', 'activities', 'value_perception', 'dominant_moods']
    string_cols = ['physical_intensity', 'accessibility', 'service_quality', 'crowd_density']
    cols_to_keep = ['location_id'] + array_cols + string_cols
    
    for col in cols_to_keep:
        if col not in df.columns:
            df[col] = None
            
    df = df[cols_to_keep].dropna(subset=['location_id']).copy()

    for col in array_cols:
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        
    for col in string_cols:
        df[col] = df[col].fillna("Missing").astype(str)
        
    return df

def custom_train_test_split(df, test_ratio=0.2):
    train_list, test_list = [], []
    for loc_id, group in df.groupby('location_id'):
        total_reviews = len(group)
        if total_reviews <= 1:
            # If only 1 review, put it in train so we can build a location profile
            train_list.append(group)
            continue
            
        n_test = int(total_reviews * test_ratio)
        if n_test == 0:
            n_test = 1 # Guarantee at least 1 test sample if we have >1 reviews
            
        if n_test == total_reviews:
            n_test -= 1 # Guarantee at least 1 train sample
            
        test_indices = random.sample(list(group.index), n_test)
        
        test_df = group.loc[test_indices]
        train_df = group.drop(test_indices)
        
        test_list.append(test_df)
        train_list.append(train_df)
        
    train_df = pd.concat(train_list) if train_list else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_list) if test_list else pd.DataFrame(columns=df.columns)
    return train_df, test_df

def encode_features(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    array_cols = ['target_audience', 'activities', 'value_perception', 'dominant_moods']
    string_cols = ['physical_intensity', 'accessibility', 'service_quality', 'crowd_density']
    
    train_features = []
    test_features = []
    feature_names = []
    
    for col in array_cols:
        train_strings = train_df[col].apply(lambda x: " ".join([str(item).replace(' ', '_') for item in x]))
        test_strings = test_df[col].apply(lambda x: " ".join([str(item).replace(' ', '_') for item in x]))
        
        tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        train_mat = tfidf.fit_transform(train_strings).toarray()
        test_mat = tfidf.transform(test_strings).toarray()
        
        if col in ['activities', 'target_audience']:
            train_mat *= 2.0
            test_mat *= 2.0
            
        train_features.append(train_mat)
        test_features.append(test_mat)
        feature_names.extend([f"{col}_{c}" for c in tfidf.get_feature_names_out()])
        
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    train_str_mat = ohe.fit_transform(train_df[string_cols])
    test_str_mat = ohe.transform(test_df[string_cols])
    
    train_features.append(train_str_mat)
    test_features.append(test_str_mat)
    feature_names.extend(ohe.get_feature_names_out(string_cols))
    
    X_train = np.hstack(train_features)
    X_test = np.hstack(test_features)
    
    X_train = normalize(X_train, norm='l2')
    X_test = normalize(X_test, norm='l2')
    
    train_encoded = pd.DataFrame(X_train, columns=feature_names, index=train_df.index)
    train_encoded['location_id'] = train_df['location_id']
    
    test_encoded = pd.DataFrame(X_test, columns=feature_names, index=test_df.index)
    test_encoded['location_id'] = test_df['location_id']
    
    return train_encoded, test_encoded, feature_names

def build_location_profiles_and_cluster(train_encoded, feature_names):
    location_profiles = train_encoded.groupby('location_id')[feature_names].mean()
    
    location_matrix = normalize(location_profiles.values, norm='l2')
    location_profiles = pd.DataFrame(location_matrix, columns=feature_names, index=location_profiles.index)
    
    n_locations = len(location_profiles)
    n_clusters = max(2, min(8, n_locations // 2)) if n_locations >= 2 else 1
    
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(location_profiles)
        location_profiles['cluster'] = clusters
    else:
        location_profiles['cluster'] = 0
        
    return location_profiles

def evaluate_multi_k(test_encoded, location_profiles, feature_names, k_list=[3, 5, 7, 9]):
    total = len(test_encoded)
    hits_at_k = {k: 0 for k in k_list}
    
    profile_features = location_profiles.drop(columns=['cluster'], errors='ignore')
    profile_matrix = profile_features.values
    location_ids = profile_features.index.tolist()
    
    for _, row in test_encoded.iterrows():
        true_location = row['location_id']
        user_vector = row[feature_names].values.reshape(1, -1)
        
        sim_scores = cosine_similarity(user_vector, profile_matrix)[0]
        
        sorted_indices = np.argsort(sim_scores)[::-1]
        max_k = max(k_list)
        actual_max_k = min(max_k, len(location_ids))
        top_k_indices = sorted_indices[:actual_max_k]
        top_k_recommendations = [location_ids[i] for i in top_k_indices]
        
        for k in k_list:
            if true_location in top_k_recommendations[:k]:
                hits_at_k[k] += 1
                
    hit_rates = {k: (hits / total) if total > 0 else 0.0 for k, hits in hits_at_k.items()}
    return hit_rates

def main():
    print("Initializing ADVANCED Recommender Pipeline (Proportional Split)...")
    df = load_data()
    if df.empty:
        return
    print(f"Loaded {len(df)} total reviews.")
    
    train_df, test_df = custom_train_test_split(df, test_ratio=0.2)
    print(f"Split completed -> Train Set: {len(train_df)}, Test Set: {len(test_df)}")
        
    train_encoded, test_encoded, feature_names = encode_features(train_df, test_df)
    location_profiles = build_location_profiles_and_cluster(train_encoded, feature_names)
    
    k_list = [3, 5, 7, 9]
    hit_rates = evaluate_multi_k(test_encoded, location_profiles, feature_names, k_list)
    print(f"\n====================================")
    print("PROPORTIONAL SPLIT EVALUATION RESULTS")
    for k in k_list:
        print(f"- Hit Rate@{k}: {hit_rates[k]:.2%}")
    print(f"====================================")

if __name__ == '__main__':
    main()
