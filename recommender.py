"""
MovieLens Hybrid Recommendation System
======================================
Architecture: Content-Based Filtering + Collaborative Filtering (KNN)
Dataset: MovieLens 1M (ratings.dat, users.dat, movies.dat)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================

def load_movielens(data_dir='.'):
    """Load and parse MovieLens 1M dataset."""

    # --- Ratings ---
    ratings = pd.read_csv(
        f'{data_dir}/ratings.dat',
        sep='::',
        engine='python',
        names=['UserID', 'MovieID', 'Rating', 'Timestamp']
    )

    # --- Users ---
    users = pd.read_csv(
        f'{data_dir}/users.dat',
        sep='::',
        engine='python',
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip']
    )

    # --- Movies ---
    movies = pd.read_csv(
        f'{data_dir}/movies.dat',
        sep='::',
        engine='python',
        names=['MovieID', 'Title', 'Genres'],
        encoding='latin-1'
    )

    return ratings, users, movies


def preprocess(ratings, users, movies):
    """
    Merge datasets and encode categorical variables.
    Returns:
        df        : merged DataFrame
        genre_cols: list of genre column names
        user_feats: DataFrame of encoded user features (for cold-start)
    """

    # Split genres into list
    movies['GenreList'] = movies['Genres'].str.split('|')

    # One-hot encode genres
    mlb = MultiLabelBinarizer()
    genre_matrix = pd.DataFrame(
        mlb.fit_transform(movies['GenreList']),
        columns=mlb.classes_,
        index=movies.index
    )
    movies = pd.concat([movies, genre_matrix], axis=1)
    genre_cols = list(mlb.classes_)

    # Encode Gender (M=1, F=0)
    le_gender = LabelEncoder()
    users['GenderEnc'] = le_gender.fit_transform(users['Gender'])  # F=0, M=1

    # Age buckets → ordinal
    age_map = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
    users['AgeEnc'] = users['Age'].map(age_map).fillna(0).astype(int)

    # Occupation is already numeric (0-20) — normalise to [0,1]
    scaler = MinMaxScaler()
    users['OccEnc'] = scaler.fit_transform(users[['Occupation']])

    # Merge all
    df = ratings.merge(users, on='UserID').merge(movies, on='MovieID')

    # User feature matrix (demographic)
    user_feats = users.set_index('UserID')[['GenderEnc', 'AgeEnc', 'OccEnc']]

    return df, movies, genre_cols, user_feats


# ============================================================
# 2. CONTENT-BASED PROFILE (genre preferences)
# ============================================================

def build_content_profiles(df, genre_cols, threshold=4):
    """
    For each user, compute a weighted genre preference vector
    from high-rated movies (Rating >= threshold).
    """
    high_rated = df[df['Rating'] >= threshold]

    # Mean genre score per user
    user_genre_profile = (
        high_rated.groupby('UserID')[genre_cols]
        .mean()
        .fillna(0)
    )
    return user_genre_profile  # shape: (n_users, n_genres)


# ============================================================
# 3. COLLABORATIVE FILTERING (KNN on demographic + genre)
# ============================================================

def build_collab_model(user_genre_profile, user_feats, n_neighbors=10):
    """
    Build KNN model combining demographic features and genre preferences.
    Hybrid feature vector = [demographic (3 dims)] + [genre preference (18 dims)]
    """
    # Align indices
    common_users = user_genre_profile.index.intersection(user_feats.index)
    X_genre = user_genre_profile.loc[common_users]
    X_demo  = user_feats.loc[common_users]

    # Normalise genre profile
    scaler = MinMaxScaler()
    X_genre_norm = pd.DataFrame(
        scaler.fit_transform(X_genre),
        index=X_genre.index,
        columns=X_genre.columns
    )

    # Concatenate (weight demographics 0.3, genres 0.7)
    X_combined = pd.concat([X_demo * 0.3, X_genre_norm * 0.7], axis=1)

    knn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,  # +1 because user itself is included
        metric='cosine',
        algorithm='brute'
    )
    knn.fit(X_combined.values)

    return knn, X_combined


# ============================================================
# 4. RECOMMENDATION ENGINE
# ============================================================

def recommend_for_user(
    user_id,
    df,
    movies,
    genre_cols,
    user_genre_profile,
    knn_model,
    X_combined,
    top_n=10,
    alpha=0.5          # blend weight: 0=pure CBF, 1=pure CF
):
    """
    Hybrid recommendation for an existing user.

    Strategy:
      - Content score  : cosine(user_genre_profile, movie_genre_vector)
      - Collab score   : mean rating of similar users on unseen movies
      - Final score    : alpha * collab + (1-alpha) * content
    """

    if user_id not in X_combined.index:
        raise ValueError(f"User {user_id} not found. Use cold_start_recommend() instead.")

    # Movies already seen
    seen = set(df[df['UserID'] == user_id]['MovieID'])

    # --- Content-Based Score ---
    user_vec = user_genre_profile.loc[[user_id]].values  # (1, n_genres)
    movie_genres = movies.set_index('MovieID')[genre_cols]
    content_scores = cosine_similarity(user_vec, movie_genres.values)[0]
    content_df = pd.Series(content_scores, index=movie_genres.index, name='ContentScore')

    # --- Collaborative Score ---
    user_idx = X_combined.index.get_loc(user_id)
    distances, indices = knn_model.kneighbors(
        X_combined.iloc[user_idx:user_idx+1].values
    )
    neighbor_ids = X_combined.index[indices[0][1:]]  # exclude self

    neighbor_ratings = df[df['UserID'].isin(neighbor_ids)][['MovieID', 'Rating']]
    collab_scores = neighbor_ratings.groupby('MovieID')['Rating'].mean()
    collab_scores = (collab_scores - 1) / 4  # normalise [1,5] → [0,1]

    # --- Combine ---
    candidates = movies[~movies['MovieID'].isin(seen)].set_index('MovieID')
    scores = pd.DataFrame(index=candidates.index)
    scores['content'] = content_df.reindex(scores.index).fillna(0)
    scores['collab']  = collab_scores.reindex(scores.index).fillna(0)
    scores['hybrid']  = alpha * scores['collab'] + (1 - alpha) * scores['content']

    top = scores.nlargest(top_n, 'hybrid')
    result = candidates.loc[top.index, ['Title', 'Genres']].copy()
    result['Score'] = top['hybrid'].values
    return result.reset_index()


# ============================================================
# 5. COLD-START STRATEGY (new user)
# ============================================================

def cold_start_recommend(
    gender, age, occupation,
    df, movies, genre_cols,
    user_genre_profile, user_feats,
    top_n=10
):
    """
    Recommend for a brand-new user using demographic similarity only.
    Find K nearest existing users by demographic profile,
    then return their top-rated movies.
    """
    age_map = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
    gender_enc = 1 if gender.upper() == 'M' else 0
    age_enc    = age_map.get(age, 2)
    occ_enc    = occupation / 20.0  # normalise

    new_vec = np.array([[gender_enc * 0.3, age_enc * 0.3, occ_enc * 0.3]])

    # KNN on demographics only
    demo_matrix = user_feats * 0.3
    knn_demo = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    knn_demo.fit(demo_matrix.values)

    distances, indices = knn_demo.kneighbors(new_vec)
    neighbor_ids = user_feats.index[indices[0]]

    # Aggregate neighbor ratings
    neighbor_ratings = df[df['UserID'].isin(neighbor_ids)][['MovieID', 'Rating']]
    top_movies = (
        neighbor_ratings[neighbor_ratings['Rating'] >= 4]
        .groupby('MovieID')
        .agg(MeanRating=('Rating', 'mean'), Count=('Rating', 'count'))
        .query('Count >= 3')
        .nlargest(top_n, 'MeanRating')
    )
    result = movies.set_index('MovieID').loc[top_movies.index, ['Title', 'Genres']].copy()
    result['MeanRating'] = top_movies['MeanRating'].values
    return result.reset_index()


# ============================================================
# 6. EVALUATION
# ============================================================

def evaluate_model(df, movies, genre_cols, user_feats, test_size=0.2, top_k=10):
    """
    Evaluate using:
      - RMSE  (rating prediction proxy via collaborative mean)
      - Precision@K and Recall@K (relevant = Rating >= 4)
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Rebuild profiles on train set
    user_genre_profile = build_content_profiles(train_df, genre_cols)
    knn_model, X_combined = build_collab_model(user_genre_profile, user_feats)

    precisions, recalls, rmses = [], [], []

    sample_users = test_df['UserID'].value_counts().head(200).index

    for uid in sample_users:
        try:
            # Ground truth: movies rated ≥4 in test set
            test_user = test_df[test_df['UserID'] == uid]
            relevant  = set(test_user[test_user['Rating'] >= 4]['MovieID'])
            if len(relevant) == 0:
                continue

            # Get recommendations
            recs = recommend_for_user(
                uid, train_df, movies, genre_cols,
                user_genre_profile, knn_model, X_combined, top_n=top_k
            )
            rec_set = set(recs['MovieID'])

            # Precision@K and Recall@K
            hits = len(rec_set & relevant)
            precisions.append(hits / top_k)
            recalls.append(hits / len(relevant))

            # RMSE approximation (collab mean rating vs actual)
            user_idx = X_combined.index.get_loc(uid)
            distances, indices = knn_model.kneighbors(
                X_combined.iloc[user_idx:user_idx+1].values
            )
            neighbor_ids = X_combined.index[indices[0][1:]]
            neighbor_ratings = train_df[train_df['UserID'].isin(neighbor_ids)]

            for _, row in test_user.iterrows():
                movie_neighbors = neighbor_ratings[
                    neighbor_ratings['MovieID'] == row['MovieID']
                ]['Rating']
                if len(movie_neighbors) > 0:
                    pred = movie_neighbors.mean()
                    rmses.append((pred - row['Rating'])**2)

        except Exception:
            continue

    print("=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Precision@{top_k}  : {np.mean(precisions):.4f}")
    print(f"Recall@{top_k}     : {np.mean(recalls):.4f}")
    print(f"RMSE (collab)   : {np.sqrt(np.mean(rmses)):.4f}" if rmses else "RMSE: N/A")
    print("=" * 40)

    return {
        'precision_at_k': np.mean(precisions),
        'recall_at_k': np.mean(recalls),
        'rmse': np.sqrt(np.mean(rmses)) if rmses else None
    }


# ============================================================
# 7. MAIN — DEMO
# ============================================================

if __name__ == '__main__':
    DATA_DIR = '.'  # Update to your directory containing .dat files

    print("Loading data...")
    ratings, users, movies = load_movielens(DATA_DIR)
    print(f"  Ratings: {len(ratings):,} | Users: {len(users):,} | Movies: {len(movies):,}")

    print("Preprocessing...")
    df, movies, genre_cols, user_feats = preprocess(ratings, users, movies)

    print("Building content profiles...")
    user_genre_profile = build_content_profiles(df, genre_cols)

    print("Building KNN collaborative model...")
    knn_model, X_combined = build_collab_model(user_genre_profile, user_feats)

    # --- Demo: Recommend for existing user ---
    demo_user = 1
    print(f"\nTop-10 recommendations for User {demo_user}:")
    recs = recommend_for_user(
        demo_user, df, movies, genre_cols,
        user_genre_profile, knn_model, X_combined, top_n=10
    )
    print(recs[['Title', 'Genres', 'Score']].to_string(index=False))

    # --- Demo: Cold-start new user ---
    print("\nCold-start recommendations (Female, Age 25, Occupation 4):")
    cold_recs = cold_start_recommend(
        gender='F', age=25, occupation=4,
        df=df, movies=movies, genre_cols=genre_cols,
        user_genre_profile=user_genre_profile, user_feats=user_feats
    )
    print(cold_recs[['Title', 'Genres', 'MeanRating']].to_string(index=False))

    # --- Evaluation ---
    print("\nRunning evaluation (this may take a moment)...")
    metrics = evaluate_model(df, movies, genre_cols, user_feats)
