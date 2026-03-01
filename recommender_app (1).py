"""
keskon regarde après — Movie Recommender
Streamlit App — version sans scikit-learn (stdlib + numpy + pandas uniquement)
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="keskon regarde après 🎬", page_icon="🎬", layout="wide")
st.title("🎬 keskon regarde après")
st.markdown("**Recommandation hybride** : Filtrage contenu + Collaboratif — *sans scikit-learn*")
st.divider()

# ─────────────────────────────────────────────
# FONCTIONS UTILITAIRES (remplacent sklearn)
# ─────────────────────────────────────────────

def cosine_sim_matrix(vec, matrix):
    """Cosine similarity entre un vecteur (1, d) et une matrice (n, d)."""
    norm_vec = np.linalg.norm(vec)
    norm_mat = np.linalg.norm(matrix, axis=1)
    if norm_vec == 0:
        return np.zeros(matrix.shape[0])
    denom = norm_mat * norm_vec
    denom[denom == 0] = 1e-10
    return matrix.dot(vec) / denom

def cosine_sim_rows(matrix):
    """Cosine similarity entre toutes les lignes d'une matrice."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normed = matrix / norms
    return normed.dot(normed.T)

def minmax_scale(arr):
    """Normalise un array numpy dans [0, 1]."""
    mn, mx = arr.min(axis=0), arr.max(axis=0)
    rng = mx - mn
    rng[rng == 0] = 1
    return (arr - mn) / rng

def multilabel_binarize(series):
    """One-hot encode une Series de listes (remplace MultiLabelBinarizer)."""
    all_labels = sorted(set(label for row in series for label in row))
    label_idx  = {l: i for i, l in enumerate(all_labels)}
    matrix     = np.zeros((len(series), len(all_labels)), dtype=np.float32)
    for i, row in enumerate(series):
        for label in row:
            matrix[i, label_idx[label]] = 1.0
    return matrix, all_labels

def knn_query(X_all, query_vec, k):
    """Retourne les indices des k plus proches voisins (cosinus)."""
    scores = cosine_sim_matrix(query_vec, X_all)
    return np.argsort(-scores)[:k+1]  # +1 car on exclut soi-même ensuite

# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
@st.cache_data
def load_data(ratings_file, users_file, movies_file):
    ratings = pd.read_csv(ratings_file, sep='::', engine='python',
                          names=['UserID','MovieID','Rating','Timestamp'])
    users   = pd.read_csv(users_file,   sep='::', engine='python',
                          names=['UserID','Gender','Age','Occupation','Zip'])
    movies  = pd.read_csv(movies_file,  sep='::', engine='python',
                          names=['MovieID','Title','Genres'], encoding='latin-1')
    return ratings, users, movies

@st.cache_data
def preprocess(_ratings, _users, _movies):
    movies = _movies.copy()
    users  = _users.copy()

    # Genres → vecteur binaire
    movies['GenreList'] = movies['Genres'].str.split('|')
    genre_matrix, genre_cols = multilabel_binarize(movies['GenreList'])
    genre_df = pd.DataFrame(genre_matrix, columns=genre_cols, index=movies.index)
    movies   = pd.concat([movies, genre_df], axis=1)

    # Encodage users
    users['GenderEnc'] = (users['Gender'] == 'M').astype(float)
    age_map = {1:0, 18:1, 25:2, 35:3, 45:4, 50:5, 56:6}
    users['AgeEnc']    = users['Age'].map(age_map).fillna(2).astype(float) / 6.0
    users['OccEnc']    = users['Occupation'].astype(float) / 20.0

    df = _ratings.merge(users, on='UserID').merge(movies, on='MovieID')
    user_feats = users.set_index('UserID')[['GenderEnc','AgeEnc','OccEnc']]

    return df, movies, genre_cols, user_feats

@st.cache_data
def build_genre_profiles(_df, genre_cols, threshold=4):
    high = _df[_df['Rating'] >= threshold]
    return high.groupby('UserID')[genre_cols].mean().fillna(0)

@st.cache_data
def build_combined_matrix(_user_genre_profile, _user_feats):
    common    = _user_genre_profile.index.intersection(_user_feats.index)
    X_genre   = _user_genre_profile.loc[common].values.astype(float)
    X_demo    = _user_feats.loc[common].values.astype(float)
    X_genre_n = minmax_scale(X_genre)
    X_combined = np.hstack([X_demo * 0.3, X_genre_n * 0.7])
    return X_combined, common  # common = index des UserID

# ─────────────────────────────────────────────
# MOTEUR DE RECOMMANDATION
# ─────────────────────────────────────────────
def recommend_existing(user_id, df, movies, genre_cols,
                        user_genre_profile, X_combined, user_index,
                        top_n=10, alpha=0.5, k=10):
    seen     = set(df[df['UserID'] == user_id]['MovieID'])
    uid_pos  = list(user_index).index(user_id)

    # Score contenu
    user_vec    = user_genre_profile.loc[user_id].values.astype(float)
    movie_genre = movies[genre_cols].values.astype(float)
    content_sc  = cosine_sim_matrix(user_vec, movie_genre)
    content_ser = pd.Series(content_sc, index=movies['MovieID'].values)

    # Score collaboratif
    query       = X_combined[uid_pos]
    neighbor_idx = knn_query(X_combined, query, k)
    neighbor_idx = [i for i in neighbor_idx if i != uid_pos][:k]
    neighbor_ids = [user_index[i] for i in neighbor_idx]
    nb_ratings   = df[df['UserID'].isin(neighbor_ids)][['MovieID','Rating']]
    collab_ser   = (nb_ratings.groupby('MovieID')['Rating'].mean() - 1) / 4

    # Fusion
    unseen_movies = movies[~movies['MovieID'].isin(seen)].copy()
    mid = unseen_movies['MovieID'].values
    scores = pd.DataFrame({'MovieID': mid})
    scores['content'] = content_ser.reindex(mid).fillna(0).values
    scores['collab']  = collab_ser.reindex(mid).fillna(0).values
    scores['hybrid']  = alpha * scores['collab'] + (1-alpha) * scores['content']
    scores = scores.nlargest(top_n, 'hybrid')

    result = unseen_movies.set_index('MovieID').loc[scores['MovieID'].values, ['Title','Genres']].copy()
    result['Score hybride %']  = (scores['hybrid'].values  * 100).round(1)
    result['Score contenu %']  = (scores['content'].values * 100).round(1)
    result['Score collabo. %'] = (scores['collab'].values  * 100).round(1)
    return result.reset_index(drop=True)


def cold_start_recommend(gender, age, occupation, df, movies, genre_cols,
                          user_feats, top_n=10):
    age_map  = {1:0, 18:1, 25:2, 35:3, 45:4, 50:5, 56:6}
    new_vec  = np.array([
        (1 if gender=='M' else 0) * 0.3,
        (age_map.get(age, 2) / 6.0)  * 0.3,
        (occupation / 20.0)           * 0.3
    ], dtype=float)

    X_demo   = user_feats.values.astype(float) * 0.3
    scores   = cosine_sim_matrix(new_vec, X_demo)
    top20_idx = np.argsort(-scores)[:20]
    nb_ids   = user_feats.index[top20_idx]

    nb_ratings = df[df['UserID'].isin(nb_ids)]
    top_movies = (
        nb_ratings[nb_ratings['Rating'] >= 4]
        .groupby('MovieID')
        .agg(MoyenneNote=('Rating','mean'), NbVotes=('Rating','count'))
        .query('NbVotes >= 3')
        .nlargest(top_n, 'MoyenneNote')
    )
    result = movies.set_index('MovieID').loc[top_movies.index, ['Title','Genres']].copy()
    result['Moyenne'] = top_movies['MoyenneNote'].round(2).values
    result['Nb votes'] = top_movies['NbVotes'].values
    return result.reset_index(drop=True)

# ─────────────────────────────────────────────
# INTERFACE
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Charger les données")
    ratings_up = st.file_uploader("ratings.dat", type="dat", key="rat")
    users_up   = st.file_uploader("users.dat",   type="dat", key="usr")
    movies_up  = st.file_uploader("movies.dat",  type="dat", key="mov")
    st.divider()
    st.header("⚙️ Paramètres")
    top_n       = st.slider("Top-N recommandations", 5, 20, 10)
    alpha       = st.slider("α  (0=Contenu  →  1=Collaboratif)", 0.0, 1.0, 0.5, 0.05)
    k_neighbors = st.slider("Voisins KNN", 5, 30, 10)

if not (ratings_up and users_up and movies_up):
    st.info("👈 Chargez vos trois fichiers `.dat` dans la barre latérale pour commencer.")
    st.stop()

with st.spinner("Chargement et prétraitement…"):
    ratings, users, movies = load_data(ratings_up, users_up, movies_up)
    df, movies_proc, genre_cols, user_feats = preprocess(ratings, users, movies)
    user_genre_profile = build_genre_profiles(df, genre_cols)
    X_combined, user_index = build_combined_matrix(user_genre_profile, user_feats)

# Stats
c1, c2, c3, c4 = st.columns(4)
c1.metric("👤 Utilisateurs", f"{users['UserID'].nunique():,}")
c2.metric("🎬 Films",        f"{movies['MovieID'].nunique():,}")
c3.metric("⭐ Notes",        f"{len(ratings):,}")
c4.metric("🎭 Genres",       len(genre_cols))
st.divider()

tab1, tab2 = st.tabs(["👤 Utilisateur existant", "🆕 Nouvel utilisateur (Cold-Start)"])

# ── TAB 1 ──
with tab1:
    st.subheader("Recommandations personnalisées")
    valid_users = sorted(user_index.tolist())
    user_id     = st.selectbox("Sélectionnez un utilisateur", valid_users)

    if st.button("🎯 Recommander", key="btn1"):
        info = users[users['UserID'] == user_id].iloc[0]
        st.markdown(f"**Profil** : Sexe={info['Gender']} | Âge={info['Age']} | Profession={info['Occupation']}")

        if user_id in user_genre_profile.index:
            prefs = user_genre_profile.loc[user_id].nlargest(5)
            st.markdown("**Top genres préférés** : " +
                " · ".join([f"`{g}` ({v:.0%})" for g, v in prefs.items() if v > 0]))

        with st.spinner("Calcul…"):
            recs = recommend_existing(
                user_id, df, movies_proc, genre_cols,
                user_genre_profile, X_combined, user_index,
                top_n=top_n, alpha=alpha, k=k_neighbors
            )

        st.success(f"✅ Top {top_n} pour l'utilisateur #{user_id}")
        st.dataframe(recs, use_container_width=True, hide_index=True)

        with st.expander("ℹ️ Comprendre les scores"):
            st.markdown(f"""
- **Score hybride** = `{alpha:.0%}` × collaboratif + `{1-alpha:.0%}` × contenu  
- **Score contenu** : similarité cosinus entre vos goûts et le genre du film  
- **Score collaboratif** : note moyenne des {k_neighbors} utilisateurs les plus proches
""")

# ── TAB 2 ──
with tab2:
    st.subheader("Recommandations pour un nouvel utilisateur")
    st.info("Aucune note requise — basé sur le profil démographique.")

    ca, cb, cc = st.columns(3)
    with ca:
        gender_cs = st.selectbox("Sexe", ["M", "F"])
    with cb:
        age_cs = st.selectbox("Tranche d'âge", [1,18,25,35,45,50,56],
            format_func=lambda x:{1:"<18",18:"18-24",25:"25-34",
                                   35:"35-44",45:"45-49",50:"50-55",56:"56+"}[x])
    with cc:
        occ_labels = {
            0:"Autre",1:"Académique",2:"Artiste",3:"Administration",4:"Étudiant",
            5:"Service client",6:"Santé",7:"Cadre",8:"Agriculteur",9:"Foyer",
            10:"Élève",11:"Avocat",12:"Programmeur",13:"Retraité",14:"Commercial",
            15:"Scientifique",16:"Indépendant",17:"Technicien",18:"Artisan",
            19:"Sans emploi",20:"Écrivain"
        }
        occ_cs = st.selectbox("Profession", list(occ_labels.keys()),
                               format_func=lambda x: f"{x} – {occ_labels[x]}")

    if st.button("🎯 Recommander", key="btn2"):
        with st.spinner("Recherche des voisins démographiques…"):
            cold = cold_start_recommend(
                gender_cs, age_cs, occ_cs,
                df, movies_proc, genre_cols, user_feats, top_n=top_n
            )
        st.success(f"✅ Top {top_n} — Cold Start")
        st.dataframe(cold, use_container_width=True, hide_index=True)
        st.caption("Ces suggestions s'affineront dès que vous aurez noté quelques films.")
