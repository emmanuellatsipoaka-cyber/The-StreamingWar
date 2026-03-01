"""
MovieLens Hybrid Recommendation System — Streamlit App
=======================================================
Déploiement : placez ratings.dat, users.dat, movies.dat dans le même dossier.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Système de Recommandation de Films")
st.markdown("**Modèle hybride** : Filtrage par contenu + Filtrage collaboratif (KNN)")
st.divider()

# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES (avec cache)
# ─────────────────────────────────────────────
@st.cache_data
def load_data(ratings_file, users_file, movies_file):
    ratings = pd.read_csv(
        ratings_file, sep='::', engine='python',
        names=['UserID', 'MovieID', 'Rating', 'Timestamp']
    )
    users = pd.read_csv(
        users_file, sep='::', engine='python',
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip']
    )
    movies = pd.read_csv(
        movies_file, sep='::', engine='python',
        names=['MovieID', 'Title', 'Genres'], encoding='latin-1'
    )
    return ratings, users, movies


@st.cache_data
def preprocess(_ratings, _users, _movies):
    movies = _movies.copy()
    users  = _users.copy()

    # Genres → multi-label binaire
    movies['GenreList'] = movies['Genres'].str.split('|')
    mlb = MultiLabelBinarizer()
    genre_matrix = pd.DataFrame(
        mlb.fit_transform(movies['GenreList']),
        columns=mlb.classes_, index=movies.index
    )
    movies = pd.concat([movies, genre_matrix], axis=1)
    genre_cols = list(mlb.classes_)

    # Encodage Gender
    users['GenderEnc'] = (users['Gender'] == 'M').astype(int)

    # Encodage Age
    age_map = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
    users['AgeEnc'] = users['Age'].map(age_map).fillna(2).astype(int)

    # Normalisation Occupation
    scaler = MinMaxScaler()
    users['OccEnc'] = scaler.fit_transform(users[['Occupation']])

    # Fusion
    df = _ratings.merge(users, on='UserID').merge(movies, on='MovieID')

    # Features démographiques
    user_feats = users.set_index('UserID')[['GenderEnc', 'AgeEnc', 'OccEnc']]

    return df, movies, genre_cols, user_feats


@st.cache_data
def build_profiles(_df, genre_cols, threshold=4):
    high_rated = _df[_df['Rating'] >= threshold]
    return high_rated.groupby('UserID')[genre_cols].mean().fillna(0)


@st.cache_resource
def build_knn_model(_user_genre_profile, _user_feats, n_neighbors=10):
    common = _user_genre_profile.index.intersection(_user_feats.index)
    X_genre = _user_genre_profile.loc[common]
    X_demo  = _user_feats.loc[common]

    scaler = MinMaxScaler()
    X_genre_norm = pd.DataFrame(
        scaler.fit_transform(X_genre),
        index=X_genre.index, columns=X_genre.columns
    )

    X_combined = pd.concat([X_demo * 0.3, X_genre_norm * 0.7], axis=1)

    knn = NearestNeighbors(
        n_neighbors=min(n_neighbors + 1, len(X_combined)),
        metric='cosine', algorithm='brute'
    )
    knn.fit(X_combined.values)
    return knn, X_combined


# ─────────────────────────────────────────────
# RECOMMANDATIONS
# ─────────────────────────────────────────────
def recommend_existing(user_id, df, movies, genre_cols,
                        user_genre_profile, knn_model, X_combined,
                        top_n=10, alpha=0.5):
    seen = set(df[df['UserID'] == user_id]['MovieID'])

    # Score contenu
    user_vec    = user_genre_profile.loc[[user_id]].values
    movie_genre = movies.set_index('MovieID')[genre_cols]
    content_sc  = cosine_similarity(user_vec, movie_genre.values)[0]
    content_df  = pd.Series(content_sc, index=movie_genre.index)

    # Score collaboratif
    idx = X_combined.index.get_loc(user_id)
    _, indices = knn_model.kneighbors(X_combined.iloc[idx:idx+1].values)
    neighbor_ids = X_combined.index[indices[0][1:]]
    nb_ratings   = df[df['UserID'].isin(neighbor_ids)][['MovieID', 'Rating']]
    collab_sc    = (nb_ratings.groupby('MovieID')['Rating'].mean() - 1) / 4

    # Fusion
    candidates = movies[~movies['MovieID'].isin(seen)].set_index('MovieID')
    scores = pd.DataFrame(index=candidates.index)
    scores['content'] = content_df.reindex(scores.index).fillna(0)
    scores['collab']  = collab_sc.reindex(scores.index).fillna(0)
    scores['hybrid']  = alpha * scores['collab'] + (1 - alpha) * scores['content']

    top = scores.nlargest(top_n, 'hybrid')
    result = candidates.loc[top.index, ['Title', 'Genres']].copy()
    result['Score hybride'] = (top['hybrid'].values * 100).round(1)
    result['Score contenu'] = (top['content'].values * 100).round(1)
    result['Score collabo.'] = (top['collab'].values * 100).round(1)
    return result.reset_index(drop=True)


def cold_start_recommend(gender, age, occupation,
                          df, movies, genre_cols,
                          user_feats, top_n=10):
    age_map   = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
    g_enc     = 1 if gender == 'M' else 0
    a_enc     = age_map.get(age, 2)
    o_enc     = occupation / 20.0
    new_vec   = np.array([[g_enc * 0.3, a_enc * 0.3, o_enc * 0.3]])

    demo_mat = user_feats * 0.3
    knn_d    = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    knn_d.fit(demo_mat.values)
    _, indices  = knn_d.kneighbors(new_vec)
    neighbor_ids = user_feats.index[indices[0]]

    nb_ratings = df[df['UserID'].isin(neighbor_ids)]
    top_movies = (
        nb_ratings[nb_ratings['Rating'] >= 4]
        .groupby('MovieID')
        .agg(MoyenneNote=('Rating', 'mean'), NbVotes=('Rating', 'count'))
        .query('NbVotes >= 3')
        .nlargest(top_n, 'MoyenneNote')
    )
    result = movies.set_index('MovieID').loc[top_movies.index, ['Title', 'Genres']].copy()
    result['Moyenne (voisins)'] = top_movies['MoyenneNote'].round(2).values
    result['Nb votes']          = top_movies['NbVotes'].values
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────
# INTERFACE STREAMLIT
# ─────────────────────────────────────────────

# ── Sidebar : upload des fichiers ──
with st.sidebar:
    st.header("📁 Données")
    ratings_up = st.file_uploader("ratings.dat", type="dat", key="rat")
    users_up   = st.file_uploader("users.dat",   type="dat", key="usr")
    movies_up  = st.file_uploader("movies.dat",  type="dat", key="mov")

    st.divider()
    st.header("⚙️ Paramètres")
    top_n = st.slider("Nombre de recommandations (Top-N)", 5, 20, 10)
    alpha = st.slider("α (0=Contenu, 1=Collaboratif)", 0.0, 1.0, 0.5, 0.05)
    k_neighbors = st.slider("Nombre de voisins KNN", 5, 30, 10)

# ── Chargement ──
if not (ratings_up and users_up and movies_up):
    st.info("👈 Chargez vos trois fichiers `.dat` dans la barre latérale pour commencer.")
    st.stop()

with st.spinner("Chargement et prétraitement des données…"):
    ratings, users, movies = load_data(ratings_up, users_up, movies_up)
    df, movies_proc, genre_cols, user_feats = preprocess(ratings, users, movies)
    user_genre_profile = build_profiles(df, genre_cols)
    knn_model, X_combined = build_knn_model(user_genre_profile, user_feats, k_neighbors)

# ── Stats rapides ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("👤 Utilisateurs", f"{users['UserID'].nunique():,}")
col2.metric("🎬 Films",        f"{movies['MovieID'].nunique():,}")
col3.metric("⭐ Notes",        f"{len(ratings):,}")
col4.metric("🎭 Genres",       len(genre_cols))
st.divider()

# ── Onglets ──
tab1, tab2 = st.tabs(["👤 Utilisateur existant", "🆕 Nouvel utilisateur (Cold-Start)"])

# ── TAB 1 : Utilisateur existant ──
with tab1:
    st.subheader("Recommandations personnalisées")

    valid_users = sorted(X_combined.index.tolist())
    user_id = st.selectbox("Sélectionnez un utilisateur", valid_users)

    if st.button("🎯 Générer les recommandations", key="btn_existing"):
        user_info = users[users['UserID'] == user_id].iloc[0]
        st.markdown(f"**Profil** : Genre={user_info['Gender']} | Âge={user_info['Age']} | Profession={user_info['Occupation']}")

        # Genres préférés
        if user_id in user_genre_profile.index:
            prefs = user_genre_profile.loc[user_id].nlargest(5)
            st.markdown("**Top 5 genres préférés** : " + " · ".join(
                [f"`{g}` ({v:.0%})" for g, v in prefs.items() if v > 0]
            ))

        with st.spinner("Calcul des recommandations…"):
            recs = recommend_existing(
                user_id, df, movies_proc, genre_cols,
                user_genre_profile, knn_model, X_combined,
                top_n=top_n, alpha=alpha
            )

        st.success(f"✅ Top {top_n} recommandations pour l'utilisateur #{user_id}")
        st.dataframe(
            recs[['Title', 'Genres', 'Score hybride', 'Score contenu', 'Score collabo.']],
            use_container_width=True, hide_index=True
        )

        with st.expander("ℹ️ Explication des scores"):
            st.markdown(f"""
            - **Score hybride** = `{alpha:.0%}` × Score collab + `{1-alpha:.0%}` × Score contenu
            - **Score contenu** : similarité cosinus entre vos genres préférés et le genre du film
            - **Score collaboratif** : note moyenne des {k_neighbors} utilisateurs les plus similaires
            """)

# ── TAB 2 : Cold-Start ──
with tab2:
    st.subheader("Recommandations pour un nouvel utilisateur")
    st.info("Aucune note requise — les recommandations sont basées sur votre profil démographique.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        gender_cs = st.selectbox("Sexe", ["M", "F"])
    with col_b:
        age_cs = st.selectbox("Tranche d'âge", [1, 18, 25, 35, 45, 50, 56],
                               format_func=lambda x: {1:"<18", 18:"18-24", 25:"25-34",
                                                       35:"35-44", 45:"45-49",
                                                       50:"50-55", 56:"56+"}[x])
    with col_c:
        occ_labels = {
            0:"Autre/non précisé", 1:"Académique/Éducateur", 2:"Artiste",
            3:"Administration", 4:"Étudiant", 5:"Service client",
            6:"Médecin/Santé", 7:"Cadre/Manager", 8:"Agriculteur",
            9:"Femme/homme au foyer", 10:"K-12 Élève", 11:"Avocat",
            12:"Programmeur", 13:"Retraité", 14:"Commercial",
            15:"Scientifique", 16:"Travailleur indépendant", 17:"Technicien",
            18:"Artisan", 19:"Sans emploi", 20:"Auteur/Écrivain"
        }
        occ_cs = st.selectbox("Profession", list(occ_labels.keys()),
                               format_func=lambda x: f"{x} – {occ_labels[x]}")

    if st.button("🎯 Générer les recommandations", key="btn_cold"):
        with st.spinner("Recherche des utilisateurs similaires…"):
            cold_recs = cold_start_recommend(
                gender_cs, age_cs, occ_cs,
                df, movies_proc, genre_cols, user_feats, top_n=top_n
            )

        st.success(f"✅ Top {top_n} recommandations (Cold-Start)")
        st.dataframe(cold_recs, use_container_width=True, hide_index=True)

        st.caption(
            "Ces recommandations proviennent des 20 utilisateurs démographiquement "
            "les plus proches. Elles s'affineront dès que vous aurez noté quelques films."
        )
