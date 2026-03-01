"""
keskon regarde après 🎬
Application de recommandation de films avec inscription / connexion / notation.
Dépendances : streamlit, pandas, numpy uniquement (pas de scikit-learn).
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import json
import os
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
# CONFIG & CSS
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="keskon regarde après 🎬",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0f;
    color: #f0f0f0;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1100px; }

/* Title */
.app-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    letter-spacing: 0.04em;
    background: linear-gradient(90deg, #e50914, #ff6b35, #f5c518);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0; padding: 0;
    line-height: 1;
}
.app-sub {
    color: #999;
    font-size: 1rem;
    margin-top: 0.3rem;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Auth card */
.auth-card {
    background: #18181b;
    border: 1px solid #2a2a2e;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.auth-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 0.06em;
    color: #f5c518;
    margin-bottom: 1.2rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #e50914, #c0000e) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(229,9,20,0.35) !important;
}

/* Movie card */
.movie-card {
    background: #18181b;
    border: 1px solid #2a2a2e;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.movie-card:hover { border-color: #e50914; }
.movie-title-text {
    font-weight: 600;
    font-size: 1rem;
    color: #f0f0f0;
}
.movie-genre-badge {
    display: inline-block;
    background: #2a2a2e;
    color: #aaa;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-right: 4px;
    margin-top: 4px;
}

/* Score badge */
.score-high   { color: #f5c518; font-weight: 700; }
.score-medium { color: #ff6b35; font-weight: 700; }
.score-low    { color: #888;    font-weight: 500; }

/* Section headers */
.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 0.06em;
    color: #f0f0f0;
    margin: 1.5rem 0 1rem 0;
    border-left: 4px solid #e50914;
    padding-left: 0.7rem;
}

/* Tabs override */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: 1px solid #2a2a2e;
}
.stTabs [data-baseweb="tab"] {
    background: #18181b !important;
    border-radius: 8px 8px 0 0 !important;
    color: #aaa !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border: 1px solid #2a2a2e !important;
    border-bottom: none !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: #0d0d0f !important;
    color: #f5c518 !important;
    border-color: #e50914 !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #1e1e22 !important;
    border: 1px solid #3a3a3e !important;
    border-radius: 8px !important;
    color: #f0f0f0 !important;
}
.stTextInput > label, .stSelectbox > label,
.stNumberInput > label { color: #aaa !important; font-size: 0.85rem !important; }

/* Alert / info */
.stAlert { border-radius: 10px !important; }

/* Stars display */
.stars { font-size: 1.1rem; letter-spacing: 2px; }

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid #2a2a2e;
    margin: 1.5rem 0;
}

/* Welcome banner */
.welcome-banner {
    background: linear-gradient(135deg, #18181b 0%, #1e1e22 100%);
    border: 1px solid #2a2a2e;
    border-radius: 16px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PERSISTANCE UTILISATEURS (storage Streamlit)
# ══════════════════════════════════════════════════════════

def hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

async def get_users_db():
    try:
        result = await st.context.storage.get("users_db")
        if result:
            return json.loads(result["value"])
    except:
        pass
    return {}

async def save_users_db(db: dict):
    try:
        await st.context.storage.set("users_db", json.dumps(db))
    except:
        pass

async def get_user_ratings(username: str) -> dict:
    try:
        result = await st.context.storage.get(f"ratings_{username}")
        if result:
            return json.loads(result["value"])
    except:
        pass
    return {}

async def save_user_ratings(username: str, ratings: dict):
    try:
        await st.context.storage.set(f"ratings_{username}", json.dumps(ratings))
    except:
        pass

# Fallback en mémoire si storage indisponible
def get_users_db_sync():
    return st.session_state.get("_users_db", {})

def save_users_db_sync(db):
    st.session_state["_users_db"] = db

def get_user_ratings_sync(username):
    return st.session_state.get(f"_ratings_{username}", {})

def save_user_ratings_sync(username, ratings):
    st.session_state[f"_ratings_{username}"] = ratings


# ══════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES MOVIELENS
# ══════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_movielens(ratings_f, users_f, movies_f):
    ratings = pd.read_csv(ratings_f, sep="::", engine="python",
                          names=["UserID","MovieID","Rating","Timestamp"])
    users   = pd.read_csv(users_f,   sep="::", engine="python",
                          names=["UserID","Gender","Age","Occupation","Zip"])
    movies  = pd.read_csv(movies_f,  sep="::", engine="python",
                          names=["MovieID","Title","Genres"], encoding="latin-1")
    return ratings, users, movies


@st.cache_data(show_spinner=False)
def preprocess_movies(_movies):
    movies = _movies.copy()
    movies["GenreList"] = movies["Genres"].str.split("|")
    all_genres = sorted({g for row in movies["GenreList"] for g in row})
    for g in all_genres:
        movies[g] = movies["GenreList"].apply(lambda gl: int(g in gl))
    return movies, all_genres


# ══════════════════════════════════════════════════════════
# MOTEUR DE RECOMMANDATION (100% numpy)
# ══════════════════════════════════════════════════════════

def cosine_sim(vec, matrix):
    """Similarité cosinus entre vecteur (d,) et matrice (n, d)."""
    nv = np.linalg.norm(vec)
    nm = np.linalg.norm(matrix, axis=1)
    if nv < 1e-10:
        return np.zeros(matrix.shape[0])
    denom = nm * nv
    denom[denom < 1e-10] = 1e-10
    return matrix.dot(vec) / denom


def build_user_genre_vector(user_ratings: dict, movies: pd.DataFrame, genre_cols: list) -> np.ndarray:
    """
    Construit le vecteur de préférences de genre d'un utilisateur
    à partir de ses notes (MovieID → note).
    Seules les notes ≥ 4 sont prises en compte.
    """
    if not user_ratings:
        return np.zeros(len(genre_cols))

    mid_to_genres = movies.set_index("MovieID")[genre_cols]
    weighted = np.zeros(len(genre_cols))
    total_w  = 0.0

    for movie_id_str, rating in user_ratings.items():
        rating = float(rating)
        if rating < 4:
            continue
        mid = int(movie_id_str)
        if mid not in mid_to_genres.index:
            continue
        w = rating - 3  # poids : 4→1, 5→2
        weighted += mid_to_genres.loc[mid].values.astype(float) * w
        total_w  += w

    if total_w > 0:
        weighted /= total_w
    return weighted


def recommend(user_ratings: dict, movies: pd.DataFrame, genre_cols: list,
              top_n: int = 12) -> pd.DataFrame:
    """
    Recommande des films non encore notés basé sur la similarité
    de genres avec le profil utilisateur.
    """
    rated_ids = {int(k) for k in user_ratings.keys()}
    user_vec  = build_user_genre_vector(user_ratings, movies, genre_cols)

    candidates = movies[~movies["MovieID"].isin(rated_ids)].copy()
    if candidates.empty:
        return pd.DataFrame()

    genre_matrix = candidates[genre_cols].values.astype(float)
    scores       = cosine_sim(user_vec, genre_matrix)

    candidates = candidates.copy()
    candidates["score"] = scores

    # Bonus popularité légère (nombre de genres en commun)
    overlap = genre_matrix.dot(user_vec > 0).astype(float)
    candidates["score"] += overlap * 0.05

    result = candidates.nlargest(top_n, "score")[
        ["MovieID", "Title", "Genres", "score"]
    ].reset_index(drop=True)
    result["score"] = (result["score"] * 100).round(1)
    return result


# ══════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════

def init_session():
    for key, val in {
        "logged_in": False,
        "username":  "",
        "page":      "auth",        # auth | rate | recommend
        "auth_tab":  "login",
        "user_ratings": {},
        "data_loaded": False,
        "movies_proc": None,
        "genre_cols":  None,
        "_users_db":   {},
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════

col_logo, col_nav = st.columns([3, 2])
with col_logo:
    st.markdown('<p class="app-title">KESKON REGARDE APRÈS</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-sub">Votre recommandation de films personnalisée 🎬</p>',
                unsafe_allow_html=True)

if st.session_state.logged_in:
    with col_nav:
        st.markdown("<br>", unsafe_allow_html=True)
        n1, n2, n3 = st.columns(3)
        if n1.button("🎬 Accueil"):
            st.session_state.page = "rate"
        if n2.button("⭐ Mes notes"):
            st.session_state.page = "my_ratings"
        if n3.button("🚪 Déconnexion"):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.session_state.page      = "auth"
            st.session_state.user_ratings = {}
            st.rerun()

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES (sidebar)
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📁 Données MovieLens")
    ratings_up = st.file_uploader("ratings.dat", type="dat", key="r_up")
    users_up   = st.file_uploader("users.dat",   type="dat", key="u_up")
    movies_up  = st.file_uploader("movies.dat",  type="dat", key="m_up")

    if ratings_up and users_up and movies_up and not st.session_state.data_loaded:
        with st.spinner("Chargement…"):
            _, _, movies_raw = load_movielens(ratings_up, users_up, movies_up)
            movies_proc, genre_cols = preprocess_movies(movies_raw)
            st.session_state.movies_proc  = movies_proc
            st.session_state.genre_cols   = genre_cols
            st.session_state.data_loaded  = True
        st.success(f"✅ {len(movies_proc):,} films chargés")

    if st.session_state.data_loaded:
        st.info(f"🎭 {len(st.session_state.genre_cols)} genres · {len(st.session_state.movies_proc):,} films")


# ══════════════════════════════════════════════════════════
# PAGE : AUTHENTIFICATION
# ══════════════════════════════════════════════════════════

def page_auth():
    col_c, col_f = st.columns([1, 1], gap="large")

    with col_c:
        st.markdown("""
        <div style="padding: 2rem 0;">
            <p style="font-size:3rem; margin:0;">🍿</p>
            <h2 style="font-family:'Bebas Neue',sans-serif;font-size:2rem;color:#f5c518;margin:0.5rem 0;">
                Découvrez votre prochain film préféré
            </h2>
            <p style="color:#888;line-height:1.7;">
                Notez les films que vous avez vus.<br>
                L'algorithme apprend vos goûts.<br>
                Recevez des recommandations sur mesure.
            </p>
            <div style="margin-top:1.5rem;padding:1rem;background:#18181b;
                        border-radius:10px;border-left:3px solid #e50914;">
                <p style="color:#ccc;font-size:0.85rem;margin:0;">
                    ✅ Inscription gratuite<br>
                    ⭐ Notez à votre rythme<br>
                    🎯 Recommandations dès 3 notes
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_f:
        tab_login, tab_signup = st.tabs(["🔑 Connexion", "✨ Inscription"])

        # ── LOGIN ──
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            login_user = st.text_input("Nom d'utilisateur", key="li_user",
                                        placeholder="ex: cinemaphile42")
            login_pass = st.text_input("Mot de passe", type="password", key="li_pass",
                                        placeholder="••••••••")

            if st.button("Se connecter →", key="btn_login", use_container_width=True):
                db = get_users_db_sync()
                if login_user in db and db[login_user]["pw"] == hash_pw(login_pass):
                    st.session_state.logged_in    = True
                    st.session_state.username     = login_user
                    st.session_state.page         = "rate"
                    st.session_state.user_ratings = get_user_ratings_sync(login_user)
                    st.rerun()
                else:
                    st.error("❌ Identifiants incorrects")

        # ── SIGNUP ──
        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            su_user  = st.text_input("Choisissez un pseudo", key="su_user",
                                      placeholder="ex: filmomaniac")
            su_pass  = st.text_input("Mot de passe", type="password", key="su_pass",
                                      placeholder="Min. 4 caractères")
            su_pass2 = st.text_input("Confirmer le mot de passe", type="password",
                                      key="su_pass2", placeholder="Répétez le mot de passe")

            if st.button("Créer mon compte →", key="btn_signup", use_container_width=True):
                db = get_users_db_sync()
                if len(su_user) < 3:
                    st.error("Le pseudo doit faire au moins 3 caractères.")
                elif su_user in db:
                    st.error("❌ Ce pseudo est déjà pris.")
                elif len(su_pass) < 4:
                    st.error("Le mot de passe doit faire au moins 4 caractères.")
                elif su_pass != su_pass2:
                    st.error("❌ Les mots de passe ne correspondent pas.")
                else:
                    db[su_user] = {"pw": hash_pw(su_pass)}
                    save_users_db_sync(db)
                    st.session_state.logged_in    = True
                    st.session_state.username     = su_user
                    st.session_state.page         = "rate"
                    st.session_state.user_ratings = {}
                    st.success("✅ Compte créé ! Bienvenue 🎉")
                    st.rerun()


# ══════════════════════════════════════════════════════════
# PAGE : NOTER DES FILMS + RECOMMANDATIONS
# ══════════════════════════════════════════════════════════

def star_display(rating):
    r = int(rating)
    return "★" * r + "☆" * (5 - r)

def page_rate():
    if not st.session_state.data_loaded:
        st.warning("👈 Chargez vos fichiers `.dat` dans la barre latérale pour commencer.")
        return

    movies   = st.session_state.movies_proc
    genres   = st.session_state.genre_cols
    username = st.session_state.username
    ur       = st.session_state.user_ratings

    # Welcome banner
    n_rated = len(ur)
    st.markdown(f"""
    <div class="welcome-banner">
        <div style="font-size:2.5rem;">👋</div>
        <div>
            <div style="font-weight:600;font-size:1.1rem;">Bonjour, <span style="color:#f5c518">{username}</span> !</div>
            <div style="color:#888;font-size:0.9rem;">
                {n_rated} film{"s" if n_rated>1 else ""} noté{"s" if n_rated>1 else ""}
                {"• Recommandations disponibles 🎯" if n_rated >= 3 else f"• Notez encore {3-n_rated} film(s) pour débloquer les recommandations"}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    # ── COLONNE GAUCHE : noter des films ──
    with left_col:
        st.markdown('<div class="section-header">⭐ NOTER UN FILM</div>', unsafe_allow_html=True)

        # Recherche
        search = st.text_input("🔍 Rechercher un film", placeholder="ex: Matrix, Titanic…",
                                key="search_movie", label_visibility="collapsed")

        if search and len(search) >= 2:
            results = movies[movies["Title"].str.contains(search, case=False, na=False)]
        else:
            # Affiche des films populaires non encore notés
            not_rated = movies[~movies["MovieID"].isin([int(k) for k in ur.keys()])]
            results   = not_rated.head(20)

        st.markdown(f"<p style='color:#888;font-size:0.8rem;margin-bottom:0.8rem;'>"
                    f"{len(results)} film(s) affiché(s)</p>", unsafe_allow_html=True)

        # Liste avec notation inline
        for _, row in results.iterrows():
            mid  = str(row["MovieID"])
            already = ur.get(mid)
            genres_html = "".join(
                f'<span class="movie-genre-badge">{g}</span>'
                for g in row["Genres"].split("|")[:4]
            )
            rated_html = (
                f'<span class="score-high">{star_display(already)}</span>'
                if already else '<span style="color:#555">Non noté</span>'
            )

            with st.container():
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title-text">{row['Title']}</div>
                    <div style="margin-top:4px">{genres_html}</div>
                    <div style="margin-top:6px;font-size:0.82rem;">{rated_html}</div>
                </div>
                """, unsafe_allow_html=True)

                r_col1, r_col2 = st.columns([2, 1])
                with r_col1:
                    note = st.select_slider(
                        f"Note",
                        options=[1, 2, 3, 4, 5],
                        value=int(already) if already else 3,
                        key=f"slider_{mid}",
                        label_visibility="collapsed",
                        format_func=lambda x: f"{'★'*x}{'☆'*(5-x)}  ({x}/5)"
                    )
                with r_col2:
                    if st.button("✓ Valider", key=f"rate_{mid}"):
                        ur[mid] = note
                        st.session_state.user_ratings = ur
                        save_user_ratings_sync(username, ur)
                        st.rerun()

    # ── COLONNE DROITE : recommandations ──
    with right_col:
        st.markdown('<div class="section-header">🎯 MES RECOMMANDATIONS</div>',
                    unsafe_allow_html=True)

        if n_rated < 3:
            st.markdown(f"""
            <div style="background:#18181b;border:1px dashed #3a3a3e;border-radius:12px;
                        padding:2rem;text-align:center;color:#666;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">🔒</div>
                <div>Notez <strong style="color:#f5c518">{3 - n_rated} film(s) de plus</strong>
                <br>pour débloquer vos recommandations</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            recs = recommend(ur, movies, genres, top_n=12)

            if recs.empty:
                st.info("Vous avez déjà noté tous les films ! 🎉")
            else:
                for _, row in recs.iterrows():
                    score = row["score"]
                    cls   = "score-high" if score >= 60 else ("score-medium" if score >= 35 else "score-low")
                    genres_html = "".join(
                        f'<span class="movie-genre-badge">{g}</span>'
                        for g in row["Genres"].split("|")[:3]
                    )
                    st.markdown(f"""
                    <div class="movie-card">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                            <div class="movie-title-text" style="flex:1;padding-right:0.5rem">
                                {row['Title']}
                            </div>
                            <div class="{cls}" style="font-size:1rem;white-space:nowrap;">
                                {score:.0f}%
                            </div>
                        </div>
                        <div style="margin-top:4px">{genres_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Bouton noter depuis reco
                    if st.button("⭐ Je veux noter ce film", key=f"note_from_rec_{row['MovieID']}"):
                        st.session_state["quick_rate_id"] = str(row["MovieID"])
                        st.session_state["quick_rate_title"] = row["Title"]

            # Quick rate modal
            if st.session_state.get("quick_rate_id"):
                qid    = st.session_state["quick_rate_id"]
                qtitle = st.session_state.get("quick_rate_title", "")
                st.markdown(f"**Noter** : _{qtitle}_")
                qnote = st.select_slider(
                    "Votre note",
                    options=[1,2,3,4,5],
                    value=3,
                    format_func=lambda x: f"{'★'*x}{'☆'*(5-x)} ({x}/5)",
                    key="quick_slider"
                )
                qc1, qc2 = st.columns(2)
                if qc1.button("✓ Confirmer"):
                    ur[qid] = qnote
                    st.session_state.user_ratings = ur
                    save_user_ratings_sync(username, ur)
                    del st.session_state["quick_rate_id"]
                    st.rerun()
                if qc2.button("✗ Annuler"):
                    del st.session_state["quick_rate_id"]
                    st.rerun()


# ══════════════════════════════════════════════════════════
# PAGE : MES NOTES
# ══════════════════════════════════════════════════════════

def page_my_ratings():
    st.markdown('<div class="section-header">📋 MES FILMS NOTÉS</div>', unsafe_allow_html=True)
    ur      = st.session_state.user_ratings
    movies  = st.session_state.movies_proc
    username = st.session_state.username

    if not ur:
        st.info("Vous n'avez encore noté aucun film. Allez dans **Accueil** pour commencer !")
        return

    rated_ids = [int(k) for k in ur.keys()]
    rated_movies = movies[movies["MovieID"].isin(rated_ids)].copy()
    rated_movies["Ma note"] = rated_movies["MovieID"].apply(lambda m: ur.get(str(m), 0))
    rated_movies["Étoiles"] = rated_movies["Ma note"].apply(star_display)

    # Résumé
    avg = rated_movies["Ma note"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Films notés",      len(ur))
    c2.metric("Note moyenne",     f"{avg:.1f} / 5")
    c3.metric("Films adorés (5★)", len(rated_movies[rated_movies["Ma note"]==5]))

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Tableau
    display = rated_movies[["Title","Genres","Étoiles","Ma note"]].sort_values(
        "Ma note", ascending=False
    ).reset_index(drop=True)
    display.index += 1
    st.dataframe(
        display,
        use_container_width=True,
        column_config={
            "Title":    st.column_config.TextColumn("Film"),
            "Genres":   st.column_config.TextColumn("Genres"),
            "Étoiles":  st.column_config.TextColumn("Note"),
            "Ma note":  st.column_config.NumberColumn("★", format="%d"),
        }
    )

    if st.button("🗑️ Supprimer toutes mes notes", type="secondary"):
        st.session_state.user_ratings = {}
        save_user_ratings_sync(username, {})
        st.rerun()


# ══════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════

if not st.session_state.logged_in:
    page_auth()
else:
    p = st.session_state.page
    if p == "rate":
        page_rate()
    elif p == "my_ratings":
        page_my_ratings()
    else:
        page_rate()
