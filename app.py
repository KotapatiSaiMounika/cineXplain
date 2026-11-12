"""CineXplain — interactive Streamlit movie recommender."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import html

import pandas as pd
import streamlit as st

from src.streamlit_engine import CineXplainEngine


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="CineXplain · Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
      :root { --ink:#f3f4f8; --muted:#a8afc2; --violet:#9b8cff; --cyan:#70d9e8; }
      .stApp { background: radial-gradient(ellipse at 8% 0%, rgba(115,91,219,.18), transparent 34%), #0b1020; color:var(--ink); }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
      h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; }
      [data-testid="stSidebar"] { background:#10172a; border-right:1px solid rgba(255,255,255,.07); }
      [data-testid="stMetric"] { background:rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.08); padding:14px 16px; border-radius:16px; }
      [data-testid="stMetricLabel"] { color:#a8afc2; }
      .hero { padding:28px 30px; border:1px solid rgba(255,255,255,.10); border-radius:24px; background:linear-gradient(120deg,rgba(91,75,168,.35),rgba(22,39,65,.74)); margin-bottom:20px; }
      .eyebrow { color:#b7adff; font-size:11px; font-weight:700; letter-spacing:.19em; text-transform:uppercase; }
      .hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(34px,5vw,54px); line-height:1.02; font-weight:700; margin:8px 0 10px; color:#fff; }
      .hero-copy { color:#c3c9d9; font-size:15px; max-width:760px; line-height:1.6; }
      .section-kicker { color:#a99cff; font-size:11px; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
      .movie-card { border:1px solid rgba(255,255,255,.10); border-radius:18px; background:rgba(255,255,255,.045); padding:14px; min-height:245px; }
      .movie-art { height:128px; display:flex; align-items:center; justify-content:center; border-radius:13px; margin-bottom:12px; background:linear-gradient(135deg,#33316b,#1b5964); color:#fff; font-size:35px; font-family:'Space Grotesk',sans-serif; font-weight:700; letter-spacing:-.04em; }
      .movie-title { color:#fff; font-family:'Space Grotesk',sans-serif; font-size:18px; line-height:1.2; font-weight:600; margin:3px 0 5px; }
      .movie-meta { color:#aeb6c9; font-size:12px; line-height:1.5; min-height:32px; }
      .score-line { display:flex; gap:7px; flex-wrap:wrap; margin:10px 0 7px; }
      .score-pill { color:#dcd8ff; background:rgba(155,140,255,.14); border:1px solid rgba(155,140,255,.2); border-radius:30px; padding:4px 8px; font-size:10px; font-weight:700; }
      .why { color:#bdc5d8; font-size:12px; line-height:1.5; margin-top:8px; }
      .small-note { color:#909bb2; font-size:12px; }
      .stButton>button[kind="primary"] { background:linear-gradient(90deg,#7867e8,#4c9eae); border:0; border-radius:12px; font-weight:700; }
      .stButton>button { border-radius:12px; }
      div[data-testid="stForm"] { background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.08); padding:18px; border-radius:18px; }
      hr { border-color:rgba(255,255,255,.08); }
      .footer { color:#7f8aa3; font-size:11px; padding:22px 0 5px; text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing your recommendation models…")
def build_engine(movie_bytes: bytes, rating_bytes: bytes | None, enriched_bytes: bytes | None):
    movies = pd.read_csv(BytesIO(movie_bytes))
    ratings = pd.read_csv(BytesIO(rating_bytes)) if rating_bytes else pd.DataFrame()
    enriched = pd.read_csv(BytesIO(enriched_bytes)) if enriched_bytes else None
    return CineXplainEngine(
        movies=movies,
        ratings=ratings,
        enriched=enriched,
        cache_dir=DATA_DIR / "cache",
    )


def read_local_bytes(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def poster_source(row: pd.Series) -> str | None:
    direct = str(row.get("poster_url", "") or "").strip()
    path = str(row.get("poster_path", "") or "").strip()
    if direct.startswith(("https://", "http://")):
        return direct
    if path.startswith("/"):
        return "https://image.tmdb.org/t/p/w500" + path
    return None


def render_movie_card(row: pd.Series, index: int) -> None:
    title = str(row.get("title", "Untitled"))
    genres = str(row.get("genres", "")).replace("|", " · ") or "Genre not listed"
    year = int(row.get("year", 0) or 0)
    content_score = float(row.get("content_score", 0) or 0)
    cf_score = float(row.get("cf_score", 0) or 0)
    hybrid_score = float(row.get("hybrid_score", 0) or 0)
    mmr_score = float(row.get("mmr_score", hybrid_score) or 0)
    reason = str(row.get("why_recommended", "A good match for your profile."))
    poster = poster_source(row)

    with st.container(border=True):
        if poster:
            st.image(poster, use_container_width=True)
        else:
            short = html.escape(" ".join(title.split()[:2]).upper())
            gradients = [
                "linear-gradient(135deg,#403777,#24717a)",
                "linear-gradient(135deg,#7d3e62,#3c4b86)",
                "linear-gradient(135deg,#315b75,#65509a)",
                "linear-gradient(135deg,#6a4b36,#315c68)",
                "linear-gradient(135deg,#5b426f,#315b56)",
            ]
            background = gradients[index % len(gradients)]
            st.markdown(
                f'<div class="movie-art" style="background:{background}">{short}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<div class="movie-title">{html.escape(title)}</div>', unsafe_allow_html=True)
        year_label = str(year) if year > 0 else "Year unknown"
        st.markdown(
            f'<div class="movie-meta">{year_label} · {html.escape(genres)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='score-line'>"
            f"<span class='score-pill'>MATCH {hybrid_score:.0%}</span>"
            f"<span class='score-pill'>CONTENT {content_score:.0%}</span>"
            f"<span class='score-pill'>DIVERSITY {mmr_score:.2f}</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="why">{html.escape(reason)}</div>', unsafe_allow_html=True)
        overview = str(row.get("overview", "") or "").strip()
        if overview:
            with st.expander("Movie details"):
                st.write(overview)
                st.caption(f"Collaborative signal: {cf_score:.3f}")


st.sidebar.markdown("## 🎬 CineXplain")
st.sidebar.caption("Personalized discovery · Hybrid ranking · MMR diversity")
st.sidebar.markdown("---")
st.sidebar.markdown("### Your data")
uploaded_movies = st.sidebar.file_uploader(
    "Upload movies.csv", type=["csv"], help="MovieLens format: movieId, title, genres. Enriched metadata is optional."
)
uploaded_ratings = st.sidebar.file_uploader(
    "Upload ratings.csv (optional)", type=["csv"], help="MovieLens format: userId, movieId, rating. Adds collaborative filtering."
)
uploaded_enriched = st.sidebar.file_uploader(
    "Upload enriched metadata (optional)", type=["csv"], help="Optional overview, cast, director, keywords and poster metadata."
)

if uploaded_movies is not None:
    movie_bytes = uploaded_movies.getvalue()
    rating_bytes = uploaded_ratings.getvalue() if uploaded_ratings is not None else None
    enriched_bytes = uploaded_enriched.getvalue() if uploaded_enriched is not None else None
    dataset_label = "Your uploaded dataset"
    dataset_is_demo = False
else:
    local_movies = read_local_bytes(DATA_DIR / "movies.csv")
    local_ratings = read_local_bytes(DATA_DIR / "ratings.csv")
    local_enriched = read_local_bytes(DATA_DIR / "movies_enriched.csv")
    if local_movies is not None:
        movie_bytes = local_movies
        rating_bytes = local_ratings
        enriched_bytes = local_enriched
        dataset_label = "Local MovieLens dataset"
        dataset_is_demo = False
    else:
        movie_bytes = (DATA_DIR / "demo_movies.csv").read_bytes()
        rating_bytes = (DATA_DIR / "demo_ratings.csv").read_bytes()
        enriched_bytes = None
        dataset_label = "Built-in demo dataset"
        dataset_is_demo = True

try:
    engine = build_engine(movie_bytes, rating_bytes, enriched_bytes)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.info("Check the CSV column names, then upload MovieLens-compatible movies.csv and ratings.csv files.")
    st.stop()

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">A smarter way to pick your next watch</div>
      <div class="hero-title">Find your next<br/>favourite film.</div>
      <div class="hero-copy">Tell CineXplain what you enjoy. It blends movie content, patterns from other viewers, and a diversity-aware final ranking — so your list feels personal without becoming repetitive.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if dataset_is_demo:
    st.info(
        "You’re viewing the included demo catalogue. Its 912 ratings are synthetic and exist only to demonstrate the collaborative and hybrid pipeline. Upload your real MovieLens files for meaningful recommendations."
    )
elif uploaded_movies is not None and uploaded_ratings is None:
    st.info("Movie metadata is loaded. Upload ratings.csv as well to enable collaborative filtering; content-based recommendations will still work without it.")

summary_cols = st.columns(4)
summary_cols[0].metric("Movies", f"{len(engine.movies):,}")
summary_cols[1].metric("Rating events", f"{len(engine.ratings):,}")
summary_cols[2].metric("Raters", f"{engine.ratings['userId'].nunique():,}" if not engine.ratings.empty else "—")
summary_cols[3].metric("Recommendation layers", "4")
st.caption(f"Data source: **{dataset_label}**  ·  Similarity model: TF-IDF  ·  Re-ranking: MMR")

st.markdown("### Build your taste profile")
st.markdown('<div class="small-note">Choose a few films you liked. Add a film you did not enjoy to help the system steer away from that kind of recommendation.</div>', unsafe_allow_html=True)

movie_options = {
    f"{row.title}  ·  #{int(row.movieId)}": int(row.movieId)
    for row in engine.movies.itertuples(index=False)
}
all_labels = list(movie_options.keys())
default_favourites = [
    label for label in all_labels
    if label.startswith(("Inception (2010)", "Interstellar (2014)"))
][:2]

with st.form("taste_profile_form"):
    favourites_col, avoid_col = st.columns([1.2, 1])
    with favourites_col:
        favourite_labels = st.multiselect(
            "Movies you enjoyed",
            options=all_labels,
            default=default_favourites,
            max_selections=8,
            placeholder="Search and select 1–8 favourites…",
        )
    with avoid_col:
        disliked_labels = st.multiselect(
            "Movies you did not enjoy (optional)",
            options=[label for label in all_labels if label not in favourite_labels],
            max_selections=5,
            placeholder="Choose up to 5 films to avoid…",
        )
    generate_col, count_col = st.columns([1, 1])
    with generate_col:
        top_k = st.slider("How many recommendations?", min_value=5, max_value=20, value=10)
    with count_col:
        diversity = st.slider(
            "Relevance ↔ diversity",
            min_value=0.20,
            max_value=1.00,
            value=0.50,
            step=0.05,
            help="1.0 prioritises relevance; lower values encourage more variety.",
        )

    filter_col1, filter_col2 = st.columns(2)
    available_genres = sorted({
        genre.strip()
        for value in engine.movies["genres"].fillna("")
        for genre in str(value).split("|")
        if genre.strip() and genre.strip().lower() != "(no genres listed)"
    })
    with filter_col1:
        excluded_genres = st.multiselect("Exclude genres", options=available_genres, placeholder="No genre exclusions")
    with filter_col2:
        nonzero_years = engine.movies.loc[engine.movies["year"] > 0, "year"]
        if not nonzero_years.empty:
            min_available, max_available = int(nonzero_years.min()), int(nonzero_years.max())
        else:
            min_available, max_available = 1900, 2030
        if min_available < max_available:
            year_range = st.slider(
                "Release year range",
                min_value=min_available,
                max_value=max_available,
                value=(min_available, max_available),
            )
        else:
            year_range = (min_available, max_available)
            st.caption(f"All catalogue titles with a known year are from {min_available}.")

    has_franchises = engine.movies["collection"].fillna("").astype(str).str.strip().ne("").any()
    max_per_franchise = None
    if has_franchises:
        max_per_franchise = st.slider(
            "Maximum films per franchise",
            min_value=1,
            max_value=5,
            value=2,
            help="A soft-looking control with a hard effect: no more than this many films from one collection enter the final candidate list.",
        )
    submitted = st.form_submit_button("✨ Generate my recommendations", type="primary", use_container_width=True)

if submitted:
    st.session_state.pop("cinexplain_result", None)
    profile = {movie_options[label]: 5.0 for label in favourite_labels}
    profile.update({movie_options[label]: 1.0 for label in disliked_labels})
    try:
        if not favourite_labels:
            raise ValueError("Select at least one favourite movie first.")
        with st.spinner("Finding similar films, blending signals and removing repetitive picks…"):
            result = engine.recommend(
                profile=profile,
                top_k=top_k,
                diversity=diversity,
                excluded_genres=excluded_genres,
                min_year=year_range[0],
                max_year=year_range[1],
                max_per_franchise=max_per_franchise,
                candidate_pool=max(80, top_k * 8),
            )
        st.session_state["cinexplain_result"] = result
        st.session_state["cinexplain_profile"] = profile
        st.session_state["cinexplain_profile_labels"] = favourite_labels
    except Exception as exc:
        st.error(str(exc))

result = st.session_state.get("cinexplain_result")
if result is not None:
    st.markdown("---")
    st.markdown("### Your recommendations")
    favourite_names = st.session_state.get("cinexplain_profile_labels", [])
    st.caption("Taste profile: " + " · ".join(favourite_names))
    weight_cols = st.columns(4)
    weight_cols[0].metric("Content weight", f"{result['content_weight']:.0%}")
    weight_cols[1].metric("Collaborative weight", f"{result['collaborative_weight']:.0%}")
    weight_cols[2].metric("Candidates", f"{result['candidate_count']:,}")
    weight_cols[3].metric("Collaborative source", "Item-KNN" if result["collaborative_source"].startswith("item-item") else "Popularity fallback")

    recommendations_tab, compare_tab, explain_tab = st.tabs(
        ["✦ Diverse picks", "Ranking comparison", "How CineXplain works"]
    )
    with recommendations_tab:
        st.markdown('<div class="section-kicker">MMR-optimised list</div>', unsafe_allow_html=True)
        st.caption("The final list balances your hybrid relevance score against similarity to movies already selected.")
        rows = result["diverse"]
        if rows.empty:
            st.warning("No recommendations were returned. Widen your filters and try again.")
        else:
            for start in range(0, len(rows), 3):
                card_cols = st.columns(3)
                for offset, (_, row) in enumerate(rows.iloc[start : start + 3].iterrows()):
                    with card_cols[offset]:
                        render_movie_card(row, start + offset)
            csv_bytes = rows.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download recommendation list (CSV)",
                data=csv_bytes,
                file_name="cinexplain_recommendations.csv",
                mime="text/csv",
            )

    with compare_tab:
        st.markdown('<div class="section-kicker">Before and after diversity re-ranking</div>', unsafe_allow_html=True)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Hybrid ILD", f"{result['hybrid_ild']:.3f}")
        metric_cols[1].metric("MMR ILD", f"{result['diverse_ild']:.3f}", delta=f"{result['diverse_ild'] - result['hybrid_ild']:+.3f}")
        metric_cols[2].metric("Lambda", f"{result['lambda']:.2f}")
        st.caption("ILD (Intra-List Diversity) is 1 minus the average pairwise content similarity. Higher values mean the list is less repetitive; relevance should be considered alongside it.")
        left, right = st.columns(2)
        with left:
            st.markdown("**Hybrid ranking**")
            st.dataframe(
                result["hybrid"][["title", "genres", "hybrid_score", "content_score", "cf_score"]],
                hide_index=True,
                use_container_width=True,
            )
        with right:
            st.markdown("**After MMR re-ranking**")
            st.dataframe(
                result["diverse"][["title", "genres", "hybrid_score", "mmr_score"]],
                hide_index=True,
                use_container_width=True,
            )

    with explain_tab:
        st.markdown("**1 · Content-based filtering**")
        st.write("TF-IDF represents each film using its title, genres and available plot/cast metadata. Cosine similarity finds films whose descriptions resemble your favourites.")
        st.markdown("**2 · Collaborative filtering**")
        st.write("A lightweight item-item model compares user-centred rating patterns. Movies liked by people who rated your favourites similarly receive a stronger collaborative signal.")
        st.markdown("**3 · Adaptive hybrid ranking**")
        st.latex(r"\alpha = \frac{68}{68 + n}\qquad S_{hybrid}=\alpha S_{content}+(1-\alpha)S_{collaborative}")
        st.write(f"You supplied {len(st.session_state.get('cinexplain_profile', {}))} ratings, so this run assigned {result['content_weight']:.1%} to content and {result['collaborative_weight']:.1%} to collaborative/popularity signals.")
        st.markdown("**4 · Maximal Marginal Relevance (MMR)**")
        st.latex(r"MMR(d)=\lambda\,Rel(d)-(1-\lambda)\max_{s\in S}Sim(d,s)")
        st.write("MMR greedily selects the next best film while penalising candidates that are too similar to films already selected. The slider controls this relevance–diversity trade-off.")
        st.warning("This is a recommendation demo, not a production ranking service. The bundled ratings are synthetic; report model quality using held-out real ratings, not the sample data.")

else:
    st.markdown("---")
    st.markdown("### What happens under the hood")
    overview_cols = st.columns(4)
    overview_items = [
        ("01", "Understand your taste", "Build a profile from films you liked and disliked."),
        ("02", "Find similar films", "Use TF-IDF and cosine similarity over movie metadata."),
        ("03", "Blend viewer signals", "Combine content and item-item collaborative scores."),
        ("04", "Keep it varied", "Apply MMR so the final list is not ten versions of the same movie."),
    ]
    for col, (number, title, body) in zip(overview_cols, overview_items):
        with col:
            st.markdown(f"**{number} · {title}**")
            st.caption(body)
    st.info("Your first run is ready: the demo starts with Inception and Interstellar selected. Change the profile or filters above, then press **Generate my recommendations**.")

with st.expander("Dataset and model diagnostics"):
    st.write({
        "source": dataset_label,
        "movies loaded": len(engine.movies),
        "rating rows loaded": len(engine.ratings),
        "unique raters": int(engine.ratings["userId"].nunique()) if not engine.ratings.empty else 0,
        "collaborative model available": engine.has_collaborative_data,
        "metadata fields": [column for column in ["overview", "cast", "director", "keywords", "poster_url"] if column in engine.movies.columns and engine.movies[column].astype(str).str.strip().ne("").any()],
    })

st.markdown('<div class="footer">CineXplain · Discover more than the obvious next watch · Built with Python + Streamlit</div>', unsafe_allow_html=True)
