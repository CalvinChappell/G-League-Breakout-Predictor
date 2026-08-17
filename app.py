"""
G-League Breakout Predictor — Streamlit dashboard.

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import model

st.set_page_config(
    page_title="G-League Breakout Predictor",
    page_icon="🏀",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Sidebar — adjustable model weights
# ----------------------------------------------------------------------------
st.sidebar.title("🏀 Model Weights")
st.sidebar.caption(
    "Adjust how much each stat category counts toward the Breakout Score. "
    "Weights are auto-normalized to sum to 100%."
)

if "weights" not in st.session_state:
    st.session_state.weights = dict(model.DEFAULT_WEIGHTS)

if st.sidebar.button("Reset to defaults"):
    st.session_state.weights = dict(model.DEFAULT_WEIGHTS)

weights = {}
for feat in model.FEATURES:
    weights[feat] = st.sidebar.slider(
        model.FEATURE_LABELS[feat],
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.weights.get(feat, model.DEFAULT_WEIGHTS[feat])),
        step=0.05,
        key=f"slider_{feat}",
    )
st.session_state.weights = weights

total_w = sum(weights.values()) or 1.0
st.sidebar.caption("Normalized weights: " + ", ".join(
    f"{model.FEATURE_LABELS[k].split(' ')[0]} {v/total_w:.0%}" for k, v in weights.items()
))

min_age = st.sidebar.slider("Min age", 18, 32, 18)
max_age = st.sidebar.slider("Max age", 18, 32, 30)

st.sidebar.divider()
st.sidebar.caption(
    "Data: Basketball-Reference.com (historical G-League/NBA advanced stats) "
    "and StatsCrew.com (2025-26 current G-League prospects). See the "
    "Methodology tab for sourcing detail and known limitations."
)

# ----------------------------------------------------------------------------
# Score everything with current weights
# ----------------------------------------------------------------------------
hist_scored, cur_scored, pool_scored = model.score_all(weights)
cur_scored = cur_scored[(cur_scored["age"] >= min_age) & (cur_scored["age"] <= max_age)]

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("G-League Breakout Predictor")
st.caption(
    "A heuristic model that scores current G-League players on how closely their "
    "statistical profile matches players who went on to become real NBA breakouts "
    "(Pascal Siakam, Rudy Gobert, Khris Middleton, and 11 others)."
)

tab_leaderboard, tab_historical, tab_compare, tab_methodology = st.tabs(
    ["📈 Prospect Leaderboard", "🏆 Historical Breakouts", "🔍 Player Explorer", "📋 Methodology"]
)

# ----------------------------------------------------------------------------
# TAB 1 — Leaderboard
# ----------------------------------------------------------------------------
with tab_leaderboard:
    st.subheader("Current G-League Prospects, Ranked by Breakout Score")

    sorted_cur = cur_scored.sort_values("breakout_score", ascending=False)

    fig = px.bar(
        sorted_cur,
        x="breakout_score",
        y="player",
        orientation="h",
        color="breakout_score",
        color_continuous_scale="Oranges",
        labels={"breakout_score": "Breakout Score (0-100)", "player": ""},
        hover_data=["gleague_team", "nba_affiliate", "age", "pts", "reb", "ast", "ts_pct"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=650, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    display_cols = [
        "player", "position", "gleague_team", "nba_affiliate", "age", "gp", "mpg",
        "pts", "reb", "ast", "ts_pct", "breakout_score",
    ]
    st.dataframe(
        sorted_cur[display_cols].rename(columns={
            "player": "Player", "position": "Pos", "gleague_team": "G-League Team",
            "nba_affiliate": "NBA Affiliate", "age": "Age", "gp": "GP", "mpg": "MPG",
            "pts": "PTS", "reb": "REB", "ast": "AST", "ts_pct": "TS%",
            "breakout_score": "Breakout Score",
        }).style.format({"TS%": "{:.1%}", "MPG": "{:.1f}", "PTS": "{:.1f}", "REB": "{:.1f}", "AST": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Best Historical Comp")
    pick = st.selectbox("Pick a prospect to see their closest historical comp(s):", sorted_cur["player"].tolist())
    prospect_row = cur_scored[cur_scored["player"] == pick].iloc[0]
    comps = model.nearest_comps(prospect_row, hist_scored, k=3)
    st.dataframe(
        comps.rename(columns={
            "player": "Historical Comp", "gleague_team": "G-League Team", "gleague_season": "Season",
            "breakout_note": "What Happened Next", "distance": "Distance (lower = closer)",
            "similarity_pct": "Similarity %",
        }),
        use_container_width=True,
        hide_index=True,
    )

# ----------------------------------------------------------------------------
# TAB 2 — Historical breakouts
# ----------------------------------------------------------------------------
with tab_historical:
    st.subheader("The 14 Historical G-League Breakout Cases")
    st.caption(
        "Each row shows the player's G-League statistical snapshot (the stint used as "
        "their 'prospect profile') and what happened in their NBA breakout season."
    )

    hist_display = hist_scored.sort_values("breakout_score", ascending=False)[
        ["player", "position", "gleague_team", "gleague_season", "age_g", "gp_g", "mpg_g",
         "pts_g", "reb_g", "ast_g", "ts_g", "per_g", "usg_g", "ws48_g", "breakout_score",
         "breakout_season", "age_bo", "breakout_note"]
    ].rename(columns={
        "player": "Player", "position": "Pos", "gleague_team": "G-League Team",
        "gleague_season": "Season", "age_g": "Age", "gp_g": "GP", "mpg_g": "MPG",
        "pts_g": "PTS", "reb_g": "REB", "ast_g": "AST", "ts_g": "TS%", "per_g": "PER",
        "usg_g": "USG%", "ws48_g": "WS/48", "breakout_score": "Breakout Score",
        "breakout_season": "NBA Breakout Season", "age_bo": "Age at Breakout",
        "breakout_note": "What Happened",
    })
    st.dataframe(hist_display, use_container_width=True, hide_index=True, height=560)

    st.info(
        "**Reading this table:** all 14 players eventually broke out, but their G-League "
        "*Breakout Scores* vary a lot — Khris Middleton scores near the bottom because his "
        "G-League sample was tiny and rough (3 games) despite becoming a multi-time All-Star. "
        "That's a real limitation of stat-based screening: some breakouts come from players "
        "whose G-League stint doesn't look special at all. See Methodology for more on this."
    )

# ----------------------------------------------------------------------------
# TAB 3 — Player explorer / scatter comparison
# ----------------------------------------------------------------------------
with tab_compare:
    st.subheader("Efficiency vs. Scoring Load — All Players")
    combined = pd.concat([hist_scored, cur_scored], ignore_index=True, sort=False)
    fig2 = px.scatter(
        combined,
        x="pts_per36",
        y="ts_pct",
        color="group",
        size="breakout_score",
        hover_name="player",
        hover_data=["gleague_team", "age", "breakout_score"],
        labels={"pts_per36": "Points per 36 min", "ts_pct": "True Shooting %"},
        color_discrete_map={"Historical Breakout": "#1f77b4", "Current Prospect": "#ff7f0e"},
    )
    fig2.update_layout(height=550)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Radar Comparison")
    col1, col2 = st.columns(2)
    with col1:
        player_a = st.selectbox("Player A", combined["player"].tolist(), index=0)
    with col2:
        default_b_idx = min(1, len(combined) - 1)
        player_b = st.selectbox("Player B", combined["player"].tolist(), index=default_b_idx)

    def get_radar_values(name):
        row = combined[combined["player"] == name].iloc[0]
        return [row[f"z_{f}"] for f in model.FEATURES]

    categories = [model.FEATURE_LABELS[f] for f in model.FEATURES]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(r=get_radar_values(player_a), theta=categories, fill="toself", name=player_a))
    fig3.add_trace(go.Scatterpolar(r=get_radar_values(player_b), theta=categories, fill="toself", name=player_b))
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=500,
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Values are z-scores relative to the full player pool (0 = league-average shape, positive = above average, sign-adjusted so higher is always better).")

# ----------------------------------------------------------------------------
# TAB 4 — Methodology
# ----------------------------------------------------------------------------
with tab_methodology:
    st.subheader("How the Breakout Score works")
    st.markdown(
        """
This is a **heuristic scoring rubric**, not a trained machine-learning classifier.

**Why not a real trained model?** The historical dataset only contains 14 players,
and every one of them is a "success" case (they all eventually broke out) — there's
no labeled negative class of "G-League standouts who never broke out" to fit a
classifier against. Building one would require tracking hundreds of G-League
players over many years and labeling outcomes, which is a much bigger data project.
Instead, this tool standardizes each prospect's G-League stat profile against the
pooled distribution of historical-breakout snapshots and current prospects, and
combines six features into a single 0-100 percentile score using **weights you can
adjust** in the sidebar.

**Features used** (chosen because they're available for both the historical and
current datasets — no advanced stats like PER/USG%/BPM for current prospects,
since the G League's official advanced-stats pages aren't easily scrapable):

- **Efficiency** — True Shooting %
- **Scoring** — Points per 36 minutes
- **Playmaking** — Assists per 36 minutes
- **Rebounding** — Rebounds per 36 minutes
- **Defense** — Steals + blocks per 36 minutes (median-imputed where missing — see caveat below)
- **Youth** — Age (inverted; younger scores higher)

Each feature is z-scored against the pooled mean/std of all 32 players (14
historical + 18 current), sign-adjusted so a higher z-score always means "more
breakout-like." The weighted sum is then converted to a 0-100 percentile rank.

**Best Historical Comp** uses Euclidean distance between a prospect's and each
historical player's standardized feature vectors — effectively "who does this
player's G-League stat-shape most resemble."

### Known limitations (read before trusting this too much)

1. **Small, hand-curated historical sample.** 14 players, no negative examples,
   selected because they're well-known success stories — this is a
   similarity/screening tool, not a calibrated probability.
2. **Middleton is a documented miss.** His actual G-League sample (3 games) was
   too small and rough to look like a real prospect profile by these stats, yet
   he became a multi-time All-Star. Small-sample G-League stints don't capture
   everything (injuries, roster context, later development).
3. **Missing advanced stats for current prospects.** PER/USG%/WS-48/BPM weren't
   reliably scrapable from stats.gleague.nba.com for the 2025-26 season, so the
   model relies on basic box-score rates instead. If you can source official
   advanced stats later, add USG% and a real BPM/PER-style metric to `model.py`'s
   `FEATURES` for a sharper model.
4. **Steals/blocks are mostly missing for current prospects** (only reported for
   players who led the league in those categories) and are median-imputed —
   treat the "Defense" component for most current players as neutral, not
   measured.
5. **Draft-pick and roster details for some current prospects were flagged as
   "unverified" during initial research** (see source notes in `data/current_prospects.csv`)
   — worth double-checking before using this to inform real decisions.
6. **G-League ≠ NBA translation isn't modeled.** Competition level, age
   adjustment beyond simple linear "younger is better," and role continuity
   (will this team actually play the prospect at the NBA level?) aren't
   captured.

### Data sources
- Historical players: [Basketball-Reference.com](https://www.basketball-reference.com)
  G-League and NBA player pages (URLs in `data/historical_breakouts.csv`).
- Current prospects: [StatsCrew.com](https://www.statscrew.com) 2025-26 G-League
  player pages, cross-checked against the [NBA G League two-way tracker](https://gleague.nba.com/twowayplayers)
  and [call-up log](https://gleague.nba.com/nba-call-ups-2025-26) (URLs in
  `data/current_prospects.csv`).
        """
    )
