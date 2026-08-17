# G-League Breakout Predictor

An interactive Streamlit dashboard that scores current NBA G-League players on
how closely their statistical profile matches players who went on to become
real NBA breakouts — Pascal Siakam, Rudy Gobert, Khris Middleton, and 11 others
who spent meaningful time in the G-League (formerly D-League) before making it.

This is a **transparent heuristic model**, not a trained classifier — see the
Methodology tab in the app for exactly how it works and its limitations.

## What's inside

- `data/historical_breakouts.csv` — 14 players who spent real time in the
  G-League/D-League and later broke out in the NBA, with their G-League
  statistical snapshot and what happened in their NBA breakout season.
- `data/current_prospects.csv` — 18 notable current (2025-26 season) G-League
  players/two-way prospects with their season stats.
- `model.py` — the scoring logic: per-36 rate stats, z-score standardization,
  a weighted composite "Breakout Score" (0-100), and nearest-neighbor
  "best historical comp" matching.
- `app.py` — the Streamlit dashboard: a sortable prospect leaderboard,
  adjustable model weights, a historical breakouts table, a scatter/radar
  player explorer, and a methodology write-up.

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the dashboard at `http://localhost:8501`.

You can also run the model standalone (no Streamlit needed) to see the
rankings printed to your terminal:

```bash
python model.py
```

## Pushing to GitHub

This folder is already a git repository with an initial commit. To push it:

```bash
git remote add origin <your-new-repo-url>
git branch -M main
git push -u origin main
```

## Data sources

- **Historical players:** [Basketball-Reference.com](https://www.basketball-reference.com)
  G-League and NBA player pages.
- **Current prospects:** [StatsCrew.com](https://www.statscrew.com) 2025-26
  G-League player pages, cross-checked against the NBA G League's two-way
  contract tracker and call-up transaction log.

Full source URLs are included per-row in both CSVs.

## Known limitations

- Only 14 historical players, all "positive" (successful) cases — no labeled
  negative class, so this can't produce a calibrated probability of breakout.
- Advanced stats (PER, USG%, BPM, WS/48) aren't available for current
  prospects (the G League's official stats site wasn't scrapable), so the
  model uses basic box-score rate stats instead.
- Steals/blocks are missing for most current prospects and are
  median-imputed.
- Khris Middleton is a known "miss" by this model — his real G-League sample
  was only 3 games and doesn't look like a strong prospect profile by these
  stats, despite him becoming a multi-time All-Star. A good reminder that
  small-sample screening has real blind spots.

See the in-app Methodology tab for the full writeup.
