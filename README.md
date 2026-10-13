# CrowdWisdomTrading — X Sentiment & Price Prediction Pipeline

A quantitative pipeline that forecasts an asset's price at **n+0 … n+5** using only
historical prices and features derived from X (Twitter) sentiment. X posts are
collected with Apify, prices with yfinance, and recent news with Tavily/Exa as
**research context only** — news never enters the feature matrix.

Everything is evaluated with expanding-window **walk-forward validation** and
reported with MAE, RMSE, directional accuracy, Sharpe, Sortino and maximum
drawdown, always against a naive last-price baseline.

> **Honesty note.** This project never fabricates data. If a credential is
> missing, the affected stage says so and stops. `--demo` runs the entire
> pipeline on seeded synthetic data that is labelled synthetic in the database,
> the logs, the charts and the reports. Demo output is never presented as a
> market result.

---

## 1. Quick start

```bash
# 1. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows / VS Code
# source .venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. configuration template
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux

# 4. run the whole pipeline on labelled synthetic data (no credentials needed)
python app.py demo

# 5. check that your real configuration is valid
python app.py check

# 6. run the live pipeline
python app.py run-all
```

Other useful commands:

```bash
python app.py run-all --symbol NVDA --interval 1d --days 180
python app.py status                     # row counts per SQLite table
python app.py ingest                     # run a single stage
python app.py sentiment
python app.py features
python app.py model
python app.py charts
python app.py report
python -m pytest -q                      # test suite
```

Windows users can also use `scripts\run_demo.ps1` and `scripts\run_live.ps1`.

---

## 2. Configuration

Copy `.env.example` to `.env` and fill in what you have. Startup validation
(`python app.py check`) reports exactly which feature is unavailable and what to
set. Credentials are read from the environment only — never hard-coded, never
logged, never written to reports.

| Variable | Purpose |
| --- | --- |
| `ASSET_SYMBOL` | yfinance symbol: `BTC-USD`, `ETH-USD`, `TSLA`, `NVDA`, `SPY`, … |
| `PERIOD_DAYS` | lookback window (default 180) |
| `BAR_INTERVAL` | bar size: `1d`, `1h`, `60m`, `30m`, `15m`, `1wk`, `1mo`, … |
| `ANNUALIZATION_PERIODS` | periods/year for Sharpe/Sortino. Blank = auto (365 crypto, 252 equity) |
| `TRANSACTION_COST_BPS` | one-way cost in basis points for the strategy metrics (0 to exclude) |
| `APIFY_TOKEN` | Apify API token |
| `APIFY_ACTOR_ID` | actor id, e.g. `apidojo/tweet-scraper-v2` |
| `APIFY_ACTOR_INPUT_JSON` | raw JSON input for the actor (default `{}`) |
| `APIFY_DATASET_ID` | optional: read an existing dataset instead of starting a run |
| `APIFY_SEARCH_KEY` | input field the search terms are injected into (default `searchTerms`) |
| `APIFY_MAX_ITEMS` / `APIFY_TIMEOUT_S` | pagination cap and run timeout |
| `TAVILY_API_KEY` / `EXA_API_KEY` | news context providers |
| `SENTIMENT_BACKEND` | `vader` (default) or `finbert` |
| `MODELS` | comma separated: `naive,random_forest,hist_gbm,xgboost` |
| `FORECAST_HORIZONS` | default `1,2,3,4,5` |
| `MIN_TRAIN_SAMPLES` / `REFIT_EVERY` | walk-forward window floor and refit cadence |
| `DEMO_MODE` | `true` to run on labelled synthetic data |

### Configuring the Apify actor

Apify hosts many X scrapers with different input and output schemas, so the
actor is configurable rather than hard-coded.

1. Pick an actor (for example [apidojo/tweet-scraper-v2](https://apify.com/apidojo/tweet-scraper-v2)).
2. Set `APIFY_ACTOR_ID` to `username/actor-name` or the actor's id.
3. Set `APIFY_ACTOR_INPUT_JSON` to the input that actor expects:

```jsonc
// apidojo/tweet-scraper
{"searchTerms": ["(Bitcoin OR BTC OR $BTC) -filter:retweets"],
 "tweetLanguage": "en", "sort": "Latest", "maxItems": 3000}
```

**Search terms come from one of two places, and each term is a separate actor run
and a separate charge:**

| `APIFY_ACTOR_INPUT_JSON` contains | Terms used |
| --- | --- |
| a non-empty `searchTerms` list | exactly those terms, one run each |
| nothing under that key | generated from the asset: `$BTC`, `BTC`, `Bitcoin`, `crypto` |

So a compound query such as `"(Bitcoin OR BTC OR $BTC) -filter:retweets"` is one
run, whereas the generated fallback is four. Writing your own query therefore
costs less and keeps filters such as `-filter:retweets` intact. `APIFY_SEARCH_KEY`
changes which field the terms are read from and written to (default `searchTerms`;
other actors use `search_queries`, `keywords` or `term`).

4. If the actor uses a different field name for search terms, set it:

```dotenv
APIFY_SEARCH_KEY=searchTerms     # or search_queries / keywords / term
```

5. To avoid paying for a new run while developing, run once in the Apify
   console and reuse the dataset:

```dotenv
APIFY_DATASET_ID=AbCdEf123456
```

Actor ids pasted from the store work in any form — `user/name`, `user~name`, the
actor's id, or a full `https://apify.com/user/name` URL; the client converts them
to the `user~name` form the REST route requires.

The client starts the run, polls until it finishes, then pages through the
dataset in batches of 1000. HTTP 401/403 is reported as an invalid token, 404 as
an unknown actor, and a failed run status is surfaced — the pipeline never
claims a collection succeeded when it did not. Field names are normalised from
the common variants (`id`/`tweet_id`, `text`/`full_text`, `likeCount`/`likes`,
`retweetCount`/`retweets`, `viewCount`/`views`, `author.followers`/
`user.followers_count`). Records with no usable id, text or timestamp are
dropped and the count is logged.

Search queries are generated per asset: cashtag (`$BTC`), ticker (`BTC`), asset
name (`Bitcoin`) and keywords. Override with `ASSET_NAME` and `ASSET_KEYWORDS`.

---

## 3. Architecture

```
crowdwisdom_sentiment_pipeline/
├── app.py                     # CLI entry point (all stages)
├── requirements.txt
├── .env.example
├── config/
│   ├── settings.py            # env -> frozen AppSettings + validation
│   └── logging_config.py
├── src/
│   ├── ingestion/
│   │   ├── apify_x.py         # Apify actor client + post normalisation
│   │   ├── prices.py          # yfinance OHLCV
│   │   ├── news.py            # Tavily + Exa (context only)
│   │   ├── synthetic.py       # labelled demo data generator
│   │   └── timeutils.py       # UTC parsing and bar alignment
│   ├── storage/database.py    # SQLite schema + idempotent upserts
│   ├── sentiment/
│   │   ├── scoring.py         # VADER (+ finance lexicon) or FinBERT
│   │   └── aggregator.py      # posts -> bar-level sentiment
│   ├── features/builder.py    # causal feature matrix + targets
│   ├── models/
│   │   ├── registry.py        # naive / RF / HistGBM / XGBoost
│   │   └── train.py           # fit + forward projection
│   ├── validation/
│   │   ├── walk_forward.py    # expanding-window evaluation
│   │   └── metrics.py         # MAE/RMSE/direction/Sharpe/Sortino/MaxDD
│   ├── reporting/
│   │   ├── outlook.py         # one-page sentiment outlook
│   │   └── evaluation_report.py  # Markdown + PDF report
│   ├── visualization/
│   │   ├── theme.py           # charcoal / teal / amber research theme
│   │   └── charts.py          # PNG + self-contained interactive HTML
│   └── pipeline.py            # stage orchestration
├── data/{raw,processed}
├── outputs/{forecasts,charts,reports}
├── scripts/
└── tests/
```

Stages communicate through SQLite, so any stage can be re-run alone:

`ingest → sentiment → features → model → charts → report`

---

## 4. Data storage

SQLite at `data/pipeline.sqlite3` with six tables:

| Table | Natural key | Contents |
| --- | --- | --- |
| `prices` | symbol, interval, timestamp, source | OHLCV |
| `x_posts` | post_id, symbol, source | text, timestamp, engagement, followers, sentiment |
| `sentiment_bars` | symbol, interval, bar_ts, source | per-bar sentiment aggregates |
| `news_items` | news_id, source | headline, URL, date, summary, relevance |
| `forecasts` | run_id, symbol, model, horizon, origin_ts | predicted and realised values |
| `evaluation_metrics` | run_id, symbol, model, horizon, metric | metric values and sample sizes |

Every row carries a `data_source` tag (`yfinance`, `apify`, `synthetic`, …), so
demo and live data can never be mixed. Inserts use `INSERT OR IGNORE` on the
natural key, which makes re-ingestion idempotent.

---

## 5. Sentiment and feature engineering

**Scoring.** VADER's compound score, plus a capped market-lexicon adjustment
(±0.25) for terms such as *pump, dump, breakout, liquidation, rug*. Label
thresholds follow VADER's: compound ≥ 0.05 positive, ≤ −0.05 negative, else
neutral. `SENTIMENT_BACKEND=finbert` switches to `ProsusAI/finbert` if you
install `transformers` and `torch`; otherwise the pipeline reports the missing
dependency rather than failing silently.

**Aggregation.** Posts are binned onto the *actual* price bar starts with
`searchsorted`, not a calendar floor, so irregular session opens do not shift or
drop posts. A bar's aggregate contains only posts published inside that bar, so
it is complete by the time the bar closes.

**Point-in-time rule.** The feature row for bar `t` uses only information known
at the close of bar `t`: that bar's OHLCV and the sentiment of posts published
inside it. Forward-looking values exist only in the target frame
(`close.shift(-k)`). Bars without posts carry the last observed score forward
(causal forward fill) and record `bars_since_post`.

| Feature | Group | Definition |
| --- | --- | --- |
| `ret_1`, `ret_2`, `ret_3`, `ret_5` | price | k-bar simple returns |
| `log_ret_1` | price | 1-bar log return |
| `vol_5`, `vol_10`, `vol_20` | price | rolling std of `ret_1` |
| `dist_sma5`, `dist_sma20` | price | close / k-bar SMA − 1 |
| `sma_spread` | price | 5-bar SMA / 20-bar SMA − 1 |
| `range_ratio` | price | (high − low) / close |
| `volume_ratio` | price | volume / 20-bar mean volume |
| `rsi_14` | price | Wilder RSI(14) / 100 |
| `drawdown_from_high` | price | close / running max close − 1 |
| `sent_mean` | sentiment | mean compound score of the bar's posts |
| `sent_weighted` | sentiment | engagement-weighted mean sentiment |
| `sent_pos_ratio`, `sent_neg_ratio`, `sent_net` | sentiment | label shares and their difference |
| `sent_volume` | sentiment | log1p(post_count) |
| `sent_engagement` | sentiment | log1p(likes + 2·reposts + views/100) |
| `sent_followers` | sentiment | log1p(mean author followers) |
| `sent_change_1`, `sent_change_3`, `sent_momentum` | sentiment | sentiment deltas and 7-bar deviation |
| `sent_lag_1..3` | sentiment | lagged mean sentiment |
| `bars_since_post` | sentiment | bars since the last bar with posts |
| `dow_sin/cos`, `month_sin/cos` | calendar | cyclical time encodings |

Engagement weight: `w = log1p(likes + 2·reposts + views/100 + followers/1000)`,
then `sent_weighted = Σ(score·w) / Σw`.

News from Tavily/Exa is stored, deduplicated by canonical URL/title and shown in
the outlook report. It is **not** a model input.

---

## 6. Modelling

Targets are **forward returns** `r_k = close[t+k]/close[t] − 1`, one column per
horizon, fitted independently (no recursive chaining). Price forecasts are
`P̂ = P_t · (1 + r̂_k)`, which keeps models scale-free: raw price levels drift far
outside the training range in a trending market and degrade badly.

| Model | Role |
| --- | --- |
| `naive` | last observed price carried forward (zero predicted return) — the benchmark |
| `random_forest` | RandomForestRegressor, robust at small sample sizes, gives importances |
| `hist_gbm` | HistGradientBoostingRegressor, handles sparse sentiment bars natively |
| `xgboost` | optional; skipped with a warning when not installed |

**n+0 is never a forecast.** The forward table emits n+0 as the last observed
close with model `observed`, then n+1 … n+5 as model output.

---

## 7. Walk-forward validation

Expanding window. At origin `i` (close of bar `i`) and horizon `k`:

1. Training rows are `j ≤ i − k`, so every training target `close[j+k]` was
   already realised at the origin. Rows whose target lands at or after the
   origin are excluded — the leakage a naive time-based split introduces.
2. Models refit every `REFIT_EVERY` origins on a window that only grows.
3. The forecast and the realised value are stored together.

No scaler or imputer is fit across the split: the feature frame contains only
causal columns and each refit sees strictly past observations. Two tests enforce
this: one asserts every recorded (last training bar, origin) pair satisfies the
constraint, the other asserts that truncating the end of the series does not
change any earlier forecast.

---

## 8. Metrics — exact definitions

* **MAE / RMSE** on price levels; `mae_ret` / `rmse_ret` on returns.
* **Directional accuracy** — share of forecasts whose predicted sign matches the
  realised sign. Zero realised returns and zero forecasts (the naive baseline
  makes no directional claim) are excluded, so a non-directional model reports
  *undefined* rather than a misleading 0.
* **Trading signal** — `sign(predicted return)`: long, short, or flat. Entry at
  the origin close, exit `k` bars later, origins sampled `k` bars apart so
  holding periods never overlap.
* **Trade return** — `position × realised k-bar return − 2 × TRANSACTION_COST_BPS/10⁴`
  (entry plus exit). Set `TRANSACTION_COST_BPS=0` to see cost-free numbers; the
  value used is printed in every report.
* **Sharpe / Sortino** — mean over standard deviation (downside deviation for
  Sortino), annualised with `sqrt(periods_per_year / k)`, risk-free rate 0.
* **Maximum drawdown** — largest peak-to-trough fall of the compounded equity
  curve of the sampled trade returns, reported as a negative fraction.
* **Benchmark** — an always-long `buy_hold` row over the same sampled origins.

These are reported with their sample sizes. A Sharpe computed on a few dozen
blocks has a confidence interval wider than the estimate, and none of this is
presented as evidence of profitability.

---

## 9. Outputs

`outputs/charts/`

| File | Shows |
| --- | --- |
| `{SYMBOL}_price_vs_sentiment.png` / `.html` | close vs mean sentiment, shared time axis |
| `{SYMBOL}_sentiment_volume.png` | sentiment intensity by sign, post volume, engagement |
| `{SYMBOL}_sentiment_price_matrix.png` | time-aligned matrix of Δ sentiment vs price moves |
| `{SYMBOL}_sentiment_return_density.png` | Δ sentiment vs next-bar return density |
| `{SYMBOL}_actual_vs_predicted_{model}.png` | realised vs out-of-sample forecast per horizon |
| `{SYMBOL}_forward_projection.png` | n+0 … n+5 projection per model |
| `{SYMBOL}_evaluation_summary.png` | MAE/RMSE by horizon and the risk-metric table |

`outputs/reports/`

* `{SYMBOL}_sentiment_outlook_{date}_{run}.md` — one page: measured snapshot,
  interpretation with explicit thresholds, forecast summary, news context,
  risks. Measured numbers and interpretation are separate sections.
* `{SYMBOL}_evaluation_report_{run}.md` / `.pdf` — methodology, coverage,
  features, models, validation, metrics, baseline comparison, charts, limitations.

`outputs/forecasts/` — forward forecast CSV and the full walk-forward prediction CSV.

The PDF is rendered with `reportlab`; if it is not installed the Markdown report
is still written and the skip is logged.

---

## 10. Demo mode

`python app.py demo` (or `DEMO_MODE=true`) generates a seeded price path and
matching posts from `src/ingestion/synthetic.py`. Post text is synthetic, but
sentiment is computed by the real VADER scorer. Every record is tagged
`data_source = 'synthetic'`, and the logs, charts, reports and CLI summary are
labelled **DEMO**. Demo output must not be quoted as a market result.

---

## 11. Tests

```bash
python -m pytest -q
```

Coverage includes timestamp normalisation, post normalisation across actor
schemas, sentiment aggregation, feature engineering, horizon construction,
future-data-leakage prevention, walk-forward training order, every metric
against hand-computed values, storage idempotency, missing credentials, empty
API responses, and an end-to-end demo run that checks the artefacts on disk.

---

## 12. Known limitations

* **Sample size.** 180 daily bars leave roughly 90–110 walk-forward origins.
  Metrics on that sample are noisy; the reports say so next to every number.
* **Social-media sampling bias.** X posts over-represent vocal, active accounts.
  Bots and coordinated campaigns are not filtered.
* **Sentiment model limits.** Lexicon/model-based scoring mis-handles sarcasm,
  memes and multi-asset threads.
* **Correlation, not causation.** Sentiment moves with price; a model can simply
  restate the last return.
* **Execution.** Fills are assumed at bar closes with a fixed cost. Slippage,
  funding and partial fills are ignored.
* **Single asset, single window.** Nothing here demonstrates stability across
  assets or regimes.

---

## 13. Reproducing a run for submission

1. `pip install -r requirements.txt`
2. Fill `.env` with your `APIFY_TOKEN`, `APIFY_ACTOR_ID`, `APIFY_ACTOR_INPUT_JSON`,
   `TAVILY_API_KEY`, `EXA_API_KEY`.
3. `python app.py check` — confirm every data source is available.
4. `python app.py run-all`
5. Share the `run_id` printed at the end. `python app.py status` shows what is
   stored, and `outputs/reports/` holds the evaluation report and outlook.
