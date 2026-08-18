# Project Journal

Progress log for the flashcard retention ranking system. Organized by project stage.

---

## 1. Exploratory Data Analysis
`eda.ipynb`

### Steps
1. **Parse the raw SLAM file.** Wrote `write_sents()` to strip comment lines, pull `user_id` and `time` off the exercise headers, and flatten each token line into a tab-separated row carrying its full source sentence.
2. **Load into pandas.** Columns: `user_id`, `sent_id`, `token`, `POS`, `Morpho-Syntactic_Features`, `Dependency-Relation`, `Dependancy-Head`, `time`, `p_recall`, `sentence`. Added `nth_occurrence` as a per-(user, token) cumulative count.
3. **Inspect distributions** with `.info()` / `.describe()`, then sorted by `time` descending and plotted a histogram to find where the tail actually starts.
4. **Handle missing values and duplicates.** Dropped exact duplicate rows, then dropped rows with null `time`.
5. **Trend examination.**
   - Aggregated exercises per `sent_id` and regressed recall time against average recall accuracy for a single user (`Nsr+jY0A`) as a sanity check on the predictor.
   - Grouped by `POS` to compare average recall time, average recall accuracy, and count.
   - Grouped by `token` to find the slowest-recall words.
6. **Clean and export** to `data/fr_en_cleaned_train.csv`.

### Notes
- Null `time` rows serve no purpose here — the whole point of the data is the relationship between recall time and recall accuracy, so rows without a time can't contribute.
- The `time` column is badly skewed. 5 hours obviously isn't real, but where to cut? 20 mins is ~3 z-scores from the mean, which still felt too generous. Settled on removing exercises over 2 minutes: slightly arbitrary but reasonable, and I can revisit it later.
- Rationale for the cutoff: these are individual questions, not whole exercises, so long durations most likely mean the user got distracted and came back. That destroys the meaning of the value — it's no longer the user's true recall time.
- The `X` POS tag was unclear at first. **Answer:** it's the universal placeholder for "other" — e.g. the `t` in *est-t-il* — things that sit outside normal sentence grammar.
- `PUNCT` has both high average recall time and low average recall accuracy. In French this mostly means hyphens in compound words and inversion. Suggests **compound words will be harder** for learners, which is useful for ranking.
- The slowest tokens are all very low-frequency. High recall time may not mean *difficult* so much as *unfamiliar* — possibly rare verb forms or accent-heavy words. Just a theory at this point.
- Caveat on the POS timing charts: `time` is recorded per whole question, not per token, so "average time by POS" really means "average time on questions containing that POS."
- **Added later, from §3:** once I established the label is an error indicator, the `avg_recall` column in these charts reads as an **error rate** — higher is worse. The `PUNCT` conclusion above still holds and in fact gets stronger: `PUNCT` sits at a **0.760** error rate and `X` at 0.746, against a corpus average of 0.166. Easiest tags are `INTJ` (0.058) and `PROPN` (0.100). The one trap I hadn't spotted is `ADP` (prepositions) at 0.401.

---

## 2. Feature Engineering
`feature_engineering.ipynb`

### Steps
1. **User-level history features**, all computed from prior attempts only:
   - `overall_recall_rate` — lagged expanding mean of `p_recall` per user.
   - `token_recall_rate` — lagged expanding mean per (user, token).
   - `token_streak` — consecutive correct streak coming into the current attempt, reset on every miss.
   - First-ever attempts filled with neutral priors: `0.5` for rates, `0` for the streak.
2. **Isolated a token table** on `token`, `POS`, `Morpho-Syntactic_Features`, `sentence` and deduplicated.
3. **Merged in Lexique/OpenLexicon features** on lower-cased orthographic form: `syllable_count` and `ortho_freq`.
4. **Built the cognate feature:**
   - Translated every unique French sentence to English with `Helsinki-NLP/opus-mt-fr-en` (MarianMT).
   - Wrote `data/alignment_input.txt` in `french ||| english` format and ran `awesome-align` (mBERT, softmax extraction) to get word-level alignments.
   - Parsed `aligned_word_pairs.txt` back onto tokens via a `sent_id` keyed to line order; kept multiple English matches joined with `|`.
   - Computed `lev_ratio` — length-normalized Levenshtein similarity in `[0, 1]`, lower-cased on both sides, taking the max over candidates.
5. **Expanded `Morpho-Syntactic_Features`** into one column per key: `Definite`, `Gender`, `Number`, `Person`, `PronType`, `Mood`, `Tense`, `VerbForm`, `Reflex`. Dropped `fPOS` as redundant with `POS`.
6. **Joined token features back** onto the main dataframe and exported `data/fr_en_features_complete.csv`.
7. **Produced a null-filled variant** for logistic regression (`fr_en_features_complete_filled.csv`), plus a features-only file.

### Notes
- **Leakage guard is the main constraint on all three history features** — every one is a *history* feature, so it must be computed from a user's prior attempts only. Hence the `shift()` before every expanding window.
- The `0.5` fill isn't a guess at the truth, it's a statement: "we have no history on this, so assume a 50/50 chance of recall."
- Kept duplicate tokens in the token table on purpose. *suis* can be a verb (first person of *être*) or a noun (plural of *sui*). Dropping duplicates would lose the POS/morpho distinction the model needs to learn the right association.
- `ortho_freq` ranges from 11000+ down to under 0.3, so it's log-transformed (`log1p`) rather than used raw.
- Motivation for the cognate feature: SLA research says learners retain words that resemble their native language. Since this is an English-based course I assume users can lean on English to infer French. May later narrow to countries where that's most plausible (US, CA, etc.).
- Word alignment is **sparse** — function words and morphological forms often get dropped, so not every token gets an English mapping. Those stay `NaN` and become `lev_ratio = 0` in the filled variant (treated as non-cognate).
- Chose the Levenshtein *ratio* over raw edit distance specifically because the ratio isn't biased by word length.
- A lot of the morpho columns are heavily null. Anticipating these will need refining once the model shows which features actually carry weight.
- Null-filling strategy per column: categoricals → `'N/A'` string; `syllable_count` → French vowel-cluster heuristic (collapse vowel groups, subtract silent final *e*, return 0 for pure punctuation); `ortho_freq` → dataset minimum, since a missing entry implies punctuation or extreme rarity; `lev_ratio` → 0.
- **Added later, from §3:** the label turned out to be an error indicator, which flips the reading of all three history features without changing the code. `overall_recall_rate` and `token_recall_rate` are **error** rates, and `token_streak` counts consecutive **misses** — it resets on a correct answer, not on a wrong one. The features are still doing the right thing and the leakage guard is unaffected; only the names read backwards. `ranking_system.py` documents this where it rebuilds them at runtime.

---

## 3. Model Training & Testing
`model.ipynb`

### Steps
1. **Temporal train/test split.** For each user, repeatedly peel their last-seen sentence into the test set, then next-to-last, and so on, until the test set hits 25% of all rows — never taking more than half of any single user's sentences. Result: 656,753 train rows (74.99%) / 218,982 test rows (25.01%).
2. **Defined three feature sets** to isolate the contribution of linguistics:
   - **Baseline** — history/time features only (`time`, `nth_occurrence`, `overall_recall_rate`, `token_recall_rate`, `token_streak`).
   - **Linguistic Only** — token-descriptive features only.
   - **Complete** — everything.
3. **Logistic regression.** `ColumnTransformer` with `OneHotEncoder` on categoricals, numerics passed through; `LogisticRegression(max_iter=1000, class_weight='balanced')`.
4. **Gradient boosted tree.** Same three splits, `OrdinalEncoder` instead of one-hot, `LGBMClassifier(class_weight='balanced')` with categorical column indices declared explicitly.
5. **Evaluated with ROC-AUC** on the held-out temporal test set.

6. **Fixed the LightGBM categorical declaration and retrained** (see notes) — the scores below are post-fix.
7. **Trained a fourth variant** with `time` dropped, for use at ranking time.

### Results

| Model | Logistic Regression | LightGBM |
|---|---|---|
| Baseline (history only) | 0.7199 | 0.7339 |
| Linguistic only | 0.6589 | 0.7172 |
| Complete | 0.7370 | **0.7735** |
| Complete, no `time` | — | 0.7583 |

*(Logistic regression scores before `class_weight='balanced'`: 0.7196 / 0.6582 / 0.7352 — essentially unchanged. LR is unaffected by the categorical fix below; only the GBT numbers moved.)*

### Notes
- The split is temporal by design so the model trains on features that change over time and is tested on predicting performance as practice continues. Peeling by sentence block keeps each practice instance contiguous.
- 25% is a *target ceiling*, not a guarantee — if the half-per-user cap is hit first, the split stops there.
- **The label is inverted from what its name suggests.** `p_recall = 1` means the learner got the token **wrong**, not that they recalled it. Three independent checks agree: the easiest, most frequent words carry the *lowest* rate (`Je` 0.043, `suis` 0.064), the rate *falls* with practice (0.223 on first exposure → 0.105 by the tenth), and label-1 attempts take longer (17.2s vs 13.1s). Only an error label produces all three. Consequences worth remembering: the models predict **P(mistake)**, `overall_recall_rate` / `token_recall_rate` are really *error* rates, and `token_streak` counts consecutive **misses** (reset by a correct answer), not consecutive correct answers. Nothing is broken by this — the features are still valid — but every name in the pipeline reads backwards.
- **The `categorical_feature` indices were wrong and it cost ~0.032 AUC.** I computed them against the input frame, but `ColumnTransformer` writes its transformer block out first, so after transform the categoricals sit at 0–10 and the numerics at 11–19. The old list declared `time`, `overall_recall_rate`, `ortho_freq` and `lev_ratio` categorical while leaving `Gender`/`Tense`/`Mood` to be read as numbers. Measured directly on the complete feature set: old indices **0.7417** (reproduces the original run exactly), no declaration at all **0.7723**, correct indices **0.7735**.
- Second trap in the same place: LightGBM 4.6 **discards** `categorical_feature` when it is passed to the `LGBMClassifier` constructor (*"categorical_feature keyword has been found in `params` and will be ignored"*). It has to go to `fit()` — through a pipeline that means `pipeline.fit(X, y, classifier__categorical_feature=CAT_IDX)`.
- **The core hypothesis holds, and more strongly than the first pass suggested.** Complete beats history-only by **+0.040** for LightGBM (0.7339 → 0.7735). Linguistic features earn their place.
- Linguistic-only has no idea who the learner is, yet still reaches 0.7172 — the token features carry real signal on their own, and now sit only ~0.017 below the history-only baseline.
- LightGBM now clears the best logistic regression by **+0.037**, a wide enough margin to justify the tree over the simpler linear model. (Before the fix the gap was a negligible +0.0047, which had me nearly calling it a wash.)
- Class balance is skewed: 108,110 positive vs 548,643 negative in train, hence `class_weight='balanced'` throughout. Given the label direction above, "positive" here means *the learner made a mistake* — so ~16.5% of attempts are errors.

### Packaging for serving
1. **Extracted shared logic into `features.py`** — `CATEGORICAL_FEATURES`, `MORPHO_COLS`, `parse_morpho`, and `estimate_syllables_fr`, imported by both notebooks and by `ranking_system.py` so training and inference cannot drift apart.
2. **Exported the fitted pipelines** with `joblib` to `models/`, each with a `.metadata.json` recording its `feature_columns`, ROC-AUC, label direction, and library versions.
3. **Froze the linguistic features** into `models/token_features.parquet` — 2,485 unique `(token, POS)` rows.
4. **Wrote `ranking_system.py`** (`UserState`, `RecallModel`, `rank`) and `test_ranking_system.py`.

- **`time` cannot be used at ranking time.** It is how long an attempt *took*, and the ranking system chooses a card *before* the attempt. Hence the no-`time` variant, which is what `ranking_system.py` serves by default. The honesty costs **0.0152** AUC (0.7735 → 0.7583) — worth it over feeding the model a placeholder for a feature it was trained to trust.
- The token table is collapsed to one row per `(token, POS)` because a flashcard is a standalone word. Verified that the genuinely lexical features survive this exactly (`ortho_freq` varies in 0/2485 groups, `syllable_count` in 8/2485), but sentence-dependent ones necessarily take a modal value — `lev_ratio` differs from the original row ~14% of the time, `Dependancy-Head` ~31%. Those two columns are the weakest part of the serving story, since dependency structure is close to meaningless for an isolated word.
- History features are rebuilt at runtime by `UserState` rather than read from the training CSV. It keys on `token` alone, matching how training grouped them — `(user_id, token)`, no POS.
- `predict_proba[:, 1]` is P(mistake), so **high score = hard card = high priority** and `rank()` sorts descending. This is the single easiest thing to get backwards in the whole project.

---

## 4. Ranking System
`ranking_system.py` — *in development*

### Steps

### Notes

---
