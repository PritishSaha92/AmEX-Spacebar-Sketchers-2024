# American Express Campus Challenge 2025 — Decision Science (National Finalist)

Achieved 0.59 MAP on the final, unseen evaluation set with a 3-stage GBDT–Transformer ensemble for offer ranking.

## Highlights
- Engineered 3k+ features with a parallelized pipeline and leakage-free customer/offer profiles with advanced temporal metrics.
- Trained a Transformer on GBDT residuals using a listwise ranking loss to correct systematic errors.
- National Finalist (Decision Science Track), American Express Campus Challenge (Jul’25).

## Repository Structure
- `notebooks/` — Final notebook: `amex final submission new.ipynb` (end-to-end pipeline and experiments)
- `src/` — Helper script(s), e.g., `scoring_v1_r3.py`
- `artifacts/` — Submissions, parameters, and importance reports (kept for reference)
- `docs/assets/` — Final presentation, figures, and data dictionary
- `data/raw/` — Raw datasets from the competition (tracked)
- `data/interim/` — Prepared datasets (gitignored)
- `data/processed/` — Post-processed datasets (gitignored)

## Key Artifacts
- Round 2 submission (evaluated on public leaderboard): `artifacts/r2_submission_fileSpacebar Sketchers.csv`
- Round 3 submission (final): `artifacts/final_residual_ensemble_submission.csv`
- Final presentation: `docs/assets/r3_ppt_Spacebar Sketchers.pdf`
- Feature importance (Round 3): `artifacts/r3_importancescore_Spacebar Sketchers.xlsx`
- Additional: `artifacts/feature_importance.xlsx`, `artifacts/best_lgbm_ranker_params.json`, `artifacts/top_100_features.json`

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place required raw datasets in `data/raw/` (see `data/README.md`).
3. Open and run `notebooks/amex final submission new.ipynb`.

Notes:
- Large datasets generated during processing (interim/processed) are intentionally not tracked by git.
- The full feature engineering and modeling pipeline lives in the final notebook above.

## Reproducibility
- 3-stage ensemble: GBDT base ranker → residual modeling → Transformer with listwise ranking loss.
- `src/scoring_v1_r3.py` provides reference scoring logic for inference.

## Acknowledgements
- American Express Campus Challenge team and organizers.
- Teammates/peers for constructive feedback.


