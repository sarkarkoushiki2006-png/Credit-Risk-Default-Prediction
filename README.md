# Credit Risk / Loan Default Prediction Model

## Overview
A machine learning pipeline that predicts the probability a borrower will default 
on a loan, using consumer lending data (LendingClub). The project compares an 
interpretable baseline model against a more complex ensemble model, evaluates 
performance using credit-risk-specific metrics, and translates model output into 
a bank-style credit scorecard with risk grades (A–E).

## Business Problem
Lenders need to estimate the likelihood a loan applicant will fail to repay, in 
order to price loans appropriately, set approval thresholds, and manage portfolio 
risk. This project builds and validates a default prediction model that could 
support that decision, and evaluates it the way credit risk teams actually 
validate scorecards — not just by accuracy.

## Dataset
- Source: LendingClub loan data (9,578 loans, 14 features)
- Target variable: `not.fully.paid` (1 = defaulted, 0 = fully repaid)
- Class balance: ~16% default rate (imbalanced — addressed via class weighting)
- Features: FICO score, interest rate, installment, log annual income, DTI, 
  revolving balance/utilization, credit inquiries, delinquencies, public 
  records, loan purpose

## Methodology
1. **EDA** — examined FICO distribution by outcome, default rate by loan purpose, 
   and feature correlations
2. **Feature engineering** — one-hot encoded categorical purpose variable; 
   scaled numeric features for logistic regression
3. **Modeling** — trained two models:
   - Logistic Regression (with balanced class weights) — chosen for interpretability
   - XGBoost — chosen to test for non-linear lift
4. **Evaluation** — AUC-ROC, KS statistic, and gains/lift chart (industry-standard 
   credit risk metrics, not accuracy, given class imbalance)
5. **Scorecard construction** — converted model probabilities into a points-based 
   score using the points-to-double-odds (PDO) method, then binned into risk 
   grades A (lowest risk) through E (highest risk)

## Results
| Model               | AUC                | KS Statistic  |

| Logistic Regression | 0.6757380001972499 | 0.2402        |
| XGBoost             | 0.6399046971435097 | 0.1969        |

- Logistic regression outperformed XGBoost on this dataset — likely due to the 
  near-linear nature of credit risk drivers, the modest dataset size, and 
  untuned XGBoost hyperparameters. This mirrors why regulated lenders often 
  favor interpretable scorecards over black-box models in production.
- Risk grades showed a monotonic relationship with actual default rates: 
  Grade A ([X]% default) → Grade E ([X]% default), validating that the score 
  meaningfully separates risk.
- Top predictive features (by logistic regression coefficient): [list top 3-4]

## Key Takeaways
- FICO score and interest rate emerged as the strongest individual predictors 
  of default, consistent with their central role in real-world credit 
  underwriting.
- Loan purpose matters: small business loans showed a meaningfully higher 
  default rate than purposes like credit card refinancing or debt 
  consolidation, reflecting the greater income volatility of small business 
  borrowers.
- The simpler, interpretable model (logistic regression) outperformed the more 
  complex ensemble model (XGBoost) on this dataset — a reminder that model 
  complexity isn't automatically better, especially on smaller datasets where 
  credit risk relationships tend to be close to linear.
- Risk grades derived from the model's scorecard showed a clear, mostly 
  monotonic relationship with actual default rates (Grade A ≈ [X]% default 
  vs. Grade D ≈ [X]% default), confirming the model produces a genuinely 
  usable risk ranking, not just a black-box probability.
- Sorting borrowers by predicted risk and reviewing only the highest-risk 
  decile would capture a disproportionate share of actual defaults — 
  demonstrated via the KS statistic and gains chart — which is the practical 
  value a lender would get from deploying this model (prioritizing review, 
  not reviewing every loan equally).

## Limitations
- Dataset is a small, well-known teaching subset — not the full LendingClub 
  data (2007–2015, ~890K rows), so results may not generalize
- XGBoost hyperparameters were not tuned via grid/random search
- Risk grades built with equal-width binning (`pd.cut`); equal-population 
  binning (`pd.qcut`) would give more statistically reliable grades

## Next Steps
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV) for XGBoost
- WOE/IV binning as an alternative, more banking-native feature engineering approach
- Test on the full LendingClub dataset for more robust validation

## Tech Stack
Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn
