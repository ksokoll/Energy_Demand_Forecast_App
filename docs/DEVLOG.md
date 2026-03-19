# Development Log

## Fourier Features vs. Raw Integers
Started with raw `hour` and `month` as features. Both worked, but 
`month` didn't even make the top 20 in feature importance. After 
implementing sine/cosine encoding, `sin_day_of_year` jumped to #1. 
The reason: raw integers create an artificial gap between December (12) 
and January (1). Fourier terms preserve the circular proximity. This 
was the single biggest "aha" moment in the project.

## Feature Engineering > Model Complexity
Ran LightGBM on raw features (weather only, no engineering): 
RMSE 4,018, worse than the naive baseline (3,815). Same model, 
same hyperparameters, with engineered features: RMSE 2,284. 
Then ran Optuna (100 trials): 0% improvement. Then ran H2O AutoML 
(Stacked Ensemble): also worse than our manual LightGBM. The 
performance came entirely from features, not from the model.

## The Leakage That Almost Happened
Initially planned to use lag_1h through lag_23h for "short-term 
patterns." Caught the mistake before training: for 24h-ahead 
forecasting, lag_1h means "the value 23 hours in the future." 
Every lag under 24 would have been data leakage. Added an assert 
guard (`assert lag >= FORECAST_HORIZON`) to prevent this permanently.

## Holiday Features: Right Decision, Wrong Metric
Holiday features showed no aggregate RMSE improvement. Almost 
removed them. Then checked per-holiday error: the model's MAE 
on weekday holidays was double the normal weekday error (2,968 vs 
1,452 MW). The features help exactly where they should, but 8 
holidays in a test year are invisible in the aggregate metric.

## DDD is Simpler Than It Sounds
Expected DDD to add significant complexity. In practice, the code 
looks almost identical to what I would have written without it: 
functions in files, organized by responsibility. The difference is 
not in the code but in the decisions behind it: which module imports 
from which, what stays pure (features) vs. what does I/O (store), 
where the boundaries are. The architecture emerges from asking 
"what changes independently?" rather than following a template.
