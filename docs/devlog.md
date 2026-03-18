## Testing Journey

### Initial State: 24 tests, major gaps

The first test pass covered calendar features, the feature contract,
lag features, and the API endpoints via mocked service. Coverage looked
green on the surface, but a self-review revealed significant blind spots:

- `features/weather.py`: zero tests. The Kelvin conversion, temp_deviation
  calculation, and interaction features with their implicit dependency on
  calendar features were completely unprotected.
- `features/holiday.py`: zero tests. The bridge day detection (the most
  complex single calculation in the feature code) and the days_since_holiday
  cap at 30 were untested.
- `data/store.py`: zero tests. Parquet vs CSV loading, error cases for
  missing files and wrong formats, the InsufficientHistoryError path.
- `serving/predict.py`: only indirectly touched through the mocked API test.
  The actual orchestration (fetch history, build features, run model) was
  not tested.
- API edge cases: the 500 error handler and health endpoint with model=None
  had no coverage.

### Round 2: Closing the gaps

Added 35 tests across 5 new test files:

- `test_weather.py` (9 tests): Kelvin conversion, deviation always positive,
  squared term correctness, weekend interaction zero on weekdays / nonzero
  on weekends, and critically `test_fails_without_calendar_features` which
  explicitly protects against pipeline ordering changes.
- `test_holiday.py` (10 tests): January 1 is a holiday, January 2 is not,
  days_since increments correctly, capped at 30, density counts nearby
  holidays, bridge day detection, immutability guard.
- `test_store.py` (11 tests): Parquet and CSV loading, three error cases
  (missing file, unsupported format, missing time column), sorted output,
  correct row count, all rows before timestamp, most recent rows returned,
  InsufficientHistoryError, and copy-not-view guard.
- `test_predict.py` (5 tests): 24 predictions returned, correct timestamps,
  plausible MW range, model version in response, ModelNotFoundError on
  missing artifact.
- `test_api.py` additions (2 tests): 500 error on unexpected exception,
  health endpoint reports model_loaded=False when model is None.

### Final State: 61 tests, 96% coverage

| Module | Coverage | Notes |
|---|---|---|
| config.py | 100% | |
| models.py | 100% | |
| features/calendar.py | 100% | |
| features/weather.py | 100% | |
| features/lag.py | 100% | |
| features/pipeline.py | 100% | |
| features/holiday.py | 96% | Untested: no-prior-holiday fallback |
| data/store.py | 100% | |
| serving/api.py | 100% | |
| serving/predict.py | 100% | |
| main.py | 0% | Entrypoint wiring, not unit-testable |

61 tests in 11.7 seconds. All domain logic and serving code at 96-100%.
The only uncovered line is a defensive fallback in holiday.py (line 61)
for the edge case where no prior holiday exists in the dataset. main.py
at 0% is expected: it only wires components together and runs on server
start.

### Key decisions

**Test pyramid followed:** 39 small tests (features, pure functions, no I/O),
17 medium tests (store with temp files, API with TestClient), 5 integration
tests (predict service with real LightGBM model). Ratio roughly 65/28/8%,
close to the 80/15/5 recommendation from SE at Google.

**Contract test as Hyrum's Law protection:** `test_feature_column_order_matches_config`
verifies that the pipeline output has exactly 51 columns in the exact order
the trained model expects. If someone renames or reorders a feature, this
test fails immediately instead of the model producing silent wrong predictions.

**Self-review drove the second round:** The gap analysis was not prompted
externally. Reviewing the coverage report and thinking "what would break
if someone changed this" identified exactly the right tests to add. This
is the code review mindset from SE at Google Ch. 9, applied to your own work.