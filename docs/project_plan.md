# Energy Demand Forecasting — Projektplan

## Rahmenbedingungen
- **Budget:** 45–60h (3 Wochen × 15–20h)
- **Deadline:** 31. März 2026 (Repo "vorzeigbar")
- **Ziel:** Production-grade ML-System, das den Sprung von Notebook → Deployable System zeigt
- **Zielgruppe des Repos:** Hiring Manager / Technical Leads bei AI-Consulting-Firmen

---

## Prioritäten (MoSCoW)

### Must Have (ohne das ist das Projekt nicht vorzeigbar)
- Saubere Package-Struktur (`src` layout, `pyproject.toml`, installierbar)
- Daten-Ingestion mit Validierung
- Feature Engineering als sklearn Custom Transformers (Lag, Calendar, Weather)
- Training Pipeline mit chronologischem Split + Baseline
- LightGBM als Primary Model, mindestens ein Vergleichsmodell
- MLflow Experiment Tracking (Params, Metrics, Artifacts)
- Evaluation: RMSE/MAE auf Original-Skala, Baseline-Vergleich, Feature Importance
- FastAPI Serving Endpoint (Predict)
- Dockerfile (funktionsfähig, nicht nur Platzhalter)
- README mit Business Context, Architektur, Ergebnissen
- ADR (Architecture Decision Records)
- Tests für kritische Pfade (Feature Engineering, Prediction Endpoint)

### Should Have (starker Portfolio-Impact, einplanen wenn Woche 1–2 im Zeitplan)
- SHAP / Explainability Plots
- Hyperparameter Tuning (Optuna oder manuelles Grid)
- GitHub Actions CI (Lint + Tests)
- Fourier Terms als saisonale Features
- Model Registry in MLflow (nicht nur Tracking)
- EDA-Notebook als separates Artefakt (Jupyter, nicht im Package)

### Could Have (nice to have, nur wenn Zeit übrig)
- Drift Detection Konzept (dokumentiert, ggf. Skeleton-Code)
- Zweites Modell (XGBoost) mit Ensemble
- EWMA Features
- Pre-commit Hooks (ruff, mypy)
- Monitoring Dashboard Konzept

### Won't Have (bewusst out of scope für V1)
- PySpark / Fabric Integration (→ separates Projekt)
- Streaming / Real-time Inference
- Frontend / Dashboard
- Kubernetes Deployment
- Full CD Pipeline mit automatischem Deployment

---

## Wochenplan

### Woche 1: Foundation + Feature Engineering (15–20h)
**Ziel:** Am Ende der Woche läuft `python -m energy_forecast.train` end-to-end durch.

| Tag | Block | Aufwand | Deliverable |
|-----|-------|---------|-------------|
Data Ingestion + Validation | 3h | Loader mit Schema-Validierung, Null-Strategie dokumentiert |
EDA (timeboxed!) | 2–3h | Jupyter Notebook: Verteilungen, Saisonalität, Korrelationen, Missing Values |
Feature Engineering | 5–6h | Custom Transformers: CalendarFeatures, LagFeatures, WeatherFeatures |
Erste Training Pipeline | 3–4h | Chronologischer Split, Baseline (Naive Seasonal), erster LightGBM Run
Projekt-Skeleton | 3–4h | Package-Struktur, pyproject.toml, Config, CI Grundgerüst |

**Risiko Woche 1:** EDA-Rabbit-Hole. Timebox auf 3h, danach weitergehen.
**Checkpoint:** Pipeline läuft, MLflow loggt, Baseline-RMSE steht.

### Woche 2: Training + Evaluation + API (15–20h)
**Ziel:** Modell ist evaluiert, API liefert Predictions, erste Tests stehen.

| Tag | Block | Aufwand | Deliverable |
|-----|-------|---------|-------------|
| Mo–Di | Feature Iteration | 4–5h | Feature Importance Analyse → gezielt neue Features, Multikollinearität prüfen |
| Mi | Evaluation Pipeline | 3h | Back-Transformation, RMSE/MAE auf Original-Skala, Residual-Analyse |
| Do | Hyperparameter Tuning | 2–3h | Optuna oder Config-basiertes Grid, Ergebnisse in MLflow |
| Fr | FastAPI Endpoint | 3–4h | /predict und /health, Pydantic Request/Response Models |
| Sa–So | Tests | 3–4h | pytest: Feature Transformers, Prediction Endpoint, Data Validation |

**Risiko Woche 2:** Over-Tuning. Klar definieren: max 2–3h für Hyperparameter, dann weiter.
**Checkpoint:** API läuft lokal, Tests grün, Metriken dokumentiert.

### Woche 3: Deployment + Dokumentation + Polish (15–20h)
**Ziel:** Repo ist vorzeigbar. Jeder Commit-Message sitzt, README erzählt die Story.

| Tag | Block | Aufwand | Deliverable |
|-----|-------|---------|-------------|
| Mo | Docker | 3h | Multi-stage Build, docker-compose für lokale Entwicklung |
| Di | CI/CD | 2–3h | GitHub Actions: Lint (ruff), Tests, Docker Build |
| Mi–Do | Dokumentation | 4–5h | README (Business Case, Architektur, Ergebnisse), ADRs, Bounded Contexts |
| Fr | SHAP + Explainability | 2–3h | Feature Importance Plots, Top-Feature-Narrative |
| Sa | Code Review + Cleanup | 3–4h | Docstrings, Type Hints, tote Imports, Konsistenz |
| So | Final Check | 1–2h | Frischer Clone → pip install → train → predict funktioniert? |

**Risiko Woche 3:** Doku-Procrastination. Doku ab Mi priorisieren, nicht als letztes.
**Checkpoint:** Frischer `git clone` → alles funktioniert → Repo ist vorzeigbar.

---

## Entscheidungspunkte (vorab klären)

### 1. Target Variable
Das Dataset hat mehrere mögliche Targets: `total load actual`, Generation pro Typ, Preise.
**Empfehlung:** `total load actual` (Gesamtverbrauch). Klar, gut erklärbar, genug Komplexität.

### 2. Validation Strategy
- **Option A:** Fester chronologischer Split (z.B. letztes Jahr = Test)
- **Option B:** TimeSeriesSplit mit expanding window (mehrere Folds)
- **Empfehlung:** Beides. Fester Split für finale Metriken (ehrlich, kein Refit). TimeSeriesSplit für Feature Selection / Hyperparameter Tuning. Im ADR dokumentieren warum.

### 3. Log-Transform
Beim paiqo-Projekt war Log-Transform ein zweischneidiges Schwert (Log-RMSE ≠ RMSE Ranking).
**Empfehlung:** Testen, aber beide Metriken reporten. Entscheidung auf Original-Skala treffen.

### 4. Vorhersage-Horizont
- **Option A:** Nächste 24h (realistischer Use Case, Lag-Features nutzbar)
- **Option B:** Nächste 7 Tage
- **Option C:** Nächstes Jahr (wie bei paiqo — aber Short-Term Lags fallen weg)
- **Empfehlung:** 24h ahead. Realistischer Use Case, erlaubt Short-Term Lags, differenziert sich vom paiqo-Projekt.

---

## Definition of Done (wann ist das Projekt "fertig"?)

- [ ] `git clone` → `pip install -e .` → kein Fehler
- [ ] `python -m energy_forecast.train` → MLflow Run mit Metriken
- [ ] `docker compose up` → API erreichbar auf localhost
- [ ] `pytest` → alle Tests grün, Coverage >80% auf kritischen Modulen
- [ ] README erzählt die komplette Story (Problem → Approach → Results → How to Run)
- [ ] Mindestens 3 ADRs dokumentiert
- [ ] Kein toter Code, keine TODO-Kommentare ohne zugehöriges Issue
- [ ] Ein Reviewer ohne Kontext kann das Projekt in 10 Minuten verstehen und lokal starten