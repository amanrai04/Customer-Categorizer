# Customer Personality Segmentation

A machine learning service that assigns every customer to one of three
behavioural segments. It trains itself on historical customer records, learns
the segments with clustering, and then serves the segment of a new customer
over a small FastAPI web app.

The two-model design is the point of the project: **K-Means** discovers the
segments in an unsupervised way, and a **Logistic Regression** classifier is
trained on those discovered labels so a segment can be predicted for a single
new customer at request time.

## Table of contents

- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [HTTP API](#http-api)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Docker](#docker)
- [Deployment](#deployment)
- [Notebooks](#notebooks)
- [License](#license)

## How it works

Raw records arrive with 26 columns of demographics, purchase behaviour and
campaign response. The training pipeline runs seven stages:

| # | Stage | What it does |
|---|-------|--------------|
| 1 | **Data ingestion** | Reads the customer collection from MongoDB, drops the `ID`, `Z_CostContact` and `Z_Revenue` bookkeeping columns, and writes a deterministic 80/20 train/test split. |
| 2 | **Data validation** | Fails the run if the ingested columns do not match `config/schema.yaml` exactly. Also compares the two splits with a PSI + Kolmogorov-Smirnov drift check and writes a report. |
| 3 | **Data transformation** | Derives 21 model features from the raw columns (customer age, total spending, days as customer, and one column per spend category), standardises the numeric features and applies a `PowerTransformer` to the heavily right-skewed ones. |
| 4 | **Data clustering** | Fits PCA down to 2 components, then K-Means with `k=3`, and uses the resulting labels as the supervised target. |
| 5 | **Model trainer** | Runs a `GridSearchCV` over the search space in `config/model.yaml` and reports weighted F1, precision and recall on the held-out split. |
| 6 | **Model evaluation** | Compares the newly trained model against the currently promoted one. The new model is only accepted if it improves accuracy by more than the configured threshold. |
| 7 | **Model pusher** | Uploads the accepted model to S3 as `model.pkl`. A regression never reaches the bucket. |

Serving is a single step: the prediction pipeline downloads `model.pkl` (once,
then caches it in memory), applies the bundled preprocessor and returns the
cluster id.

## Architecture

```
                       ┌──────────────────────────────┐
   training request    │        FastAPI  app         │
   GET /train  ───────▶│  TrainPipeline.run_pipeline()│
                       └──────────────┬───────────────┘
                                      │
     ┌────────┬───────────┬────────────┼────────────┬───────────┐
     ▼        ▼           ▼            ▼            ▼           ▼
 ingestion  validation transformation clustering  trainer   evaluation
     │        │           │            │            │           │
     │        │           │            │            │           └── accept?
     │        │           │            │            │                    │
     │        │           │            ▼            ▼                    ▼
     │        │           │      ┌──────────┐  best model      ┌──────────┐
     │        │           │      │K-Means 3 │──────────────▶   │  S3      │
     │        │           │      │ + PCA(2) │  (GridSearchCV) │ model.pkl│
     ▼        ▼           ▼      └──────────┘                  └──────────┘
  MongoDB   schema +   preprocessor.pkl
            drift      train.npy / test.npy
```

Stage-by-stage diagrams are in [`docs/architecture.md`](docs/architecture.md).

## Quickstart

### Prerequisites

- Python 3.11
- A MongoDB database holding the customer records (local or Atlas)
- An AWS S3 bucket for the promoted model

### 1. Install

```bash
git clone https://github.com/amanrai04/Customer-Categorizer.git
cd Customer-Categorizer

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
export MONGO_DB_URL="mongodb+srv://<user>:<password>@<cluster>.mongodb.net"
export AWS_ACCESS_KEY_ID="<your-key-id>"
export AWS_SECRET_ACCESS_KEY="<your-secret>"
export AWS_DEFAULT_REGION="ap-south-1"
```

Windows PowerShell uses `$env:MONGO_DB_URL = "..."` instead of `export`. See
[Configuration](#configuration) for the full list of variables.

### 3. Load the data

The pipeline reads a MongoDB collection. Upload the customer records as a CSV
with [MongoDB Compass](https://www.mongodb.com/products/compass), or with the
`mongoimport` tool:

```bash
mongoimport --uri "$MONGO_DB_URL" \
            --collection customers \
            --type csv \
            --headerline \
            --file customers.csv
```

The columns must match `config/schema.yaml`.

### 4. Run the app

```bash
python app.py
```

Then open <http://localhost:5000>, submit the form, and check
<http://localhost:5000/health>.

### 5. Train the model

```bash
curl http://localhost:5000/train
```

The first run has no promoted model, so it is accepted and pushed to S3. Later
runs only replace it if the new model is genuinely better.

## Configuration

Every setting is an environment variable; nothing sensitive is committed.

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `MONGO_DB_URL` | yes | - | MongoDB connection string used by ingestion |
| `AWS_ACCESS_KEY_ID` | yes | - | S3 access key |
| `AWS_SECRET_ACCESS_KEY` | yes | - | S3 secret key |
| `AWS_DEFAULT_REGION` | no | `ap-south-1` | Region for the S3 client |
| `TRAINING_BUCKET_NAME` | no | `customer-segmentation-bucket` | Bucket holding `model.pkl` |
| `APP_HOST` | no | `0.0.0.0` | Interface uvicorn binds to |
| `APP_PORT` | no | `5000` | Port uvicorn listens on |
| `CORS_ORIGINS` | no | `*` | Comma separated allowed origins, e.g. `https://my-app.azurewebsites.net` |

Model behaviour is configured in files rather than the environment:

| File | Controls |
|------|----------|
| `config/schema.yaml` | The expected raw columns, and which ones to drop |
| `config/prediction_schema.yaml` | The 21 model features, their order and their dtypes |
| `config/model.yaml` | The candidate models and the `GridSearchCV` search space |

> `config/prediction_schema.yaml` is an ordered contract. The classifier selects
> its inputs by column name, but the prediction pipeline builds its dataframe in
> the order given there, so reordering the keys changes the inputs the model
> receives.

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Renders the prediction form |
| `POST` | `/` | Scores a submitted customer and re-renders the form with the segment |
| `GET` | `/train` | Runs the full training pipeline |
| `GET` | `/health` | Liveness probe, returns `{"status": "ok"}` |
| `GET` | `/docs` | Interactive OpenAPI documentation |

## Project layout

```
.
├── app.py                     FastAPI application
├── config/                    Schema and model search space
├── docs/                      Architecture, setup and deployment guides
├── notebooks/                 Exploratory analysis, feature engineering, model selection
├── scripts/                   Azure resource helpers
├── src/
│   ├── cloud_storage/         S3 client
│   ├── components/            The seven pipeline stages
│   ├── configuration/         MongoDB client
│   ├── constant/              Paths, file names, environment variable names
│   ├── data_access/           MongoDB access and input coercion
│   ├── entity/                Config and artifact dataclasses
│   ├── exception.py           Single exception type for the whole project
│   ├── logger.py              Shared logger
│   ├── ml/                    Estimators and metrics
│   ├── pipeline/              Training and prediction orchestration
│   └── utils/                 YAML / dataframe / model helpers
├── static/                    Stylesheet
├── templates/                 Jinja2 form template
└── tests/                     Unit and end-to-end tests
```

The output of a training run never lands inside `src/`. Everything is written
to a timestamped folder:

```
customer_segmentation/
├── artifact/
│   └── <MM_DD_YYYY_HH_MM_SS>/
│       ├── data_ingestion/          customer.csv, train.csv, test.csv
│       ├── data_validation/         drift report
│       ├── data_transformation/     train.npy, test.npy, preprocessing.pkl
│       ├── data_clustering/
│       └── model_trainer/           model.pkl
└── logs/
    └── customer_segmentation.log
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite covers feature engineering, schema consistency, the preprocessor, the
clustering configuration, input coercion, the drift statistics, the HTTP routes
and CORS wiring. Two tests are end-to-end: `tests/test_training_pipeline.py` runs
the whole training pipeline and `tests/test_app.py` drives the FastAPI app
through a test client - both with MongoDB and S3 stubbed out, so no cloud
account is needed.

```bash
flake8 app.py src tests     # lint, same rules as CI
```

## Docker

```bash
docker build -t customer-segmentation .
docker run -p 5000:5000 --env-file .env customer-segmentation
```

The image runs as an unprivileged user and exposes a `/health` probe.

## Deployment

Deployment is optional and lives in `.github/workflows/deploy.yml`; it only
runs once the repository secrets are configured.

- [`docs/manual_setup.md`](docs/manual_setup.md) - create the Azure resources by
  hand in the portal
- [`docs/automated_setup.md`](docs/automated_setup.md) - create the same
  resources with the Azure CLI
- [`docs/deployment.md`](docs/deployment.md) - the repository secrets and what
  each one is for

`.github/workflows/ci.yml` runs linting, bytecode compilation, the test suite
and a Docker build on every push and pull request.

## Notebooks

| Notebook | Contents |
|----------|----------|
| `notebooks/EDA.ipynb` | Data loading, profiling and univariate analysis |
| `notebooks/Feature_engineering_and_clustering.ipynb` | Feature derivation, clustering and cluster profiling |
| `notebooks/Feature_Selection_and_classification.ipynb` | Model comparison that led to the Logistic Regression baseline in `config/model.yaml` |

The notebooks are kept for provenance and need a few packages that the
application itself does not, so they are not in `requirements.txt`:

```bash
pip install matplotlib seaborn plotly statsmodels xgboost catboost kneed
```

`Feature_Selection_and_classification.ipynb` compares eight classifiers; Logistic
Regression won at 0.931 accuracy, ahead of AdaBoost at 0.926 and CatBoost at
0.924, which is why it is the candidate in `config/model.yaml`.

## License

Released under the [MIT License](LICENSE).
