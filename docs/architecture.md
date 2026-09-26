# Architecture

The project is a two-pipeline ML service: a **training pipeline** that runs on
demand and promotes a model to object storage, and a **prediction pipeline**
that serves the promoted model.

Stage-by-stage diagrams are below. Every file and directory name matches what
the pipeline actually writes under
`customer_segmentation/artifact/<MM_DD_YYYY_HH_MM_SS>/`.

## End to end

```mermaid
flowchart TD
    A[GET /train] --> B[Data ingestion]
    B --> C[Data validation]
    C --> D[Data transformation]
    D --> E[Data clustering]
    E --> F[Model trainer]
    F --> G[Model evaluation]
    G -->|accepted| H[Model pusher]
    G -->|rejected| I[log and stop]
    H --> J[(S3: model.pkl)]
    J --> K[GET / - prediction form]
    K --> L[Prediction pipeline]
    L --> J
```

## 1. Data ingestion

`src/components/data_ingestion.py`

```mermaid
flowchart LR
    A[(MongoDB<br/>customers collection)] -->|CustomerData| B[DataFrame]
    B --> C[Drop ID, Z_CostContact,<br/>Z_Revenue]
    C --> D[feature_store/customer.csv]
    C --> E[Split 80/20<br/>random_state=42]
    E --> F[ingested/train.csv]
    E --> G[ingested/test.csv]
```

Reads the collection through `CustomerData`, drops the three columns listed in
`config/schema.yaml` under `drop_columns`, and writes a deterministic split. The
unsplit frame is kept as `feature_store/customer.csv` so a run can be inspected
after the fact.

## 2. Data validation

`src/components/data_validation.py`

```mermaid
flowchart TD
    A[train.csv] --> C{Schema matches<br/>config/schema.yaml?}
    B[test.csv] --> C
    C -->|no| F[raise CustomerException<br/>run stops]
    C -->|yes| D[PSI + Kolmogorov-Smirnov<br/>drift check]
    D --> E[drift_report/report.yaml]
    E --> G[DataValidationArtifact]
```

A schema mismatch fails the run. Drift is advisory: it is measured, written to
`report.yaml` and logged as a warning, because a random split of a homogeneous
population is expected to trip the statistical test from time to time.

## 3. Data transformation

`src/components/data_transformation.py`

```mermaid
flowchart LR
    A[train.csv / test.csv] --> B[build_features]
    B --> C[26 raw columns -><br/>21 model features]
    C --> D{ColumnTransformer}
    D -->|8 right-skewed| E[PowerTransformer]
    D -->|remaining numeric| F[StandardScaler]
    E --> G[preprocessing.pkl]
    F --> G
    G --> H[train.npy / test.npy]
```

`build_features()` derives `Age`, `Total_Spending`, `Days_as_Customer`,
`Parental Status` and one column per spend category, then drops the raw columns
it consumed. The fitted `ColumnTransformer` is saved as `preprocessing.pkl` and
travels with the model, so inference needs no separate preprocessing step.

## 4. Data clustering

`src/components/data_clustering.py`

```mermaid
flowchart LR
    A[train.npy] --> B[PCA n_components=2]
    C[test.npy] --> B
    B --> D[KMeans n_clusters=3<br/>random_state=42]
    D --> E[add cluster column]
    E --> F[(train.npy + test.npy<br/>with cluster label)]
```

PCA compresses the 21 features to 2 components for grouping. The resulting
cluster id becomes the supervised target, which is what bridges the
unsupervised and supervised halves of the project.

## 5. Model trainer

`src/components/model_trainer.py`

```mermaid
flowchart TD
    A[train.npy] --> B[ModelFactory]
    B --> C[Build candidates from<br/>config/model.yaml]
    C --> D[GridSearchCV cv=3]
    D --> E[Best estimator]
    E --> F[CustomerSegmentationModel<br/>preprocessor + classifier]
    F --> G[(trained_model/model.pkl)]
    E --> H[calculate_metric]
    H --> I[weighted f1, precision, recall]
    G --> J[ModelTrainerArtifact]
    I --> J
```

The model file is self-contained: it bundles the fitted preprocessor with the
classifier, so a single pickle can score a raw customer record.

## 6. Model evaluation

`src/components/model_evaluation.py`

```mermaid
flowchart TD
    A[test.npy] --> C[Score both models]
    B[Is there a promoted model?] -->|no| D[baseline = 0.0]
    B -->|yes| E[Score promoted model]
    E --> C
    D --> C
    C --> F[changed_accuracy =<br/>trained - baseline]
    F --> G{changed_accuracy > 0.02?}
    G -->|yes| H[ModelEvaluationArtifact<br/>is_model_accepted = True]
    G -->|no| I[ModelEvaluationArtifact<br/>is_model_accepted = False]
```

Both models are scored on the already-transformed test matrix, so the
preprocessor bundled in each pickle is deliberately bypassed - scoring the raw
estimator keeps the comparison honest. The first run has no incumbent, so
anything above zero counts as an improvement.

## 7. Model pusher

`src/components/model_pusher.py`

```mermaid
flowchart LR
    A[accepted ModelTrainerArtifact] --> B[SimpleStorageService]
    B --> C[(S3: model.pkl)]
```

Uploads the model to the bucket configured by `TRAINING_BUCKET_NAME`. A
regression is never written, so the promoted model can only improve.

## Prediction pipeline

`src/pipeline/prediction_pipeline.py`

```mermaid
flowchart TD
    A[Form values] --> B[CustomerInputBuilder<br/>coerces to schema dtypes]
    B --> C[DataFrame in the order of<br/>config/prediction_schema.yaml]
    C --> D[CustomerClusterEstimator]
    D --> E{Model cached<br/>in memory?}
    E -->|no| F[Download model.pkl from S3]
    F --> G[Cache the model]
    E -->|yes| H
    G --> H[preprocessor.transform]
    H --> I[classifier.predict]
    I --> J[Cluster id]
```

The model is downloaded once per process and then cached, so repeated
predictions do not hit S3 every time. The input frame is built in the exact
order of `config/prediction_schema.yaml`, which is the contract the classifier
was trained against.

## Artefact layout

```mermaid
flowchart TD
    A[customer_segmentation/] --> B[artifact/]
    A --> C[logs/]
    B --> D[MM_DD_YYYY_HH_MM_SS/]
    D --> E[data_ingestion/<br/>feature_store/customer.csv<br/>ingested/train.csv<br/>ingested/test.csv]
    D --> F[data_validation/<br/>drift_report/report.yaml]
    D --> G[data_transformation/<br/>transformed/train.npy<br/>transformed/test.npy<br/>transformed_object/preprocessing.pkl]
    D --> H[model_trainer/<br/>trained_model/model.pkl]
    C --> I[customer_segmentation.log]
```

Nothing is written inside `src/`.
