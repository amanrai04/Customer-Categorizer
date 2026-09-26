# Assignment: extend the customer segmentation project

The project ships with one working implementation of each decision. Each task
below asks you to replace that decision with a different, equally valid one and
get the pipeline running again.

1. **Swap the classifier.** Logistic Regression is used because
   `notebooks/Feature_Selection_and_classification.ipynb` showed it scoring
   highest among the candidates that were compared. Replace it with a different
   algorithm that performs better. Hyperparameter tuning is encouraged, and the
   tuned model is what should end up in the pipeline.

2. **Swap the dimensionality reduction.** The clustering component reduces the
   feature space with PCA. Replace PCA with LDA and make the pipeline run.

3. **Swap the CI provider.** Continuous integration runs on GitHub Actions.
   Move it to CircleCI.

4. **Swap the database.** Customer records are read from MongoDB. Move the
   project to Cassandra.

5. **Swap the cloud.** The application is deployed to Azure. Deploy it to AWS
   instead.

## Where to change things

| Task | File to edit |
|------|--------------|
| Classifier | `config/model.yaml` |
| Dimensionality reduction | `src/components/data_clustering.py` |
| CI | `.github/workflows/ci.yml` |
| Database | `src/data_access/customer_data.py`, `src/configuration/mongo_db_connection.py` |
| Cloud | `.github/workflows/deploy.yml`, `Dockerfile` |

A change is only complete when `pytest -q` passes and a full training run
finishes without the pipeline rejecting the new model.
