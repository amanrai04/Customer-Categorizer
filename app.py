"""FastAPI application exposing the customer segmentation model.

Two endpoints:

* ``GET  /``       renders the prediction form
* ``POST /``       scores the submitted customer and re-renders the form
* ``GET  /train``  triggers a full retraining run

The form fields are derived from ``config/prediction_schema.yaml`` rather than
being listed by hand, so adding a feature to the schema is enough for it to
appear in the UI and reach the model in the right order.
"""

import sys
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as run_app

from src.constant.application import APP_HOST, APP_PORT, get_cors_origins
from src.entity.config_entity import PredictionSchemaConfig
from src.logger import logging
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import TrainPipeline

app = FastAPI(
    title="Customer Segmentation API",
    description=(
        "Predicts which customer segment a customer belongs to by combining "
        "K-Means clustering with a supervised classifier."
    ),
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

#: Allowed CORS origins, read from the CORS_ORIGINS environment variable.
cors_origins = get_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    # Credentials are only meaningful for a specific origin, so they are enabled
    # only when the CORS origins are not the catch-all "*".
    allow_credentials=cors_origins != ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

prediction_schema_config = PredictionSchemaConfig()

#: Features that are entered as a choice in the form. Anything else becomes a
#: number input.
SELECT_FIELDS = {
    "Education": {
        "label": "Education of customer",
        "options": [(0, "Basic"), (1, "2n Cycle"), (2, "Graduation"), (3, "Master"), (4, "PhD")],
    },
    "Marital Status": {
        "label": "Living with a partner",
        "options": [(0, "No"), (1, "Yes")],
    },
    "Parental Status": {
        "label": "Has children at home",
        "options": [(0, "No"), (1, "Yes")],
    },
}

#: Help text shown in each number input.
FIELD_HINTS = {
    "Age": "Customer age",
    "Children": "Number of children at home",
    "Income": "Yearly income of the household",
    "Total_Spending": "Total amount spent in the last two years",
    "Days_as_Customer": "Number of days since the first purchase",
    "Recency": "Days since the last purchase",
    "Wines": "Amount spent on wine",
    "Fruits": "Amount spent on fruit",
    "Meat": "Amount spent on meat",
    "Fish": "Amount spent on fish",
    "Sweets": "Amount spent on sweets",
    "Gold": "Amount spent on gold products",
    "Web": "Purchases through the website",
    "Catalog": "Purchases from the catalogue",
    "Store": "Purchases in store",
    "Discount Purchases": "Purchases made with a discount",
    "Total Promo": "Promotional offers accepted",
    "NumWebVisitsMonth": "Website visits in the last month",
}


def build_form_fields() -> List[dict]:
    """Describe every input the prediction form needs.

    Returns:
        One descriptor per schema feature, holding the field name, a human
        readable label and either a set of choices or a numeric input hint.
    """
    fields = []
    for column in prediction_schema_config.column_names():
        select = SELECT_FIELDS.get(column)
        fields.append(
            {
                "name": column,
                "label": select["label"] if select else FIELD_HINTS.get(column, column),
                "is_select": select is not None,
                "options": select["options"] if select else [],
                "hint": "" if select else FIELD_HINTS.get(column, column),
            }
        )
    return fields


async def read_form_values(request: Request) -> List:
    """Read the submitted form into a list ordered as the prediction schema.

    Returns:
        The raw string values, in the order the model expects them.
    """
    form = await request.form()
    return [form.get(column) for column in prediction_schema_config.column_names()]


def render_form(request: Request, cluster: Optional[int] = None) -> Response:
    """Render the prediction form, optionally showing a predicted cluster."""
    return templates.TemplateResponse(
        request=request,
        name="customer.html",
        context={"fields": build_form_fields(), "cluster": cluster},
    )


@app.get("/health")
async def health() -> dict:
    """Liveness probe used by the container platform."""
    return {"status": "ok"}


@app.get("/train")
async def train() -> Response:
    """Trigger a full retraining run.

    Returns:
        A short status message describing the outcome.
    """
    logging.info("Received a training request")
    try:
        TrainPipeline().run_pipeline()
        return JSONResponse({"status": True, "message": "Training completed successfully"})
    except Exception as error:
        logging.error(f"Training failed: {error}", exc_info=True)
        return JSONResponse(
            {"status": False, "message": f"Training failed: {error}"}, status_code=500
        )


@app.get("/")
async def show_form(request: Request) -> Response:
    """Render the empty prediction form."""
    return render_form(request)


@app.post("/")
async def predict(request: Request) -> Response:
    """Score the submitted customer and re-render the form with the result.

    Returns:
        The form, with ``cluster`` set to the predicted segment.
    """
    try:
        input_values = await read_form_values(request)
        predicted_cluster = PredictionPipeline().run_pipeline(input_data=input_values)
        return render_form(request, cluster=int(predicted_cluster[0]))
    except Exception as error:
        logging.error(f"Prediction failed: {error}", exc_info=True)
        return JSONResponse(
            {"status": False, "message": f"Prediction failed: {error}"}, status_code=500
        )


if __name__ == "__main__":
    try:
        logging.info(f"Starting the API server on {APP_HOST}:{APP_PORT}")
        run_app(app, host=APP_HOST, port=APP_PORT)
    except Exception as error:  # pragma: no cover - startup failure path
        print(f"Error starting the application: {error}", file=sys.stderr)
        raise
