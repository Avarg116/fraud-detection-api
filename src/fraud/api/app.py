"""FastAPI app that scores transactions and stores every prediction in Postgres."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fraud.api.db import open_pool, save_prediction
from fraud.api.model import FraudModel
from fraud.api.schemas import Health, Prediction, Transaction

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load model + db pool once at startup
    app.state.model = FraudModel(MODEL_DIR, MODEL_VERSION)
    app.state.pool = open_pool(os.environ["DATABASE_URL"])
    yield
    app.state.pool.close()


app = FastAPI(title="Fraud scoring API", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [{"type": e["type"], "loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request) -> Health:
    model: FraudModel = request.app.state.model
    return Health(status="ok", model_version=model.version, threshold=model.threshold)


@app.post("/predict")
def predict(tx: Transaction, request: Request) -> Prediction:
    model: FraudModel = request.app.state.model
    score, decision = model.predict(tx)

    prediction_id = save_prediction(
        request.app.state.pool, tx, score, decision, model.version, model.threshold
    )
    return Prediction(
        prediction_id=prediction_id,
        fraud_score=score,
        decision=decision,
        model_version=model.version,
    )
