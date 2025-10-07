from enum import StrEnum
import time
import uuid
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Request, Response
from pydantic import BaseModel, model_validator


class PredictionJobCreateRequest(BaseModel):
    ### aqui agregamos lo q haya q recibir
    data: str


import random


class PredictionResult(BaseModel):
    price_changes: list[float]


class PredictionJobStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class PredictionJob(BaseModel):
    id: uuid.UUID
    status: PredictionJobStatus
    result: PredictionResult | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_result(self):
        if self.status == PredictionJobStatus.PENDING:
            if self.result is not None or self.error is not None:
                raise ValueError("result and error must be None if status is PENDING")
        elif self.status == PredictionJobStatus.COMPLETED:
            if self.result is None:
                raise ValueError("result must not be None if status is COMPLETED")
            if self.error is not None:
                raise ValueError("error must be None if status is COMPLETED")
        elif self.status == PredictionJobStatus.FAILED:
            if self.result is None:
                raise ValueError("result must be None if status is FAILED")
            if self.error is None:
                raise ValueError("error must not be None if status is FAILED")
        return self


app = FastAPI()


# aqui se guardan los resultados de las predicciones... ojo q estan en memoria nomas, si se reinicia el proceso se pierden, y nunca las borramos asiq se podrian acumular
_predictions: dict[uuid.UUID, PredictionJob] = {}


def compute_prediction(prediction_id: uuid.UUID, data: PredictionJobCreateRequest):
    try:
        ### Aqui hacemos la pega... usando data
        time.sleep(10)  # Reduced sleep for faster testing
        ### y usamos el resultado
        # The client expects a list of 5 floats, corresponding to the 5 active contracts.
        prediction_result = PredictionResult(
            price_changes=[random.uniform(-5, 15) for _ in range(5)]
        )
        ### y lo guardamos en el dict pa q despues se pueda consultar el resultado
        _predictions[prediction_id] = PredictionJob(
            id=prediction_id,
            status=PredictionJobStatus.COMPLETED,
            result=prediction_result,
        )
    except Exception as e:
        _predictions[prediction_id] = PredictionJob(
            id=prediction_id,
            status=PredictionJobStatus.FAILED,
            error=str(e),
        )


@app.post("/predictions", status_code=status.HTTP_202_ACCEPTED)
def predict(
    data: PredictionJobCreateRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    response: Response,
) -> PredictionJob:
    new_prediction_id = uuid.uuid4()
    job = PredictionJob(
        id=new_prediction_id,
        status=PredictionJobStatus.PENDING,
    )
    _predictions[new_prediction_id] = job
    background_tasks.add_task(compute_prediction, new_prediction_id, data)
    response.status_code = status.HTTP_201_CREATED
    response.headers["Location"] = str(
        request.url_for("get_prediction", prediction_id=new_prediction_id)
    )
    return job


@app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: uuid.UUID) -> PredictionJob:
    if prediction_id not in _predictions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found"
        )
    return _predictions[prediction_id]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
