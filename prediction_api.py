"""A FastAPI application for handling asynchronous prediction jobs.

This API provides endpoints to create and check the status of prediction jobs.
When a new job is created, it is added to a background queue for processing.
The status and result of the job can be retrieved later using its unique ID.

Note:
    The prediction jobs are stored in memory (`_predictions` dictionary).
    This means that job data will be lost if the application is restarted.
    This implementation is for demonstration purposes and is not suitable for
    a production environment without a persistent job store.
"""
from enum import StrEnum
import time
import uuid
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Request, Response
from pydantic import BaseModel, model_validator


class PredictionJobCreateRequest(BaseModel):
    """Defines the request model for creating a new prediction job.

    Attributes:
        data (str): The input data for the prediction, expected as a
            CSV-formatted string.
    """
    data: str


class PredictionResult(BaseModel):
    """Defines the structure for the result of a completed prediction.

    Attributes:
        predicted_price (float): The predicted price value from the model.
    """
    predicted_price: float


class PredictionJobStatus(StrEnum):
    """Enumeration for the possible statuses of a prediction job."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class PredictionJob(BaseModel):
    """Represents a prediction job, including its status and result.

    Attributes:
        id (uuid.UUID): The unique identifier for the job.
        status (PredictionJobStatus): The current status of the job.
        result (PredictionResult | None): The result of the prediction. This is
            only present if the status is 'completed'.
        error (str | None): A description of the error. This is only present
            if the status is 'failed'.
    """
    id: uuid.UUID
    status: PredictionJobStatus
    result: PredictionResult | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_result(self):
        """Validates the model state based on the job status."""
        if self.status == PredictionJobStatus.PENDING:
            if self.result is not None or self.error is not None:
                raise ValueError("result and error must be None if status is PENDING")
        elif self.status == PredictionJobStatus.COMPLETED:
            if self.result is None:
                raise ValueError("result must not be None if status is COMPLETED")
            if self.error is not None:
                raise ValueError("error must be None if status is COMPLETED")
        elif self.status == PredictionJobStatus.FAILED:
            # Note: The original had a validation error here. A failed job
            # should have an error, not a result. Correcting the logic.
            if self.result is not None:
                raise ValueError("result must be None if status is FAILED")
            if self.error is None:
                raise ValueError("error must not be None if status is FAILED")
        return self


app = FastAPI()

# In-memory dictionary to store prediction job states.
# Warning: This is not persistent. Data will be lost on restart.
_predictions: dict[uuid.UUID, PredictionJob] = {}


def compute_prediction(prediction_id: uuid.UUID, data: PredictionJobCreateRequest):
    """
    Simulates the actual prediction computation as a background task.

    This function is intended to be run in the background. It simulates a
    long-running process (e.g., a machine learning model inference), and
    upon completion, it updates the job's status in the central `_predictions`
    dictionary to either 'completed' or 'failed'.

    Args:
        prediction_id (uuid.UUID): The ID of the job to process.
        data (PredictionJobCreateRequest): The input data for the prediction.
    """
    try:
        # Simulate a 30-second computation
        time.sleep(30)
        # The result is hardcoded for this simulation
        prediction_result = PredictionResult(predicted_price=100.0)
        # Store the successful result
        _predictions[prediction_id] = PredictionJob(
            id=prediction_id,
            status=PredictionJobStatus.COMPLETED,
            result=prediction_result,
        )
    except Exception as e:
        # Store the failure information
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
    """Creates and queues a new prediction job.

    This endpoint accepts prediction data, creates a new job with a 'pending'
    status, and schedules the `compute_prediction` function to run in the
    background.

    Args:
        data (PredictionJobCreateRequest): The request body containing the data.
        background_tasks (BackgroundTasks): FastAPI dependency to manage
            background tasks.
        request (Request): The incoming request object, used to build the
            location URL.
        response (Response): The outgoing response object, used to set headers.

    Returns:
        The newly created prediction job with a 'pending' status. The response
        status code is set to 201 (Created) and includes a 'Location' header
        pointing to the URL for retrieving the job status.
    """
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
    """Retrieves the status and result of a specific prediction job.

    Args:
        prediction_id (uuid.UUID): The ID of the prediction job to retrieve.

    Returns:
        The PredictionJob object corresponding to the given ID.

    Raises:
        HTTPException: If no prediction job with the given ID is found (404).
    """
    if prediction_id not in _predictions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found"
        )
    return _predictions[prediction_id]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)