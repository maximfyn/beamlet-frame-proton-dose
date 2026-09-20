from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, status
import uvicorn
import inference
from uvicorn.config import LOGGING_CONFIG

# Global state for our model
MODELS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model during startup so it's ready and time isn't counted against invoke
    MODELS["predictor"] = inference.init_model()
    yield
    MODELS.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    if "predictor" in MODELS and MODELS["predictor"] is not None:
        return Response(status_code=status.HTTP_200_OK)
    return Response(status_code=status.HTTP_404_NOT_FOUND)

@app.post("/invoke")
async def invoke():
    predictor = MODELS["predictor"]
    # Run the inference on the mounted data
    inference.run(predictor)
    return Response(status_code=status.HTTP_201_CREATED)

if __name__ == "__main__":
    log_config = LOGGING_CONFIG.copy()
    log_config["handlers"]["default"]["stream"] = "ext://sys.stdout"
    uvicorn.run(app, host="0.0.0.0", port=4743, log_config=log_config)
