
from fastapi import FastAPI,HTTPException
import uvicorn
import sys
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from Hate_Speech_Classification.pipeline.Predict import PredictionPipeline
from Hate_Speech_Classification.Exception import CustomException
from Hate_Speech_Classification.constrants import *


import os




app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")




@app.get("/train", tags=["training"])
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")



@app.post("/predict", tags=["prediction"])
async def predict_route(words:str):
    try:

        
        predictor = PredictionPipeline()
        prediction = predictor.run_pipeline(words)
        return {"prediction": prediction}
    except Exception as e:
        
        raise CustomException(e, sys) from e




if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)