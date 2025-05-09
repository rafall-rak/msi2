from fastapi import FastAPI

from model import HouseInfo, PredictedHouseInfo
from nn_clasifier import NNClassifier

app = FastAPI()
classifier = NNClassifier()


@app.post("/housing_prices", response_model=PredictedHouseInfo)
def classify(house_info: HouseInfo):
    return classifier.predict(house_info)
