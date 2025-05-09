import logging
import pickle

import numpy as np
import torch

from model import HouseInfo
from model import PredictedHouseInfo, FurnishingStatus


def map_boolean(boolean_value) -> float:
    if boolean_value:
        return float(1.0)
    else:
        return float(-1.0)


def map_furnishing_status(furnishing_status) -> float:
    if furnishing_status == FurnishingStatus.furnished:
        return float(1.0)
    elif furnishing_status == FurnishingStatus.semi_furnished:
        return float(0.0)
    else:
        return float(-1.0)


class NNClassifier:
    def __init__(self):
        self.model = torch.load('./model/model.pt').to('cpu')
        with open('./model/std_scaler.pickle', 'rb') as file:
            self.std_scaler = pickle.load(file)

        self.logger = logging.getLogger('uvicorn.error')
        self.logger.setLevel(logging.DEBUG)

    def predict(self, house_info: HouseInfo):
        tensor = self.prepare_data(house_info)
        predicted_price_tensor = self.model(tensor)

        predicted_price = predicted_price_tensor.item()

        return PredictedHouseInfo(
            area=house_info.area,
            bedrooms=house_info.bedrooms,
            bathrooms=house_info.bathrooms,
            stories=house_info.stories,
            mainroad=house_info.mainroad,
            guestroom=house_info.guestroom,
            basement=house_info.basement,
            hotwaterheating=house_info.hotwaterheating,
            airconditioning=house_info.airconditioning,
            parking=house_info.parking,
            prefarea=house_info.prefarea,
            furnishing_status=house_info.furnishing_status,
            predicted_price=predicted_price,  # Placeholder for predicted price
        )

    def prepare_data(self, house_info) -> torch.Tensor:
        # Prepare data in the same way as was done during training
        array = np.array([
            house_info.area,
            float(house_info.bedrooms),
            float(house_info.bathrooms),
            float(house_info.stories),
            map_boolean(house_info.mainroad),
            map_boolean(house_info.guestroom),
            map_boolean(house_info.basement),
            map_boolean(house_info.hotwaterheating),
            map_boolean(house_info.airconditioning),
            float(house_info.parking),
            map_boolean(house_info.prefarea),
            map_furnishing_status(house_info.furnishing_status),
        ], dtype=np.float32).reshape(1, -1)

        self.logger.info(array.dtype)
        self.logger.info(array.shape)

        scaled_array = self.std_scaler.transform(array)
        return torch.tensor(scaled_array, device='cpu')
