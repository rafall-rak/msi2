from enum import Enum
from pydantic import BaseModel


class FurnishingStatus(str, Enum):
    furnished = "furnished"
    semi_furnished = "semi-furnished"
    unfurnished = "unfurnished"

class HouseInfo(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: bool
    guestroom: bool
    basement: bool
    hotwaterheating: bool
    airconditioning: bool
    parking: int
    prefarea: bool
    furnishing_status: FurnishingStatus

class PredictedHouseInfo(HouseInfo):
    predicted_price: float
