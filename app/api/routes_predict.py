from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.core.dependencies import get_current_user, get_api_key
from app.services.model_service import predict_car_price


router = APIRouter()

class Carfeatures(BaseModel):
    company: str
    name: str
    year: int
    owner: str
    fuel: str
    seller_type: str
    transmission: str
    km_driven: int
    mileage_mpg: float
    engine_cc: float
    max_power_bhp: float
    torque_nm: float
    seats: int
    
    
@router.post("/predict")
def predict_price(car: Carfeatures, user: str = Depends(get_current_user), _=Depends(get_api_key)):
    prediction = predict_car_price(car.model_dump())
    return {"prediction_price": prediction}