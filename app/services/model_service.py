import joblib
import pandas as pd 
from app.core.config import settings
from app.cache.redis_cache import get_cached_predictions, set_cached_predictions


model = joblib.load(settings.MODEL_PATH)


def predict_car_price(data: dict):
    cache_key = " ".join([str(val) for val in data.values()])
    cached_predictions = get_cached_predictions(cache_key)
    if cached_predictions is not None:
        return cached_predictions
    df = pd.DataFrame(data)
    prediction = model.predict(df)[0]
    set_cached_predictions(cache_key, {"prediction": prediction})
    return {"prediction": prediction}
