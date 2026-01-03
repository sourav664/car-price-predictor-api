import json
import redis
from app.core.config import settings


redist_client = redis.Redis.from_url(settings.REDIS_URL)


def get_cached_predictions(key: str):
    value = redist_client.get(key)
    if value is None:
        return None
    return json.loads(value)


def set_cached_predictions(key: str, value: dict, expiry: int = 60 * 60) :
    redist_client.setex(key, expiry, json.dumps(value))   