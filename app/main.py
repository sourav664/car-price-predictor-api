from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from app.api import routes_auth, routes_predict
from app.middleware.logging_middleware import LoggingMiddleware
from app.core.exceptions import register_exception_handlers

app = FastAPI()

# link middleware
app.add_middleware(LoggingMiddleware)

# link routes
app.include_router(routes_auth.router, tags=["Authentication"])
app.include_router(routes_predict.router, tags=["Predictions"])


# monitoring using Prometheus
Instrumentator().instrument(app).expose(app)

# exception handling
register_exception_handlers(app)