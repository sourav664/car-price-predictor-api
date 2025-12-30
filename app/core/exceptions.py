from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse



def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(content={"detail": str(exc)}, status_code=500)