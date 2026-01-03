FROM python:3.12

WORKDIR /app


COPY ./app .


RUN pip install --no-cache-dir --upgrade pip &&  \
    pip install --no-cache-dir-r requirements.txt


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "8000"]