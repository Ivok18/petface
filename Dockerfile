FROM python:3.10.6
COPY src src
COPY requirements_prod.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn src.main:app --host 0.0.0.0 --port $PORT