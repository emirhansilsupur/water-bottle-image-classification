# syntax=docker/dockerfile:1


ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION} as base

WORKDIR /demos

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"] 
