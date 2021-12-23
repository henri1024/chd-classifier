FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY . /app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

WORKDIR /app

EXPOSE 9999

CMD ["python", "main.py"]