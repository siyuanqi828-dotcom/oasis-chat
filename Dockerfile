FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi uvicorn httpx pydantic jinja2 python-multipart
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
COPY templates/ templates/
COPY static/ static/
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
