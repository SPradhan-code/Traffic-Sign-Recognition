FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY apps ./apps
COPY configs ./configs
COPY models ./models
RUN pip install --no-cache-dir -e .
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "apps.api:app", "--host", "0.0.0.0", "--port", "8000"]
