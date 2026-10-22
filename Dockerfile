FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000 SAMPLE_MODE=true
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 appuser
COPY . .
RUN mkdir -p /app/logs /app/.hmem && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["python", "run.py", "--serve"]
