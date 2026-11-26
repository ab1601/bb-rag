# bb-rag: Chat-Oberfläche + Agent. Texte und Index werden NICHT ins Image kopiert, sondern als Volumes eingebunden.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    BB_EMBED_DEVICE=cpu \
    BB_CORPUS=/data/research-corpus.json \
    BB_STORAGE=/app/storage \
    HF_HOME=/models \
    BB_MODEL_CACHE=/models

WORKDIR /app
COPY requirements.txt .
# CPU-Version von PyTorch (deutlich kleiner als die CUDA-Variante)
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY bb_rag ./bb_rag
COPY app.py .

RUN useradd --create-home bb && mkdir -p /app/storage /models && chown -R bb /app /models
USER bb

EXPOSE 7860
CMD ["python", "app.py"]
