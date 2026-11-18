FROM python:3.13-slim
WORKDIR /app
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY data_pipeline.py decoder.py run_server.py ./
COPY backend ./backend
COPY artifacts/model ./artifacts/model
COPY artifacts/experiments/model_comparison.json artifacts/experiments/tuning.json artifacts/experiments/sequence_summary.json ./artifacts/experiments/
ENV PATHSNAP_DATA_DIR=/data PATHSNAP_TRUTH_DIR=/truth OMP_NUM_THREADS=4
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"
CMD ["python", "run_server.py"]
