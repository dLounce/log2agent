FROM python:3.12-slim

# system deps: graphviz (process map rendering) + build tools for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    graphviz-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python deps first (layer caching — deps change less than code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the app
COPY pipeline.py app.py template.j2 mockbank_server.py ./
COPY data/ ./data/
COPY cache/ ./cache/

# streamlit config: run headless, accept external connections
EXPOSE 8501
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "app.py"]
