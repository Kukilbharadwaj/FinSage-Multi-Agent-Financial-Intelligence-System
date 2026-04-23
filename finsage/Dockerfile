FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    API_HOST=127.0.0.1 \
    API_PORT=8000 \
    MCP_HOST=127.0.0.1 \
    MCP_PORT=8001

WORKDIR /app

# System libs needed by scientific stack (numpy/pandas/faiss/torch deps).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Install dependencies.
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

COPY . /app

EXPOSE 7860

# app.py bootstraps MCP server + FastAPI backend and then renders Streamlit UI.
CMD sh -c "streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT:-7860}"
