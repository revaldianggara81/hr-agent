#!/usr/bin/env bash
set -eo pipefail

APP="recruitment_agent.py"
PORT=8502

echo "========================================"
echo "Starting AI Recruitment System Setup..."
echo "========================================"

# =============================================================================
# Step 1: Install dependencies (only if needed)
# =============================================================================
if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing dependencies..."
    if ! pip install -r requirements.txt; then
        echo "[ERROR] Failed to install dependencies"
        exit 1
    fi
fi

# =============================================================================
# Step 2: Check .env file
# =============================================================================
if [ ! -f ".env" ]; then
    echo ""
    echo "[ERROR] .env file not found!"
    echo "Please create .env before running the app."
    exit 1
fi

# =============================================================================
# Step 3: Load Ollama URL safely
# =============================================================================
OLLAMA_URL=$(grep '^ollama_base_url=' .env | cut -d '=' -f2-)

if [ -z "${OLLAMA_URL}" ]; then
    OLLAMA_URL="http://localhost:11434"
fi

echo "[INFO] Using Ollama URL: ${OLLAMA_URL}"

# =============================================================================
# Step 4: Check Ollama connectivity
# =============================================================================
echo "[INFO] Checking Ollama service..."

if curl -sf "${OLLAMA_URL}/api/tags" > /dev/null; then
    echo "[INFO] Ollama is running"
else
    echo "[WARNING] Ollama is NOT reachable at ${OLLAMA_URL}"
    echo "[WARNING] Start it using: ollama serve"
fi

# =============================================================================
# Step 5: Run Streamlit
# =============================================================================
echo ""
echo "[INFO] Starting Streamlit app on port ${PORT}..."
echo "========================================"

exec streamlit run "${APP}" \
    --server.port "${PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false