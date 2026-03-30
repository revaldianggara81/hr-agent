#!/usr/bin/env bash
set -eo pipefail   # ❗ jangan pakai -u dulu biar tidak terlalu strict

APP="recruitment_agent.py"
PORT=8502

echo "Starting setup..."

# =============================================================================
# Step 1: Install dependencies
# =============================================================================
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    if ! pip install -r requirements.txt; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
fi

# =============================================================================
# Step 2: Check .env
# =============================================================================
if [ ! -f ".env" ]; then
    echo ""
    echo "WARNING: .env file not found. Please create one before running."
    exit 1
fi

# =============================================================================
# Step 3: Load Ollama URL safely
# =============================================================================
OLLAMA_URL=$(grep '^ollama_base_url=' .env | cut -d '=' -f2-)

if [ -z "${OLLAMA_URL}" ]; then
    OLLAMA_URL="http://localhost:11434"
fi

echo "Using Ollama URL: ${OLLAMA_URL}"

# =============================================================================
# Step 4: Check Ollama
# =============================================================================
if ! curl -sf "${OLLAMA_URL}/api/tags" > /dev/null; then
    echo "WARNING: Ollama not reachable at ${OLLAMA_URL}"
fi

# =============================================================================
# Step 5: Run Streamlit
# =============================================================================
echo "Starting AI Recruitment System on port ${PORT}..."

streamlit run "${APP}" \
    --server.port "${PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false