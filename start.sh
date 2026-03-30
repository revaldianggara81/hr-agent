#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/revaldianggara81/hr-agent"
REPO_DIR="hr-agent"
APP="recruitment_agent.py"
PORT=8502

# =============================================================================
# Step 1: Clone or update repo
# =============================================================================
if [ ! -d "${REPO_DIR}" ]; then
    echo "Cloning repository from ${REPO_URL}..."
    git clone "${REPO_URL}" "${REPO_DIR}"
else
    echo "Repository already exists. Pulling latest changes..."
    git -C "${REPO_DIR}" pull
fi

cd "${REPO_DIR}"

# =============================================================================
# Step 2: Install dependencies
# =============================================================================
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt --quiet
fi

# =============================================================================
# Step 3: Check .env
# =============================================================================
if [ ! -f ".env" ]; then
    echo ""
    echo "WARNING: .env file not found. Please create one before running."
    echo "Required variables:"
    echo "  ollama_base_url=http://localhost:11434"
    echo "  ollama_model=qwen2.5:7b"
    echo "  zoom_account_id=..."
    echo "  zoom_client_id=..."
    echo "  zoom_client_secret=..."
    echo "  email_sender=..."
    echo "  email_app_password=..."
    echo "  company_name=..."
    echo ""
    exit 1
fi

# =============================================================================
# Step 4: Check Ollama
# =============================================================================
OLLAMA_URL=$(grep -E '^ollama_base_url=' .env | cut -d '=' -f2- | tr -d ' ')
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

if ! curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    echo ""
    echo "WARNING: Ollama does not appear to be running at ${OLLAMA_URL}"
    echo "Start it with: ollama serve"
    echo ""
fi

# =============================================================================
# Step 5: Run the app
# =============================================================================
echo ""
echo "Starting AI Recruitment System on port ${PORT}..."

streamlit run "${APP}" \
    --server.port "${PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
