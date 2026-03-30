# AI Recruitment System

An AI-powered recruitment assistant that automates CV screening, candidate selection, interview scheduling, and email notifications — all running on-premise using Ollama (Qwen).

## Overview

This system helps HR teams process multiple job applications at once. Upload candidate CVs in PDF format, define the job requirements, and the system will:

1. Analyze each CV against the job requirements using a local LLM (Qwen via Ollama)
2. Select or reject candidates based on skill match
3. Send personalized email notifications to each candidate via SMTP
4. Schedule Zoom interviews automatically for selected candidates

No cloud AI dependencies — all inference runs locally through Ollama.

---

## Architecture

```
User (Browser)
    |
    v
Streamlit App (port 8502)
    |
    |-- PDF Parsing (PyPDF2)
    |-- Resume Analysis --> Ollama (Qwen) via REST API
    |-- Email Drafting  --> Ollama (Qwen) via REST API
    |-- Email Sending   --> Gmail SMTP
    |-- Interview Link  --> Zoom REST API (Server-to-Server OAuth)
```

---

## Requirements

| Component | Description |
|-----------|-------------|
| Python 3.10+ | Runtime |
| Ollama | Local LLM inference server |
| Qwen model | e.g. `qwen2.5:7b` pulled in Ollama |
| Gmail account | For sending emails (App Password required) |
| Zoom account | Server-to-Server OAuth app for meeting creation |

---

## Quick Start (New Environment)

The `start.bash` script handles everything: clone, install, and run.

```bash
# 1. Download the start script
curl -O https://raw.githubusercontent.com/revaldianggara81/hr-agent/main/start.bash
chmod +x start.bash

# 2. Create .env file with your credentials (see Configuration section)
nano .env

# 3. Run
bash start.bash
```

The app will be available at: `http://<your-host>:8502`

---

## Manual Setup

```bash
git clone https://github.com/revaldianggara81/hr-agent
cd hr-agent

pip install -r requirements.txt

cp .env.example .env   # or create .env manually
nano .env

streamlit run v2_app_cv_matching_rekrutment.py --server.port 8502
```

---

## Configuration

Create a `.env` file in the project root:

```env
# Ollama (On-Premise LLM)
ollama_base_url=http://localhost:11434
ollama_model=qwen2.5:7b

# Zoom (Server-to-Server OAuth)
zoom_account_id=your_zoom_account_id
zoom_client_id=your_zoom_client_id
zoom_client_secret=your_zoom_client_secret

# Email (Gmail SMTP)
email_sender=your@gmail.com
email_app_password=xxxx xxxx xxxx xxxx
company_name=Your Company Name
```

### Getting a Gmail App Password

1. Go to [myaccount.google.com/security](https://myaccount.google.com/security)
2. Enable 2-Step Verification
3. Search for "App Passwords" and generate one for "Mail"
4. Use the 16-character password in `email_app_password`

### Getting Zoom Server-to-Server OAuth Credentials

1. Go to [marketplace.zoom.us](https://marketplace.zoom.us) and create an app
2. Choose **Server-to-Server OAuth**
3. Add scope: `meeting:write:admin`
4. Copy Account ID, Client ID, and Client Secret into `.env`

---

## Running with Docker (Ollama)

If Ollama is running as a Docker container:

```bash
docker run -d \
  --name claims_ollama \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama:latest

# Pull the Qwen model
docker exec claims_ollama ollama pull qwen2.5:7b
```

When Ollama runs in Docker and exposes port 11434 to the host, keep `ollama_base_url=http://localhost:11434` in `.env`.

If the app itself also runs inside Docker on the same network, change to:

```env
ollama_base_url=http://claims_ollama:11434
```

---

## Supported Job Roles

The system comes with predefined requirements for the following roles (all editable in the UI):

- AI Engineer
- Back End Developer
- Front End Developer
- AI Solution Architect
- Data Engineer

---

## How It Works

### CV Analysis

Each uploaded PDF is parsed and its text is sent to the local Qwen model with a structured prompt. The model returns a JSON response containing:

```json
{
  "selected": true,
  "feedback": "Strong Python and ML background...",
  "matching_skills": ["Python", "PyTorch"],
  "missing_skills": ["Kubernetes"],
  "experience_level": "mid"
}
```

Selection threshold: at least 70% skill match.

### Email Flow

| Outcome | Email Sent |
|---------|-----------|
| Selected | Congratulations email + interview details |
| Rejected | Feedback email with upskilling suggestions |
| Selected (interview) | Zoom link + date/time confirmation |

Email content is drafted by the local Qwen model, then sent via Gmail SMTP.

### Interview Scheduling

Zoom meetings are created via the REST API (deterministic — guaranteed link). Interviews are scheduled for the next business day at 11:00 AM Jakarta Time (UTC+7), 60 minutes duration.

---

## Project Structure

```
hr-agent/
├── recruitment_agent.py   # Main Streamlit application
├── requirements.txt                   # Python dependencies
├── start.bash                         # One-command setup and launch script
├── .env                               # Environment variables (not committed)
└── README.md
```

---

## Troubleshooting

**Ollama not reachable**
```
ERROR: Cannot connect to Ollama at http://localhost:11434
```
Run `ollama serve` or check that the Docker container is up: `docker ps | grep ollama`

**Model not found**
```bash
ollama pull qwen2.5:7b
```

**Email not sending**
- Confirm `email_app_password` is a Gmail App Password (not your account password)
- Make sure 2-Step Verification is enabled on the Gmail account

**Zoom link shows "not available"**
- Check that your Zoom OAuth app has the `meeting:write:admin` scope
- Verify all three Zoom credentials in `.env`

**CV email not auto-detected**
- Manually enter the candidate email in the text area provided in the UI
