from typing import Tuple, Dict, Optional, List
import io
import time
import json
import requests
import PyPDF2
from datetime import datetime, timedelta
import pytz
import pandas as pd
import re

import streamlit as st
<<<<<<< HEAD
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.email import EmailTools
from phi.tools.zoom import ZoomTool
=======
from dotenv import load_dotenv
>>>>>>> addc0a0 (update pipeline)
from phi.utils.log import logger


<<<<<<< HEAD
class CustomZoomTool(ZoomTool):
    def __init__(self, *, account_id: Optional[str] = None, client_id: Optional[str] = None, client_secret: Optional[str] = None, name: str = "zoom_tool"):
        super().__init__(account_id=account_id, client_id=client_id, client_secret=client_secret, name=name)
        self.token_url = "https://zoom.us/oauth/token"
        self.access_token = None
        self.token_expires_at = 0

    def get_access_token(self) -> str:
        if self.access_token and time.time() < self.token_expires_at:
            return str(self.access_token)

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"grant_type": "account_credentials", "account_id": self.account_id}

        try:
            response = requests.post(self.token_url, headers=headers, data=data, auth=(self.client_id, self.client_secret))
=======

# =============================================================================
# Ollama GenAI Helper Class
# =============================================================================
class OllamaChat:
    """Wrapper class for Ollama local LLM inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        max_tokens: int = 4000,
        temperature: float = 0.3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a chat message and return the response text."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
>>>>>>> addc0a0 (update pipeline)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "").strip()

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            logger.error(f"Ollama chat error: {str(e)}")
            raise

    def is_available(self) -> bool:
        """Check if Ollama service is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return list of available model names from Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


<<<<<<< HEAD

def init_session_state() -> None:
    """Initialize session state variables."""
    defaults = {
        'openai_api_key': "", 'zoom_account_id': "", 'zoom_client_id': "",
        'zoom_client_secret': "", 'email_sender': "", 'email_passkey': "",
        'company_name': "", 'custom_role_name': "", 'custom_requirements': "",
        'batch_results': [], 'processing_complete': False
=======
# =============================================================================
# Zoom REST API helpers
# =============================================================================
def get_zoom_access_token_s2s(account_id: str, client_id: str, client_secret: str) -> str:
    """Get Zoom access token (Server-to-Server OAuth)."""
    resp = requests.post(
        "https://zoom.us/oauth/token",
        data={
            "grant_type": "account_credentials",
            "account_id": account_id,
        },
        auth=(client_id, client_secret),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "access_token" not in data:
        raise RuntimeError(f"Zoom token response missing access_token: {data}")
    return str(data["access_token"])


def create_zoom_meeting_rest(
    *,
    access_token: str,
    topic: str,
    start_time_iso: str,
    duration_minutes: int = 60,
    timezone: str = "Asia/Jakarta",
    agenda: str = "",
) -> Dict:
    """Create a scheduled Zoom meeting and return meeting JSON."""
    payload = {
        "topic": topic,
        "type": 2,
        "start_time": start_time_iso,
        "duration": duration_minutes,
        "timezone": timezone,
        "agenda": agenda,
        "settings": {
            "waiting_room": True,
            "join_before_host": False,
            "mute_upon_entry": True,
        },
    }

    resp = requests.post(
        "https://api.zoom.us/v2/users/me/meetings",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# =============================================================================
# Session State
# =============================================================================
def init_session_state() -> None:
    """Initialize session state from .env."""
    defaults = {
        "ollama_base_url": os.getenv("ollama_base_url", "http://localhost:11434"),
        "ollama_model": os.getenv("ollama_model", "qwen2.5:7b"),
        "zoom_account_id": os.getenv("zoom_account_id", ""),
        "zoom_client_id": os.getenv("zoom_client_id", ""),
        "zoom_client_secret": os.getenv("zoom_client_secret", ""),
        "email_sender": os.getenv("email_sender", ""),
        "email_passkey": os.getenv("email_app_password", ""),
        "company_name": os.getenv("company_name", ""),
        "custom_role_name": "AI Engineer",
        "custom_requirements": "",
        "last_selected_role": "",
        "batch_results": [],
        "processing_complete": False,
>>>>>>> addc0a0 (update pipeline)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


<<<<<<< HEAD
def _get_model() -> OpenAIChat:
    """Return a configured OpenAI model instance."""
    return OpenAIChat(id="gpt-4o", api_key=st.session_state.openai_api_key)


def create_resume_analyzer() -> Agent:
    """Creates resume analysis agent with custom requirements."""
    if not st.session_state.openai_api_key:
        st.error("Please enter your OpenAI API key first.")
        return None

    return Agent(
        model=_get_model(),
        description="You are an expert technical recruiter who analyzes resumes.",
        instructions=[
            "Analyze the resume against the provided job requirements",
            "Be lenient with candidates who show strong potential",
            "Consider project experience as valid experience",
            "Value hands-on experience with key technologies",
            "Return a JSON response with selection decision and feedback"
        ],
        markdown=True
    )


def create_email_agent(receiver_email: str) -> Agent:
    """Create email agent for specific receiver."""
    return Agent(
        model=_get_model(),
        tools=[EmailTools(
            receiver_email=receiver_email,
            sender_email=st.session_state.email_sender,
            sender_name=st.session_state.company_name,
            sender_passkey=st.session_state.email_passkey
        )],
        description="You are a professional recruitment coordinator.",
        instructions=[
            "Draft and send professional recruitment emails",
            "Use all lowercase letters for casual, human tone",
            "Maintain friendly yet professional tone",
            "Always end emails with: 'best,\\nthe ai recruiting team'",
            f"Company name: '{st.session_state.company_name}'"
        ],
        markdown=True,
        show_tool_calls=True
    )


def create_scheduler_agent() -> Agent:
    """Create Zoom scheduler agent."""
    zoom_tools = CustomZoomTool(
        account_id=st.session_state.zoom_account_id,
        client_id=st.session_state.zoom_client_id,
        client_secret=st.session_state.zoom_client_secret
    )

    return Agent(
        name="Interview Scheduler",
        model=_get_model(),
        tools=[zoom_tools],
        description="You are an interview scheduling coordinator.",
        instructions=[
            "Schedule interviews during business hours (9 AM - 5 PM Jakarta Time)",
            "Create meetings with proper titles and descriptions",
            "Use ISO 8601 format for dates"
        ],
        markdown=True,
        show_tool_calls=True
=======
# =============================================================================
# Ollama Client Factory
# =============================================================================
def get_ollama_client() -> Optional[OllamaChat]:
    """Create and return Ollama chat client."""
    client = OllamaChat(
        base_url=st.session_state.ollama_base_url,
        model=st.session_state.ollama_model,
>>>>>>> addc0a0 (update pipeline)
    )
    if not client.is_available():
        st.error(
            f"Ollama is not reachable at `{st.session_state.ollama_base_url}`. "
            "Please ensure Ollama is running."
        )
        return None
    return client


<<<<<<< HEAD
@st.cache_data
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes. Cached to avoid re-processing the same file."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
=======
# =============================================================================
# Direct Email Sending via SMTP
# =============================================================================
def send_email_direct(receiver_email: str, subject: str, body: str) -> bool:
    """Send email via SMTP."""
    try:
        msg = MIMEMultipart()
        msg["From"] = f"{st.session_state.company_name} <{st.session_state.email_sender}>"
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(st.session_state.email_sender, st.session_state.email_passkey)
            server.send_message(msg)
        return True
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return False


# =============================================================================
# PDF & Text Extraction
# =============================================================================
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
>>>>>>> addc0a0 (update pipeline)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        return ""


def extract_email_from_text(text: str) -> Optional[str]:
    """Extract email address from resume text."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)

    if emails:
        personal_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'icloud.com']
        for email in emails:
            if any(domain in email.lower() for domain in personal_domains):
                return email
        return emails[0]

    return None


<<<<<<< HEAD
def analyze_resume(resume_text: str, requirements: str, analyzer: Agent) -> Tuple[bool, str, dict]:
    """Analyze single resume against requirements."""
    try:
        response = analyzer.run(
            f"""Analyze this resume against the requirements and respond in valid JSON:
=======
# =============================================================================
# Resume Analysis using Ollama
# =============================================================================
def analyze_resume(
    resume_text: str,
    requirements: str,
    client: OllamaChat,
) -> Tuple[bool, str, dict]:
    """Analyze a single resume against job requirements."""
    try:
        system_prompt = (
            "You are an expert technical recruiter who analyzes resumes. "
            "Analyze the resume against the provided job requirements. "
            "Be lenient with candidates who show strong potential. "
            "Consider project experience as valid experience. "
            "Value hands-on experience with key technologies. "
            "Return a JSON response with selection decision and feedback. "
            "Return ONLY the JSON object without any markdown formatting or code blocks."
        )
>>>>>>> addc0a0 (update pipeline)

            Role Requirements:
            {requirements}

            Resume Text:
            {resume_text}

            Return JSON:
            {{
                "selected": true/false,
                "feedback": "Detailed feedback",
                "matching_skills": ["skill1", "skill2"],
                "missing_skills": ["skill3", "skill4"],
                "experience_level": "junior/mid/senior"
            }}

            Criteria:
            1. Match at least 70% of required skills
            2. Consider practical experience and projects
            3. Value transferable skills
            4. Look for continuous learning

            Return ONLY the JSON object without markdown.
            """
        )

        assistant_message = next((msg.content for msg in response.messages if msg.role == 'assistant'), None)
        if not assistant_message:
            raise ValueError("No assistant message found")

<<<<<<< HEAD
        result = json.loads(assistant_message.strip())
=======
        response_text = client.chat(user_prompt, system_prompt)

        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        result = json.loads(cleaned)
>>>>>>> addc0a0 (update pipeline)
        if not isinstance(result, dict) or not all(k in result for k in ["selected", "feedback"]):
            raise ValueError("Invalid response format")

        return result["selected"], result["feedback"], result

    except Exception as e:
        logger.error(f"Error analyzing resume: {str(e)}")
        return False, f"Error: {str(e)}", {}


<<<<<<< HEAD
def send_selection_email(email_agent: Agent, role: str) -> None:
    """Send selection email to candidate."""
    email_agent.run(
        f"""Send selection email for {role} position.

        Include:
        1. Congratulate on selection
        2. Explain next steps
        3. Mention interview details coming soon
        4. End with: best,\\nthe ai recruiting team
        """
=======
# =============================================================================
# Email Drafting with Ollama
# =============================================================================
def draft_email_with_ollama(
    client: OllamaChat,
    email_type: str,
    role: str,
    feedback: str = "",
    candidate_email: str = "",
    candidate_name: str = "",
    interview_details: str = "",
) -> Tuple[str, str]:
    """Draft email content using Ollama. Returns (subject, body)."""
    greeting_name = (
        candidate_name
        if candidate_name
        else (candidate_email.split("@")[0] if candidate_email else "there")
    )

    if email_type == "selection":
        prompt = f"""Draft a selection email for the {role} position.

Greeting: Start with "hi {greeting_name},"

Style: all lowercase, casual but professional, human tone.

Include:
1. Congratulate on selection
2. Explain next steps
3. Mention interview details coming soon
4. End with: best,\\nthe ai recruiting team

Return ONLY the email body text, no subject line, no quotes."""

    elif email_type == "rejection":
        prompt = f"""Draft a rejection email for the {role} position.

Greeting: Start with "hi {greeting_name},"

Style: all lowercase, empathetic, human tone.

Include:
1. Thank for applying
2. Include this feedback: {feedback}
3. Encourage upskilling
4. Suggest learning resources
5. End with: best,\\nthe ai recruiting team

Return ONLY the email body text, no subject line, no quotes."""

    elif email_type == "interview_confirmation":
        prompt = f"""Draft an interview confirmation email for the {role} position.

Greeting: Start with "hi {greeting_name},"

Style: professional but friendly.

Include these details:
{interview_details}

Notes to include:
- Join 5 minutes early
- Timezone converter: https://www.timeanddate.com/worldclock/converter.html
- Be confident and prepare well!

End with: best,\\nthe ai recruiting team

Return ONLY the email body text, no subject line, no quotes."""
    else:
        prompt = f"Draft a professional email about: {email_type}"

    system_prompt = "You are a professional recruitment coordinator. Draft concise, human-sounding emails."
    body = client.chat(prompt, system_prompt)

    subject_map = {
        "selection": f"Congratulations! You've been selected for {role}",
        "rejection": f"Update on your {role} application",
        "interview_confirmation": f"Interview Confirmation - {role} Position",
    }
    subject = subject_map.get(email_type, f"Update regarding {role}")

    return subject, body.strip()


# =============================================================================
# Send Emails
# =============================================================================
def send_selection_email(
    role: str,
    candidate_email: str,
    client: OllamaChat,
    candidate_name: str = "",
) -> bool:
    """Send selection email to candidate."""
    subject, body = draft_email_with_ollama(
        client,
        "selection",
        role,
        candidate_email=candidate_email,
        candidate_name=candidate_name,
>>>>>>> addc0a0 (update pipeline)
    )


<<<<<<< HEAD
def send_rejection_email(email_agent: Agent, role: str, feedback: str) -> None:
    """Send rejection email with feedback."""
    email_agent.run(
        f"""Send rejection email for {role} position.

        Style:
        1. All lowercase
        2. Empathetic and human
        3. Include feedback: {feedback}
        4. Encourage upskilling
        5. Suggest learning resources
        6. End with: best,\\nthe ai recruiting team
        """
=======
def send_rejection_email(
    role: str,
    feedback: str,
    candidate_email: str,
    client: OllamaChat,
    candidate_name: str = "",
) -> bool:
    """Send rejection email with feedback."""
    subject, body = draft_email_with_ollama(
        client,
        "rejection",
        role,
        feedback=feedback,
        candidate_email=candidate_email,
        candidate_name=candidate_name,
>>>>>>> addc0a0 (update pipeline)
    )


<<<<<<< HEAD
def schedule_interview(scheduler: Agent, candidate_email: str, email_agent: Agent, role: str) -> Tuple[bool, str]:
    """Schedule interview and send confirmation. Returns (success, zoom_link)."""
    try:
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        current_time_jkt = datetime.now(jakarta_tz)
        tomorrow_jkt = current_time_jkt + timedelta(days=1)
        interview_time = tomorrow_jkt.replace(hour=11, minute=0, second=0, microsecond=0)
        formatted_time_iso = interview_time.strftime('%Y-%m-%dT%H:%M:%S')

        meeting_response = scheduler.run(
            f"""Schedule 60-minute technical interview:
            - Title: '{role} Technical Interview'
            - Date: {formatted_time_iso}
            - Timezone: Asia/Jakarta
            - Attendee: {candidate_email}
            - Duration: 60 minutes
            """
        )

        meeting_link, meeting_id = "Zoom link not available", "N/A"
        try:
            raw_response = meeting_response.messages[-1].content.strip()
            if raw_response.startswith("{"):
                meeting_info = json.loads(raw_response)
                meeting_link = meeting_info.get("join_url", meeting_link)
                meeting_id = meeting_info.get("id", meeting_id)
            else:
                if "https://" in raw_response:
                    meeting_link = raw_response.split("https://")[1].split()[0]
                    meeting_link = "https://" + meeting_link
        except Exception as e:
            logger.warning(f"Could not parse meeting response: {e}")

=======
# =============================================================================
# Interview Scheduling
# =============================================================================
def schedule_interview(
    candidate_email: str,
    role: str,
    client: OllamaChat,
    candidate_name: str = "",
) -> Tuple[bool, str]:
    """Schedule interview via Zoom REST API and send confirmation email."""
    try:
        jakarta_tz = pytz.timezone("Asia/Jakarta")
        current_time_jkt = datetime.now(jakarta_tz)
        interview_time = (current_time_jkt + timedelta(days=1)).replace(
            hour=11, minute=0, second=0, microsecond=0
        )

        formatted_time_iso = interview_time.strftime("%Y-%m-%dT%H:%M:%S")
>>>>>>> addc0a0 (update pipeline)
        pretty_date = interview_time.strftime("%A, %d %B %Y")
        pretty_time = interview_time.strftime("%I:%M %p")

<<<<<<< HEAD
        email_agent.run(
            f"""Send interview confirmation to {candidate_email}:

            Subject: Interview Confirmation – {role} Position

            Dear Candidate,

            Technical interview details for {role}:

            📅 Date: {pretty_date}
            🕒 Time: {pretty_time} (Jakarta Time, UTC+7)
            ⏳ Duration: 60 minutes
            🔗 Zoom Link: {meeting_link}

            Notes:
            - Join 5 minutes early
            - Timezone converter: https://www.timeanddate.com/worldclock/converter.html
            - Be confident and prepare well!

            best,
            the ai recruiting team
            """
        )
        return True, meeting_link
=======
        meeting_link = "Zoom link not available"
        meeting_id = "N/A"

        try:
            account_id = st.session_state.zoom_account_id
            client_id = st.session_state.zoom_client_id
            client_secret = st.session_state.zoom_client_secret

            if not account_id or not client_id or not client_secret:
                raise RuntimeError("Missing Zoom credentials in .env.")

            access_token = get_zoom_access_token_s2s(account_id, client_id, client_secret)
            meeting = create_zoom_meeting_rest(
                access_token=access_token,
                topic=f"{role} Technical Interview",
                start_time_iso=formatted_time_iso,
                duration_minutes=60,
                timezone="Asia/Jakarta",
                agenda=f"Technical interview for {role}. Candidate: {candidate_email}",
            )
            meeting_link = meeting.get("join_url", meeting_link)
            meeting_id = meeting.get("id", meeting_id)

        except Exception as zoom_e:
            logger.error(f"Zoom meeting creation failed: {zoom_e}")

        interview_details = (
            f"Date: {pretty_date}\n"
            f"Time: {pretty_time} (Jakarta Time, UTC+7)\n"
            f"Duration: 60 minutes\n"
            f"Zoom Link: {meeting_link}\n"
            f"Meeting ID: {meeting_id}"
        )

        subject, body = draft_email_with_ollama(
            client,
            "interview_confirmation",
            role,
            candidate_email=candidate_email,
            candidate_name=candidate_name,
            interview_details=interview_details,
        )
        send_email_direct(candidate_email, subject, body)

        return True, interview_time_str
>>>>>>> addc0a0 (update pipeline)

    except Exception as e:
        logger.error(f"Error scheduling interview: {str(e)}")
        return False, ""


<<<<<<< HEAD
def process_batch_applications(candidates_data: List[Dict], role_name: str, requirements: str) -> List[Dict]:
    """Process multiple applications at once with robust per-candidate error handling."""
=======
# =============================================================================
# Batch Processing
# =============================================================================
def process_batch_applications(
    candidates_data: List[Dict], role_name: str, requirements: str
) -> List[Dict]:
    """Process multiple applications in sequence."""
>>>>>>> addc0a0 (update pipeline)
    results = []
    processing_errors = []
    analyzer = create_resume_analyzer()

<<<<<<< HEAD
    if not analyzer:
        st.error("Failed to create analyzer")
=======
    client = get_ollama_client()
    if not client:
        st.error("Failed to connect to Ollama.")
>>>>>>> addc0a0 (update pipeline)
        return results

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, candidate in enumerate(candidates_data):
        status_text.text(f"Processing {candidate['email']} ({idx + 1}/{len(candidates_data)})...")

<<<<<<< HEAD
        try:
            is_selected, feedback, analysis_details = analyze_resume(
                candidate['resume_text'],
                requirements,
                analyzer
            )
        except Exception as e:
            err_msg = f"{candidate['email']} ({candidate['filename']}): Resume analysis failed — {str(e)}"
            logger.error(err_msg)
            processing_errors.append(err_msg)
            results.append({
                'email': candidate['email'],
                'filename': candidate['filename'],
                'selected': False,
                'feedback': f'Processing error: {str(e)}',
                'analysis': {},
                'email_sent': False,
                'interview_scheduled': False,
                'zoom_link': '',
                'error': str(e)
            })
            progress_bar.progress((idx + 1) / len(candidates_data))
            continue
=======
        is_selected, feedback, analysis_details = analyze_resume(
            candidate["resume_text"],
            requirements,
            client,
        )
>>>>>>> addc0a0 (update pipeline)

        result = {
            'email': candidate['email'],
            'filename': candidate['filename'],
            'selected': is_selected,
            'feedback': feedback,
            'analysis': analysis_details,
            'email_sent': False,
            'interview_scheduled': False,
            'zoom_link': '',
            'error': None
        }

        try:
<<<<<<< HEAD
            email_agent = create_email_agent(candidate['email'])

            if is_selected:
                send_selection_email(email_agent, role_name)
                result['email_sent'] = True

                scheduler = create_scheduler_agent()
                success, zoom_link = schedule_interview(scheduler, candidate['email'], email_agent, role_name)
                result['interview_scheduled'] = success
                result['zoom_link'] = zoom_link
            else:
                send_rejection_email(email_agent, role_name, feedback)
                result['email_sent'] = True
=======
            if is_selected:
                result["email_sent"] = send_selection_email(
                    role_name,
                    candidate["email"],
                    client,
                    candidate_name=candidate_name,
                )
                scheduled, interview_time = schedule_interview(
                    candidate["email"],
                    role_name,
                    client,
                    candidate_name=candidate_name,
                )
                result["interview_scheduled"] = scheduled
                result["interview_time"] = interview_time
            else:
                result["email_sent"] = send_rejection_email(
                    role_name,
                    feedback,
                    candidate["email"],
                    client,
                    candidate_name=candidate_name,
                )
>>>>>>> addc0a0 (update pipeline)

        except Exception as e:
            err_msg = f"{candidate['email']} ({candidate['filename']}): Email/Zoom failed — {str(e)}"
            logger.error(err_msg)
            processing_errors.append(err_msg)
            result['error'] = str(e)

        results.append(result)
        progress_bar.progress((idx + 1) / len(candidates_data))

<<<<<<< HEAD
    status_text.text("✅ Processing complete!")

    if processing_errors:
        st.warning(f"⚠️ {len(processing_errors)} error(s) occurred during processing:")
        with st.expander("🔍 Error Log", expanded=True):
            for err in processing_errors:
                st.text(f"  • {err}")

    return results


def main() -> None:
    st.set_page_config(
        page_title="AI Recruitment System - Batch Processing",
        layout="wide"
    )

    st.title(" AI Recruitment System - Batch Processing")
    st.markdown("Upload multiple CVs and process them all at once")

    init_session_state()

    # Sidebar Configuration
    with st.sidebar:
        st.header(" Configuration")

        st.subheader("OpenAI Settings")
        api_key = st.text_input("OpenAI API Key", type="password",
                               value=st.session_state.openai_api_key)
        if api_key:
            st.session_state.openai_api_key = api_key

        st.subheader("Zoom Settings")
        zoom_account_id = st.text_input("Zoom Account ID", type="password",
                                       value=st.session_state.zoom_account_id)
        zoom_client_id = st.text_input("Zoom Client ID", type="password",
                                      value=st.session_state.zoom_client_id)
        zoom_client_secret = st.text_input("Zoom Client Secret", type="password",
                                          value=st.session_state.zoom_client_secret)

        st.subheader("Email Settings")
        email_sender = st.text_input("Sender Email", value=st.session_state.email_sender)
        email_passkey = st.text_input("Email App Password", type="password",
                                     value=st.session_state.email_passkey)
        company_name = st.text_input("Company Name", value=st.session_state.company_name)

        if zoom_account_id: st.session_state.zoom_account_id = zoom_account_id
        if zoom_client_id: st.session_state.zoom_client_id = zoom_client_id
        if zoom_client_secret: st.session_state.zoom_client_secret = zoom_client_secret
        if email_sender: st.session_state.email_sender = email_sender
        if email_passkey: st.session_state.email_passkey = email_passkey
        if company_name: st.session_state.company_name = company_name

    # Check required configs
    required_configs = {
        'OpenAI API Key': st.session_state.openai_api_key,
        'Zoom Account ID': st.session_state.zoom_account_id,
        'Zoom Client ID': st.session_state.zoom_client_id,
        'Zoom Client Secret': st.session_state.zoom_client_secret,
        'Email Sender': st.session_state.email_sender,
        'Email Password': st.session_state.email_passkey,
        'Company Name': st.session_state.company_name
=======
    status_text.text("Processing complete.")
    return results


# =============================================================================
# Job Role Definitions
# =============================================================================
ROLE_REQUIREMENTS = {
    "AI Engineer": """Required Skills:
- Python (advanced)
- Machine Learning frameworks (TensorFlow, PyTorch, or scikit-learn)
- Deep Learning (CNN, RNN, Transformer architectures)
- Natural Language Processing (NLP) or Computer Vision
- Data preprocessing & feature engineering (Pandas, NumPy)
- Model training, evaluation, and deployment
- LLM integration (LangChain, or similar)
- Git version control
- 2+ years experience in AI/ML

Nice to have:
- MLOps (MLflow, Kubeflow, or Weights & Biases)
- Cloud AI services (AWS SageMaker, GCP Vertex AI)
- RAG (Retrieval-Augmented Generation) implementation
- Vector databases (FAISS, Pinecone, ChromaDB)
- Docker & Kubernetes for model serving
- Fine-tuning LLMs
- Research paper implementation experience""",

    "Back End Developer": """Required Skills:
- Python (Django, FastAPI, or Flask) or Java (Spring Boot) or Go
- Database design & management (PostgreSQL, MySQL)
- RESTful API design & development
- Authentication & authorization (OAuth2, JWT)
- Docker containerization
- Git version control
- 3+ years experience

Nice to have:
- Microservices architecture
- Message queues (RabbitMQ, Kafka, Redis Streams)
- Caching strategies (Redis, Memcached)
- Cloud infrastructure (AWS, GCP, or Azure)
- Kubernetes orchestration
- CI/CD pipelines (GitHub Actions, Jenkins)
- Database optimization & query tuning
- TDD/BDD practices
- GraphQL""",

    "Front End Developer": """Required Skills:
- HTML5, CSS3, JavaScript (ES6+)
- React.js or Vue.js or Angular
- Responsive design & mobile-first approach
- State management (Redux, Vuex, or Zustand)
- RESTful API integration
- Git version control
- 2+ years experience

Nice to have:
- TypeScript
- Next.js or Nuxt.js
- Tailwind CSS or Styled Components
- Unit testing (Jest, React Testing Library)
- CI/CD pipeline familiarity
- Figma/design tool collaboration
- Web performance optimization (Lighthouse, Core Web Vitals)""",

    "AI Solution Architect": """Required Skills:
- Strong understanding of AI/ML concepts and architectures
- Cloud architecture design (AWS, GCP, or Azure)
- Experience designing end-to-end AI pipelines
- LLM deployment and integration patterns (RAG, fine-tuning, prompt engineering)
- API design and microservices architecture
- Data pipeline design (ETL/ELT)
- Excellent communication and stakeholder management
- 5+ years experience in software engineering, 2+ in AI/ML

Nice to have:
- MLOps and model monitoring experience
- Enterprise architecture frameworks (TOGAF)
- Experience with vector databases and semantic search
- Kubernetes and containerization at scale
- Security and compliance in AI systems (responsible AI)
- Pre-sales or technical consulting experience""",

    "Data Engineer": """Required Skills:
- Python or Scala
- SQL (advanced - complex queries, optimization, window functions)
- ETL/ELT pipeline design and implementation
- Data warehouse concepts (dimensional modeling, star/snowflake schema)
- Big data technologies (Apache Spark, Hadoop, or Databricks)
- Cloud data services (AWS Glue, GCP BigQuery, or Azure Data Factory)
- Git version control
- 3+ years experience

Nice to have:
- Apache Kafka or streaming data platforms
- Orchestration tools (Apache Airflow, Prefect, or Dagster)
- Data quality frameworks (Great Expectations, dbt tests)
- NoSQL databases (MongoDB, Cassandra, DynamoDB)
- Docker & Kubernetes
- Data governance and cataloging
- CI/CD for data pipelines""",
}


# =============================================================================
# Main Streamlit App
# =============================================================================
def main() -> None:
    st.set_page_config(
        page_title="AI Recruitment System",
        layout="wide",
    )

    st.title("AI Agent Recruitment System")
    st.caption("Powered by Ollama (On-Premise)")

    init_session_state()

    # -------------------------------------------------------------------------
    # Sidebar Configuration
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")
        st.subheader("Ollama Settings")

        st.session_state.ollama_base_url = st.text_input(
            "Ollama Base URL",
            value=st.session_state.ollama_base_url,
            help="Default: http://localhost:11434",
        )

        # Fetch available models from Ollama
        available_models: List[str] = []
        client_check = OllamaChat(base_url=st.session_state.ollama_base_url)
        if client_check.is_available():
            available_models = client_check.list_models()
            st.success("Ollama connected")
        else:
            st.warning("Ollama not reachable")

        if available_models:
            current_model = st.session_state.ollama_model
            if current_model not in available_models:
                current_model = available_models[0]
            st.session_state.ollama_model = st.selectbox(
                "Model",
                options=available_models,
                index=available_models.index(current_model),
                help="Select a locally available Ollama model",
            )
        else:
            st.session_state.ollama_model = st.text_input(
                "Model Name",
                value=st.session_state.ollama_model,
                help="e.g. qwen2.5:7b",
            )

        st.markdown("---")
        st.caption(f"Model: `{st.session_state.ollama_model}`")

    # -------------------------------------------------------------------------
    # Check required configs
    # -------------------------------------------------------------------------
    required_configs = {
        "Zoom Account ID": st.session_state.zoom_account_id,
        "Zoom Client ID": st.session_state.zoom_client_id,
        "Zoom Client Secret": st.session_state.zoom_client_secret,
        "Email Sender": st.session_state.email_sender,
        "Email Password": st.session_state.email_passkey,
        "Company Name": st.session_state.company_name,
>>>>>>> addc0a0 (update pipeline)
    }

    missing_configs = [k for k, v in required_configs.items() if not v]
    if missing_configs:
<<<<<<< HEAD
        st.warning(f" Please configure: {', '.join(missing_configs)}")
        return

    # Main Interface
    st.markdown("---")

    # Step 1: Define Role
    st.header(" Step 1: Define Job Role")
    col1, col2 = st.columns(2)

    with col1:
        role_name = st.text_input("Role/Position Name",
                                  value=st.session_state.custom_role_name,
                                  placeholder="e.g., Senior Backend Engineer")
        if role_name:
            st.session_state.custom_role_name = role_name
=======
        st.warning(f"Missing config in .env: {', '.join(missing_configs)}")
        return

    st.info(
        f"Mode: On-Premise Ollama — Model: `{st.session_state.ollama_model}` — "
        f"Email via SMTP"
    )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Step 1: Define Job Role
    # -------------------------------------------------------------------------
    st.header("Step 1: Define Job Role")

    role_options = list(ROLE_REQUIREMENTS.keys())

    col1, col2 = st.columns(2)

    with col1:
        role_name = st.selectbox(
            "Select Role / Position",
            options=role_options,
            index=role_options.index(st.session_state.custom_role_name)
            if st.session_state.custom_role_name in role_options
            else 0,
            help="Select a predefined role. Requirements will auto-fill below.",
        )
        st.session_state.custom_role_name = role_name
>>>>>>> addc0a0 (update pipeline)

    with col2:
        st.info(" Tip: Be specific about the role title")

    requirements = st.text_area(
        "Job Requirements (Skills, Experience, etc.)",
        value=st.session_state.custom_requirements,
        height=200,
        placeholder="""Example:
Required Skills:
- Python, Django, FastAPI
- PostgreSQL, Redis
- Docker, Kubernetes
- REST API design
- 3+ years experience

Nice to have:
- AWS/GCP experience
- Microservices architecture
- TDD/BDD practices
        """
    )
    if requirements:
        st.session_state.custom_requirements = requirements

<<<<<<< HEAD
    if not role_name or not requirements:
        st.warning(" Please define role name and requirements to continue")
=======
    if not requirements:
        st.warning("Please define requirements to continue.")
>>>>>>> addc0a0 (update pipeline)
        return

    st.markdown("---")

<<<<<<< HEAD
    # Step 2: Upload CVs and Emails
    st.header(" Step 2: Upload CVs and Enter Emails")
=======
    # -------------------------------------------------------------------------
    # Step 2: Upload CVs and Enter Emails
    # -------------------------------------------------------------------------
    st.header("Step 2: Upload CVs and Enter Emails")
>>>>>>> addc0a0 (update pipeline)

    uploaded_files = st.file_uploader(
        "Upload Resume PDFs (multiple files allowed)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
<<<<<<< HEAD
        st.success(f" {len(uploaded_files)} CV(s) uploaded")
=======
        st.success(f"{len(uploaded_files)} CV(s) uploaded.")
>>>>>>> addc0a0 (update pipeline)

        with st.expander(" View Uploaded CVs", expanded=True):
            for file in uploaded_files:
                st.text(f"- {file.name}")

<<<<<<< HEAD
        st.subheader(" Candidate Emails")
        st.info(" System will auto-detect emails from CVs. You can edit or add missing emails below.")
=======
        st.subheader("Candidate Emails")
        st.caption("System will auto-detect emails from CVs. You can edit or add missing emails below.")
>>>>>>> addc0a0 (update pipeline)

        # Read bytes once; extract_text_from_pdf is cached on bytes content
        cv_data = []
        for file in uploaded_files:
            file_bytes = file.read()
            resume_text = extract_text_from_pdf(file_bytes)
            extracted_email = extract_email_from_text(resume_text) if resume_text else None
            cv_data.append({
                'filename': file.name,
                'resume_text': resume_text,
                'extracted_email': extracted_email
            })

<<<<<<< HEAD
        with st.expander(" Auto-Detected Emails Preview", expanded=True):
            preview_data = []
            for i, data in enumerate(cv_data):
                status = " Found" if data['extracted_email'] else " Not found"
                preview_data.append({
                    'No': i + 1,
                    'CV File': data['filename'],
                    'Detected Email': data['extracted_email'] or '(not found)',
                    'Status': status
                })
            preview_df = pd.DataFrame(preview_data)
            st.dataframe(preview_df, use_container_width=True)

        default_emails = '\n'.join([
            data['extracted_email'] if data['extracted_email'] else ''
            for data in cv_data
        ])
=======
        with st.expander("Auto-Detected Info Preview", expanded=True):
            preview_data = []
            for i, data in enumerate(cv_data):
                preview_data.append(
                    {
                        "No": i + 1,
                        "CV File": data["filename"],
                        "Detected Name": data["extracted_name"] or "(not found)",
                        "Name Found": "Yes" if data["extracted_name"] else "No",
                        "Detected Email": data["extracted_email"] or "(not found)",
                        "Email Found": "Yes" if data["extracted_email"] else "No",
                    }
                )
            st.dataframe(pd.DataFrame(preview_data), use_container_width=True)

        default_emails = "\n".join(
            [data["extracted_email"] if data["extracted_email"] else "" for data in cv_data]
        )
>>>>>>> addc0a0 (update pipeline)

        emails_input = st.text_area(
            "Candidate Emails (edit if needed)",
            value=default_emails,
            height=150,
<<<<<<< HEAD
            placeholder="candidate1@email.com\ncandidate2@email.com\ncandidate3@email.com",
            help="Auto-detected emails are pre-filled. Edit or add missing emails."
=======
            placeholder="candidate1@email.com\ncandidate2@email.com",
            help="Auto-detected emails are pre-filled. Edit or add missing emails.",
>>>>>>> addc0a0 (update pipeline)
        )

        if emails_input:
            emails = [e.strip() for e in emails_input.split('\n') if e.strip()]

            if len(emails) != len(uploaded_files):
<<<<<<< HEAD
                st.error(f" Mismatch! You have {len(uploaded_files)} CVs but {len(emails)} emails")
            else:
                st.success(f" {len(emails)} emails matched with CVs")

                with st.expander(" CV-Email Mapping", expanded=True):
                    mapping_df = pd.DataFrame({
                        'CV File': [d['filename'] for d in cv_data],
                        'Email': emails
                    })
                    st.dataframe(mapping_df, use_container_width=True)

                if st.button(" Process All Applications", type="primary", use_container_width=True):
                    with st.spinner(" Processing applications..."):
                        # Reuse already-read resume text from cv_data (no redundant PDF re-read)
=======
                st.error(
                    f"Mismatch: {len(uploaded_files)} CVs but {len(emails)} emails provided."
                )
            else:
                st.success(f"{len(emails)} emails matched with CVs.")

                with st.expander("CV - Name - Email Mapping", expanded=True):
                    mapping_df = pd.DataFrame(
                        {
                            "CV File": [f.name for f in uploaded_files],
                            "Candidate Name": [
                                data["extracted_name"] or "(unknown)" for data in cv_data
                            ],
                            "Email": emails,
                        }
                    )
                    st.dataframe(mapping_df, use_container_width=True)

                if st.button("Process All Applications", type="primary", use_container_width=True):
                    with st.spinner("Processing applications..."):
>>>>>>> addc0a0 (update pipeline)
                        candidates_data = []
                        for cv, email in zip(cv_data, emails):
                            if cv['resume_text']:
                                candidates_data.append({
                                    'email': email,
                                    'filename': cv['filename'],
                                    'resume_text': cv['resume_text']
                                })

                        results = process_batch_applications(
                            candidates_data,
                            st.session_state.custom_role_name,
                            st.session_state.custom_requirements
                        )

                        st.session_state.batch_results = results
                        st.session_state.processing_complete = True
                        st.rerun()

    # -------------------------------------------------------------------------
    # Step 3: Results
    # -------------------------------------------------------------------------
    if st.session_state.processing_complete and st.session_state.batch_results:
        st.markdown("---")
<<<<<<< HEAD
        st.header(" Processing Results")
=======
        st.header("Processing Results")
>>>>>>> addc0a0 (update pipeline)

        results = st.session_state.batch_results
        selected_count = sum(1 for r in results if r['selected'])
        rejected_count = len(results) - selected_count
        success_rate = (selected_count / len(results) * 100) if results else 0.0

<<<<<<< HEAD
        # --- Real-time Statistics Dashboard ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📋 Total CV", len(results))
        col2.metric("✅ Kandidat Terpilih", selected_count)
        col3.metric("❌ Kandidat Ditolak", rejected_count)
        col4.metric("📊 Success Rate", f"{success_rate:.1f}%")

        # Experience Level Distribution Chart
        exp_levels = [r.get('analysis', {}).get('experience_level', 'unknown') for r in results]
        if any(lvl != 'unknown' for lvl in exp_levels):
            st.subheader("📈 Experience Level Distribution")
            exp_series = pd.Series(exp_levels, name="count")
            exp_counts = exp_series.value_counts().rename_axis("Experience Level").to_frame("Count")
            st.bar_chart(exp_counts)

        st.markdown("---")

        # --- Interactive Results Table ---
        st.subheader("📋 Tabel Hasil Seleksi")
        st.caption("Klik header kolom untuk sorting. Gunakan filter bawaan Streamlit untuk eksplorasi data.")

        table_data = []
        for r in results:
            analysis = r.get('analysis', {})
            matching = analysis.get('matching_skills', [])
            table_data.append({
                'Email': r['email'],
                'File CV': r['filename'],
                'Status': '✅ Terpilih' if r['selected'] else '❌ Ditolak',
                'Experience Level': analysis.get('experience_level', '-'),
                'Matching Skills': ', '.join(matching) if matching else '-',
                'Email Sent': '✅' if r['email_sent'] else '❌',
                'Zoom Scheduled': '✅' if r['interview_scheduled'] else '❌',
                'Zoom Link': r.get('zoom_link', '') or '-',
                'Error': r.get('error', '') or '-'
            })

        results_df = pd.DataFrame(table_data)
        st.dataframe(results_df, use_container_width=True, height=300)

        st.markdown("---")

        # --- Detailed Candidate Cards ---
        st.subheader(" Detailed Results")

        if selected_count > 0:
            st.success(f" Selected Candidates ({selected_count})")
            for result in results:
                if result['selected']:
                    with st.expander(f" {result['email']} - {result['filename']}"):
                        st.write("**Feedback:**", result['feedback'])
                        st.write("**Email Sent:**", "✅ Yes" if result['email_sent'] else "❌ No")
                        st.write("**Interview Scheduled:**", "✅ Yes" if result['interview_scheduled'] else "❌ No")
                        if result.get('zoom_link'):
                            st.write("**Zoom Link:**", result['zoom_link'])
                        if result.get('analysis'):
                            st.json(result['analysis'])
                        if result.get('error'):
                            st.warning(f"⚠️ Partial error: {result['error']}")
=======
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Processed", len(results))
        col2.metric("Selected", selected_count)
        col3.metric("Rejected", rejected_count)

        st.subheader("Detailed Results")

        if selected_count > 0:
            st.success(f"Selected Candidates ({selected_count})")
            for result in results:
                if result["selected"]:
                    display_name = result.get("name", "") or result["email"]
                    with st.expander(f"[Selected] {display_name} — {result['filename']}"):
                        if result.get("name"):
                            st.write("**Candidate:**", result["name"])
                        st.write("**Email:**", result["email"])
                        st.write("**Feedback:**", result["feedback"])
                        st.write("**Email Sent:**", "Yes" if result["email_sent"] else "No")
                        st.write(
                            "**Interview Scheduled:**",
                            "Yes" if result["interview_scheduled"] else "No",
                        )
                        if result.get("interview_time"):
                            st.write("**Interview Time:**", result["interview_time"])
                        if result.get("analysis"):
                            st.json(result["analysis"])
>>>>>>> addc0a0 (update pipeline)

        if rejected_count > 0:
            st.error(f"Rejected Candidates ({rejected_count})")
            for result in results:
<<<<<<< HEAD
                if not result['selected']:
                    with st.expander(f"{result['email']} - {result['filename']}"):
                        st.write("**Feedback:**", result['feedback'])
                        st.write("**Email Sent:**", "✅ Yes" if result['email_sent'] else "❌ No")
                        if result.get('analysis'):
                            st.json(result['analysis'])
                        if result.get('error'):
                            st.warning(f"⚠️ Partial error: {result['error']}")
=======
                if not result["selected"]:
                    display_name = result.get("name", "") or result["email"]
                    with st.expander(f"[Rejected] {display_name} — {result['filename']}"):
                        if result.get("name"):
                            st.write("**Candidate:**", result["name"])
                        st.write("**Email:**", result["email"])
                        st.write("**Feedback:**", result["feedback"])
                        st.write("**Email Sent:**", "Yes" if result["email_sent"] else "No")
                        if result.get("analysis"):
                            st.json(result["analysis"])
>>>>>>> addc0a0 (update pipeline)

        # --- Downloadable Report (all columns) ---
        st.markdown("---")
        if st.button("📥 Export Results to CSV"):
            export_data = []
            for r in results:
                analysis = r.get('analysis', {})
                export_data.append({
                    'Email': r['email'],
                    'Filename': r['filename'],
                    'Selected': r['selected'],
                    'Experience Level': analysis.get('experience_level', ''),
                    'Feedback': r['feedback'],
                    'Matching Skills': ', '.join(analysis.get('matching_skills', [])),
                    'Missing Skills': ', '.join(analysis.get('missing_skills', [])),
                    'Email Sent': r['email_sent'],
                    'Interview Scheduled': r['interview_scheduled'],
                    'Zoom Link': r.get('zoom_link', '') or '',
                    'Error': r.get('error', '') or ''
                })

            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"recruitment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    # Reset button
    if st.sidebar.button("Start New Batch"):
        st.session_state.batch_results = []
        st.session_state.processing_complete = False
        st.rerun()


if __name__ == "__main__":
    main()
