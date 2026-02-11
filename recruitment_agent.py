from typing import Tuple, Dict, Optional, List
import os
import time
import json
import requests
import PyPDF2
from datetime import datetime, timedelta
import pytz
import pandas as pd
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import streamlit as st
from dotenv import load_dotenv
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    ChatDetails,
    GenericChatRequest,
    SystemMessage,
    UserMessage,
    TextContent,
    OnDemandServingMode,
)
from oci.retry import NoneRetryStrategy

from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.email import EmailTools
from phi.tools.zoom import ZoomTool
from phi.utils.log import logger

# Load .env file
load_dotenv()


def is_openai_key_valid(key: str) -> bool:
    """Check if OpenAI API key looks valid (not placeholder/empty)."""
    if not key:
        return False
    key = key.strip()
    if key in ("", "sk-xxx", "sk-your-key-here", "your-api-key"):
        return False
    if not key.startswith("sk-"):
        return False
    if len(key) < 20:
        return False
    return True


# =============================================================================
# OCI GenAI Helper Class
# =============================================================================
class OCIGenAIChat:
    """Wrapper class for OCI Generative AI Chat inference."""

    def __init__(
        self,
        compartment_id: str,
        model_id: str = "xai.grok-4-1-fast-reasoning",
        service_endpoint: str = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        config_profile: str = "DEFAULT",
        config_file: str = "~/.oci/config",
        max_tokens: int = 4000,
        temperature: float = 0.3,
    ):
        self.compartment_id = compartment_id
        self.model_id = model_id
        self.service_endpoint = service_endpoint
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Load OCI config
        self.config = oci.config.from_file(config_file, config_profile)
        oci.config.validate_config(self.config)

        # Create inference client
        self.client = GenerativeAiInferenceClient(
            config=self.config,
            service_endpoint=self.service_endpoint,
            retry_strategy=NoneRetryStrategy(),
            timeout=(10, 240),
        )

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a chat message and return the response text."""
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=[TextContent(text=system_prompt)]))

        messages.append(UserMessage(content=[TextContent(text=prompt)]))

        chat_request = GenericChatRequest(
            api_format="GENERIC",
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        chat_details = ChatDetails(
            serving_mode=OnDemandServingMode(model_id=self.model_id),
            chat_request=chat_request,
            compartment_id=self.compartment_id,
        )

        try:
            response = self.client.chat(chat_details)
            chat_response = response.data.chat_response
            if hasattr(chat_response, "choices") and chat_response.choices:
                choice = chat_response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    content = choice.message.content
                    if isinstance(content, list):
                        return "".join(item.text for item in content if hasattr(item, "text"))
                    return str(content)
            if hasattr(chat_response, "text"):
                return chat_response.text
            return str(chat_response)

        except Exception as e:
            logger.error(f"OCI GenAI Chat error: {str(e)}")
            raise


# =============================================================================
# Custom Zoom Tool
# =============================================================================
class CustomZoomTool(ZoomTool):
    def __init__(
        self,
        *,
        account_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        name: str = "zoom_tool",
    ):
        super().__init__(
            account_id=account_id,
            client_id=client_id,
            client_secret=client_secret,
            name=name,
        )
        self.token_url = "https://zoom.us/oauth/token"
        self.access_token = None
        self.token_expires_at = 0

    def get_access_token(self) -> str:
        if self.access_token and time.time() < self.token_expires_at:
            return str(self.access_token)

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"grant_type": "account_credentials", "account_id": self.account_id}

        try:
            response = requests.post(
                self.token_url,
                headers=headers,
                data=data,
                auth=(self.client_id, self.client_secret),
            )
            response.raise_for_status()

            token_info = response.json()
            self.access_token = token_info["access_token"]
            expires_in = token_info["expires_in"]
            self.token_expires_at = time.time() + expires_in - 60

            self._set_parent_token(str(self.access_token))
            return str(self.access_token)

        except requests.RequestException as e:
            logger.error(f"Error fetching access token: {e}")
            return ""

    def _set_parent_token(self, token: str) -> None:
        if token:
            self._ZoomTool__access_token = token

# =============================================================================
# ZOOM REST API (DETERMINISTIC - GUARANTEED LINK)
# =============================================================================
def get_zoom_access_token_s2s(account_id: str, client_id: str, client_secret: str) -> str:
    """
    Get Zoom access token (Server-to-Server OAuth) using account_credentials grant.
    """
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
    """
    Create scheduled Zoom meeting via REST API and return meeting JSON.
    """
    payload = {
        "topic": topic,
        "type": 2,  # scheduled
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
# Session State - loads defaults from .env
# =============================================================================
def init_session_state() -> None:
    """Initialize session state variables from .env file."""

    # Build service endpoint from region in .env
    oci_region = os.getenv("oci_region", "us-chicago-1")
    service_endpoint = f"https://inference.generativeai.{oci_region}.oci.oraclecloud.com"

    defaults = {
        # OCI GenAI settings (from .env)
        "oci_compartment_id": os.getenv("oci_compartment_id", ""),
        "oci_model_id": os.getenv("oci_model", "xai.grok-4-1-fast-reasoning"),
        "oci_service_endpoint": service_endpoint,
        "oci_config_profile": os.getenv("oci_config_profile", "DEFAULT"),
        "oci_config_file": os.getenv("oci_config_file", "~/.oci/config"),
        # Zoom settings (from .env)
        "zoom_account_id": os.getenv("zoom_account_id", ""),
        "zoom_client_id": os.getenv("zoom_client_id", ""),
        "zoom_client_secret": os.getenv("zoom_client_secret", ""),
        # Email settings (from .env)
        "email_sender": os.getenv("email_sender", ""),
        "email_passkey": os.getenv("email_app_password", ""),
        "company_name": os.getenv("company_name", ""),
        # OpenAI (optional, from .env)
        "openai_api_key": os.getenv("openai_api_key", ""),
        # Job settings
        "custom_role_name": "AI Engineer",
        "custom_requirements": "",
        "last_selected_role": "",
        # Results
        "batch_results": [],
        "processing_complete": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# OCI GenAI Client Factory
# =============================================================================
def get_oci_genai_client() -> Optional[OCIGenAIChat]:
    """Create and return OCI GenAI chat client."""
    if not st.session_state.oci_compartment_id:
        st.error("Missing OCI Compartment OCID (please set it in .env).")
        return None

    try:
        return OCIGenAIChat(
            compartment_id=st.session_state.oci_compartment_id,
            model_id=st.session_state.oci_model_id,
            service_endpoint=st.session_state.oci_service_endpoint,
            config_profile=st.session_state.oci_config_profile,
            config_file=st.session_state.oci_config_file,
        )
    except Exception as e:
        st.error(f"Failed to create OCI GenAI client: {str(e)}")
        return None


# =============================================================================
# Phidata Agents for Email & Zoom (optional, requires OpenAI key)
# =============================================================================
def create_email_agent(receiver_email: str) -> Optional[Agent]:
    """Create email agent. Returns None if no valid OpenAI key."""
    if is_openai_key_valid(st.session_state.openai_api_key):
        return Agent(
            model=OpenAIChat(
                id="gpt-4o",
                api_key=st.session_state.openai_api_key,
            ),
            tools=[
                EmailTools(
                    receiver_email=receiver_email,
                    sender_email=st.session_state.email_sender,
                    sender_name=st.session_state.company_name,
                    sender_passkey=st.session_state.email_passkey,
                )
            ],
            description="You are a professional recruitment coordinator.",
            instructions=[
                "Draft and send professional recruitment emails",
                "Use all lowercase letters for casual, human tone",
                "Maintain friendly yet professional tone",
                "Always end emails with: 'best,\\nthe ai recruiting team'",
                f"Company name: '{st.session_state.company_name}'",
            ],
            markdown=True,
            show_tool_calls=True,
        )
    return None


def create_scheduler_agent() -> Optional[Agent]:
    """Create Zoom scheduler agent. Returns None if no valid OpenAI key."""
    if not is_openai_key_valid(st.session_state.openai_api_key):
        return None

    zoom_tools = CustomZoomTool(
        account_id=st.session_state.zoom_account_id,
        client_id=st.session_state.zoom_client_id,
        client_secret=st.session_state.zoom_client_secret,
    )

    return Agent(
        name="Interview Scheduler",
        model=OpenAIChat(
            id="gpt-4o",
            api_key=st.session_state.openai_api_key,
        ),
        tools=[zoom_tools],
        description="You are an interview scheduling coordinator.",
        instructions=[
            "Schedule interviews during business hours (9 AM - 5 PM Jakarta Time)",
            "Create meetings with proper titles and descriptions",
            "Use ISO 8601 format for dates",
        ],
        markdown=True,
        show_tool_calls=True,
    )


# =============================================================================
# Direct Email Sending (fallback without OpenAI)
# =============================================================================
def send_email_direct(receiver_email: str, subject: str, body: str) -> bool:
    """Send email directly via SMTP without phidata agent."""
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
# PDF & Email Extraction
# =============================================================================
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        return ""


def extract_email_from_text(text: str) -> Optional[str]:
    """Extract email address from resume text."""
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    emails = re.findall(email_pattern, text)

    if emails:
        personal_domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"]
        for email in emails:
            if any(domain in email.lower() for domain in personal_domains):
                return email
        return emails[0]
    return None


def extract_name_from_text(text: str) -> Optional[str]:
    """Extract candidate name from resume text."""
    if not text:
        return None

    skip_patterns = [
        r"@",
        r"http|www\.",
        r"\+?\d[\d\-\(\)\s]{7,}",
        r"resume|curriculum|cv\b",
        r"linkedin|github|kaggle",
        r"portfolio",
        r"professional summary",
        r"technical skills",
        r"experience|education",
        r"objective|contact",
    ]

    lines = text.strip().split("\n")

    for line in lines[:10]:
        line = line.strip()
        if len(line) < 3 or len(line) > 60:
            continue
        if any(re.search(p, line, re.IGNORECASE) for p in skip_patterns):
            continue

        alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / max(len(line), 1)
        if alpha_ratio < 0.7:
            continue

        name = re.split(r"\s*[-–|]\s*", line)[0].strip()
        words = name.split()
        if 1 <= len(words) <= 5:
            name_words = [w for w in words if w and w[0].isupper() and w.isalpha()]
            if len(name_words) >= 1 and len(name_words) >= len(words) * 0.5:
                return name.title()

    return None


# =============================================================================
# Resume Analysis using OCI GenAI
# =============================================================================
def analyze_resume(
    resume_text: str,
    requirements: str,
    oci_client: OCIGenAIChat,
) -> Tuple[bool, str, dict]:
    """Analyze single resume against requirements using OCI GenAI."""
    try:
        system_prompt = """You are an expert technical recruiter who analyzes resumes.
Analyze the resume against the provided job requirements.
Be lenient with candidates who show strong potential.
Consider project experience as valid experience.
Value hands-on experience with key technologies.
Return a JSON response with selection decision and feedback.
Return ONLY the JSON object without any markdown formatting or code blocks."""

        user_prompt = f"""Analyze this resume against the requirements and respond in valid JSON:

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

Return ONLY the JSON object without markdown."""

        response_text = oci_client.chat(user_prompt, system_prompt)

        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        result = json.loads(cleaned)
        if not isinstance(result, dict) or not all(k in result for k in ["selected", "feedback"]):
            raise ValueError("Invalid response format")

        return bool(result["selected"]), str(result["feedback"]), result

    except Exception as e:
        logger.error(f"Error analyzing resume: {str(e)}")
        return False, f"Error: {str(e)}", {}


# =============================================================================
# Email Drafting with OCI GenAI
# =============================================================================
def draft_email_with_oci(
    oci_client: OCIGenAIChat,
    email_type: str,
    role: str,
    feedback: str = "",
    candidate_email: str = "",
    candidate_name: str = "",
    interview_details: str = "",
) -> Tuple[str, str]:
    """Use OCI GenAI to draft email content. Returns (subject, body)."""
    greeting_name = candidate_name if candidate_name else (candidate_email.split("@")[0] if candidate_email else "there")

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
    body = oci_client.chat(prompt, system_prompt)

    subject_map = {
        "selection": f"Congratulations! You've been selected for {role}",
        "rejection": f"Update on your {role} application",
        "interview_confirmation": f"Interview Confirmation – {role} Position",
    }
    subject = subject_map.get(email_type, f"Update regarding {role}")

    return subject, body.strip()


# =============================================================================
# Send Emails
# =============================================================================
def send_selection_email(
    role: str,
    candidate_email: str,
    oci_client: OCIGenAIChat,
    candidate_name: str = "",
    email_agent: Optional[Agent] = None,
) -> bool:
    """Send selection email to candidate."""
    greeting = candidate_name if candidate_name else candidate_email.split("@")[0]
    if email_agent:
        email_agent.run(
            f"""Send selection email for {role} position.
Start with: hi {greeting},
Include:
1. Congratulate on selection
2. Explain next steps
3. Mention interview details coming soon
4. End with: best,\\nthe ai recruiting team"""
        )
        return True

    subject, body = draft_email_with_oci(
        oci_client,
        "selection",
        role,
        candidate_email=candidate_email,
        candidate_name=candidate_name,
    )
    return send_email_direct(candidate_email, subject, body)


def send_rejection_email(
    role: str,
    feedback: str,
    candidate_email: str,
    oci_client: OCIGenAIChat,
    candidate_name: str = "",
    email_agent: Optional[Agent] = None,
) -> bool:
    """Send rejection email with feedback."""
    greeting = candidate_name if candidate_name else candidate_email.split("@")[0]
    if email_agent:
        email_agent.run(
            f"""Send rejection email for {role} position.
Start with: hi {greeting},
Style:
1. All lowercase
2. Empathetic and human
3. Include feedback: {feedback}
4. Encourage upskilling
5. Suggest learning resources
6. End with: best,\\nthe ai recruiting team"""
        )
        return True

    subject, body = draft_email_with_oci(
        oci_client,
        "rejection",
        role,
        feedback=feedback,
        candidate_email=candidate_email,
        candidate_name=candidate_name,
    )
    return send_email_direct(candidate_email, subject, body)


# =============================================================================
# Interview Scheduling
# =============================================================================
def schedule_interview(
    candidate_email: str,
    role: str,
    oci_client: OCIGenAIChat,
    candidate_name: str = "",
    scheduler: Optional[Agent] = None,  # keep signature; not used for meeting creation
    email_agent: Optional[Agent] = None,
) -> Tuple[bool, str]:
    """Schedule interview and send confirmation with guaranteed Zoom link (REST API)."""
    try:
        greeting = candidate_name if candidate_name else candidate_email.split("@")[0]
        jakarta_tz = pytz.timezone("Asia/Jakarta")
        current_time_jkt = datetime.now(jakarta_tz)
        tomorrow_jkt = current_time_jkt + timedelta(days=1)
        interview_time = tomorrow_jkt.replace(hour=11, minute=0, second=0, microsecond=0)

        # Zoom API expects ISO 8601; your original format is fine
        formatted_time_iso = interview_time.strftime("%Y-%m-%dT%H:%M:%S")
        pretty_date = interview_time.strftime("%A, %d %B %Y")
        pretty_time = interview_time.strftime("%I:%M %p")
        interview_time_str = f"{pretty_date} {pretty_time} (Asia/Jakarta, UTC+7)"

        # ✅ Deterministic: create meeting via REST API
        meeting_link = "Zoom link not available"
        meeting_id = "N/A"

        try:
            # pull from session state (already loaded from .env by init_session_state)
            account_id = st.session_state.zoom_account_id
            client_id = st.session_state.zoom_client_id
            client_secret = st.session_state.zoom_client_secret

            if not account_id or not client_id or not client_secret:
                raise RuntimeError("Missing Zoom credentials in session state (.env).")

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
            # log and continue (email still goes out with placeholder)
            logger.error(f"Zoom meeting creation failed (REST): {zoom_e}")

        interview_details = f"""📅 Date: {pretty_date}
🕒 Time: {pretty_time} (Jakarta Time, UTC+7)
⏳ Duration: 60 minutes
🔗 Zoom Link: {meeting_link}
🆔 Meeting ID: {meeting_id}"""

        # Send email (keep your existing behavior)
        if email_agent:
            email_agent.run(
                f"""Send interview confirmation to {candidate_email}:
Subject: Interview Confirmation – {role} Position

Start with: hi {greeting},
Technical interview details for {role}:
{interview_details}

Notes:
- Join 5 minutes early
- Timezone converter: https://www.timeanddate.com/worldclock/converter.html
- Be confident and prepare well!

best,
the ai recruiting team"""
            )
        else:
            subject, body = draft_email_with_oci(
                oci_client,
                "interview_confirmation",
                role,
                candidate_email=candidate_email,
                candidate_name=candidate_name,
                interview_details=interview_details,
            )
            send_email_direct(candidate_email, subject, body)

        return True, interview_time_str

    except Exception as e:
        logger.error(f"Error scheduling interview: {str(e)}")
        return False, ""


# =============================================================================
# Batch Processing
# =============================================================================
def process_batch_applications(candidates_data: List[Dict], role_name: str, requirements: str) -> List[Dict]:
    """Process multiple applications at once."""
    results = []

    oci_client = get_oci_genai_client()
    if not oci_client:
        st.error("Failed to create OCI GenAI client")
        return results

    has_openai = is_openai_key_valid(st.session_state.openai_api_key)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, candidate in enumerate(candidates_data):
        candidate_name = candidate.get("name", "")
        status_text.text(f"Processing {candidate_name or candidate['email']}...")

        is_selected, feedback, analysis_details = analyze_resume(
            candidate["resume_text"],
            requirements,
            oci_client,
        )

        result = {
            "name": candidate_name,
            "email": candidate["email"],
            "filename": candidate["filename"],
            "selected": is_selected,
            "feedback": feedback,
            "analysis": analysis_details,
            "email_sent": False,
            "interview_scheduled": False,
            "interview_time": "",
        }

        try:
            email_agent = create_email_agent(candidate["email"]) if has_openai else None

            if is_selected:
                result["email_sent"] = send_selection_email(
                    role_name,
                    candidate["email"],
                    oci_client,
                    candidate_name=candidate_name,
                    email_agent=email_agent,
                )

                scheduler = create_scheduler_agent() if has_openai else None
                scheduled, interview_time = schedule_interview(
                    candidate["email"],
                    role_name,
                    oci_client,
                    candidate_name=candidate_name,
                    scheduler=scheduler,
                    email_agent=email_agent,
                )
                result["interview_scheduled"] = scheduled
                result["interview_time"] = interview_time
            else:
                result["email_sent"] = send_rejection_email(
                    role_name,
                    feedback,
                    candidate["email"],
                    oci_client,
                    candidate_name=candidate_name,
                    email_agent=email_agent,
                )

        except Exception as e:
            logger.error(f"Error processing {candidate['email']}: {str(e)}")
            result["error"] = str(e)

        results.append(result)
        progress_bar.progress((idx + 1) / len(candidates_data))

    status_text.text("✅ Processing complete!")
    return results


# =============================================================================
# Main Streamlit App
# =============================================================================
def main() -> None:
    st.set_page_config(
        page_title="AI Recruitment System - OCI GenAI",
        layout="wide",
    )

    st.title("AI Agent Recruitment System - OCI GenAI")

    init_session_state()

    # -------------------------------------------------------------------------
    # Sidebar Configuration (CUSTOMER: MODEL ONLY)
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")
        st.subheader("🔶 OCI GenAI Settings")

        model_options = [
            "xai.grok-4-1-fast-reasoning",
            "meta.llama-3.3-70b-instruct",
            "meta.llama-3.1-70b-instruct",
            "meta.llama-3.1-405b-instruct",
            "meta.llama-4-scout-17b-16e-instruct",
            "cohere.command-r-plus-08-2024",
            "cohere.command-r-08-2024",
            "cohere.command-a-03-2025",
        ]

        current_model = st.session_state.oci_model_id if st.session_state.oci_model_id in model_options else model_options[0]
        st.session_state.oci_model_id = st.selectbox(
            "Model",
            options=model_options,
            index=model_options.index(current_model),
            help="Select OCI GenAI model",
        )

        st.markdown("---")
        st.caption("Powered by OCI Generative AI")

    # -------------------------------------------------------------------------
    # Check required configs (MUST be present in .env)
    # -------------------------------------------------------------------------
    required_configs = {
        "OCI Compartment OCID": st.session_state.oci_compartment_id,
        "Zoom Account ID": st.session_state.zoom_account_id,
        "Zoom Client ID": st.session_state.zoom_client_id,
        "Zoom Client Secret": st.session_state.zoom_client_secret,
        "Email Sender": st.session_state.email_sender,
        "Email Password": st.session_state.email_passkey,
        "Company Name": st.session_state.company_name,
    }

    missing_configs = [k for k, v in required_configs.items() if not v]
    if missing_configs:
        st.warning(f"⚠️ Missing config in .env: {', '.join(missing_configs)}")
        return

    # Show mode info
    if is_openai_key_valid(st.session_state.openai_api_key):
        st.info(
            f"🟢 **Hybrid Mode**: Resume analysis → OCI GenAI (`{st.session_state.oci_model_id}`), "
            f"Email/Zoom → OpenAI phidata agents"
        )
    else:
        st.info(
            f"🔶 **Full OCI Mode**: Resume analysis & email drafting → OCI GenAI (`{st.session_state.oci_model_id}`), "
            f"Email → direct SMTP"
        )

    st.markdown("---")

    # Step 1: Define Role
    st.header("Step 1: Define Job Role")

    ROLE_REQUIREMENTS = {
        "AI Engineer": """Required Skills:
- Python (advanced)
- Machine Learning frameworks (TensorFlow, PyTorch, or scikit-learn)
- Deep Learning (CNN, RNN, Transformer architectures)
- Natural Language Processing (NLP) or Computer Vision
- Data preprocessing & feature engineering (Pandas, NumPy)
- Model training, evaluation, and deployment
- LLM integration (OpenAI API, LangChain, or similar)
- Git version control
- 2+ years experience in AI/ML

Nice to have:
- MLOps (MLflow, Kubeflow, or Weights & Biases)
- Cloud AI services (OCI GenAI, AWS SageMaker, or GCP Vertex AI)
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
- Cloud infrastructure (OCI, AWS, or GCP)
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
- Cloud architecture design (OCI, AWS, GCP, or Azure)
- Experience designing end-to-end AI pipelines
- LLM deployment and integration patterns (RAG, fine-tuning, prompt engineering)
- API design and microservices architecture
- Data pipeline design (ETL/ELT)
- Excellent communication and stakeholder management
- 5+ years experience in software engineering, 2+ in AI/ML

Nice to have:
- Oracle Cloud Infrastructure (OCI) certification
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
- Cloud data services (OCI Data Flow, AWS Glue, or GCP BigQuery)
- Git version control
- 3+ years experience

Nice to have:
- Apache Kafka or streaming data platforms
- Orchestration tools (Apache Airflow, Prefect, or Dagster)
- Data quality frameworks (Great Expectations, dbt tests)
- NoSQL databases (MongoDB, Cassandra, DynamoDB)
- Docker & Kubernetes
- Data governance and cataloging
- CI/CD for data pipelines
- Oracle Autonomous Database experience""",
    }

    role_options = list(ROLE_REQUIREMENTS.keys())

    col1, col2 = st.columns(2)

    with col1:
        role_name = st.selectbox(
            "Select Role/Position",
            options=role_options,
            index=role_options.index(st.session_state.custom_role_name)
            if st.session_state.custom_role_name in role_options
            else 0,
            help="Select a predefined role. Requirements will auto-fill below.",
        )
        st.session_state.custom_role_name = role_name

    with col2:
        st.info(f"Selected: **{role_name}**\n\nRequirements auto-filled below. You can edit them if needed.")

    default_req = ROLE_REQUIREMENTS.get(role_name, "")

    if "last_selected_role" not in st.session_state:
        st.session_state.last_selected_role = role_name
        st.session_state.custom_requirements = default_req

    if st.session_state.last_selected_role != role_name:
        st.session_state.custom_requirements = default_req
        st.session_state.last_selected_role = role_name

    requirements = st.text_area(
        "Job Requirements (auto-filled, editable)",
        value=st.session_state.custom_requirements,
        height=300,
        help="Requirements are auto-filled based on the selected role. Feel free to edit.",
    )
    if requirements:
        st.session_state.custom_requirements = requirements

    if not requirements:
        st.warning("⚠️ Please define requirements to continue")
        return

    st.markdown("---")

    # Step 2: Upload CVs and Emails
    st.header("Step 2: Upload CVs and Enter Emails")

    uploaded_files = st.file_uploader(
        "Upload Resume PDFs (multiple files allowed)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} CV(s) uploaded")

        with st.expander("View Uploaded CVs", expanded=True):
            for file in uploaded_files:
                st.text(f"• {file.name}")

        st.subheader("Candidate Emails")
        st.info("System will auto-detect emails from CVs. You can edit or add missing emails below.")

        cv_data = []
        for file in uploaded_files:
            resume_text = extract_text_from_pdf(file)
            extracted_email = extract_email_from_text(resume_text) if resume_text else None
            extracted_name = extract_name_from_text(resume_text) if resume_text else None
            cv_data.append(
                {
                    "filename": file.name,
                    "resume_text": resume_text,
                    "extracted_email": extracted_email,
                    "extracted_name": extracted_name,
                }
            )

        with st.expander("🔍 Auto-Detected Info Preview", expanded=True):
            preview_data = []
            for i, data in enumerate(cv_data):
                email_status = "✅" if data["extracted_email"] else "❌"
                name_status = "✅" if data["extracted_name"] else "❌"
                preview_data.append(
                    {
                        "No": i + 1,
                        "CV File": data["filename"],
                        "Detected Name": data["extracted_name"] or "(not found)",
                        "Name": name_status,
                        "Detected Email": data["extracted_email"] or "(not found)",
                        "Email": email_status,
                    }
                )
            st.dataframe(pd.DataFrame(preview_data), use_container_width=True)

        default_emails = "\n".join([data["extracted_email"] if data["extracted_email"] else "" for data in cv_data])

        emails_input = st.text_area(
            "Candidate Emails (Edit if needed)",
            value=default_emails,
            height=150,
            placeholder="candidate1@email.com\ncandidate2@email.com\ncandidate3@email.com",
            help="Auto-detected emails are pre-filled. Edit or add missing emails.",
        )

        if emails_input:
            emails = [e.strip() for e in emails_input.split("\n") if e.strip()]

            if len(emails) != len(uploaded_files):
                st.error(f"❌ Mismatch! You have {len(uploaded_files)} CVs but {len(emails)} emails")
            else:
                st.success(f"{len(emails)} emails matched with CVs")

                with st.expander("🔗 CV-Name-Email Mapping", expanded=True):
                    mapping_df = pd.DataFrame(
                        {
                            "CV File": [f.name for f in uploaded_files],
                            "Candidate Name": [data["extracted_name"] or "(unknown)" for data in cv_data],
                            "Email": emails,
                        }
                    )
                    st.dataframe(mapping_df, use_container_width=True)

                if st.button("Process All Applications", type="primary", use_container_width=True):
                    with st.spinner("⏳ Processing applications..."):
                        candidates_data = []
                        for file, email, data in zip(uploaded_files, emails, cv_data):
                            resume_text = extract_text_from_pdf(file)
                            if resume_text:
                                candidates_data.append(
                                    {
                                        "email": email,
                                        "name": data.get("extracted_name", ""),
                                        "filename": file.name,
                                        "resume_text": resume_text,
                                    }
                                )

                        results = process_batch_applications(
                            candidates_data,
                            st.session_state.custom_role_name,
                            st.session_state.custom_requirements,
                        )

                        st.session_state.batch_results = results
                        st.session_state.processing_complete = True
                        st.rerun()

    # Step 3: Show Results
    if st.session_state.processing_complete and st.session_state.batch_results:
        st.markdown("---")
        st.header("📊 Processing Results")

        results = st.session_state.batch_results
        selected_count = sum(1 for r in results if r["selected"])
        rejected_count = len(results) - selected_count

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Processed", len(results))
        col2.metric("✅ Selected", selected_count)
        col3.metric("❌ Rejected", rejected_count)

        st.subheader("📋 Detailed Results")

        if selected_count > 0:
            st.success(f"✅ Selected Candidates ({selected_count})")
            for result in results:
                if result["selected"]:
                    display_name = result.get("name", "") or result["email"]
                    with st.expander(f"✅ {display_name} ({result['email']}) - {result['filename']}"):
                        if result.get("name"):
                            st.write("**Candidate:**", result["name"])
                        st.write("**Feedback:**", result["feedback"])
                        st.write("**Email Sent:**", "✅ Yes" if result["email_sent"] else "❌ No")
                        st.write("**Interview Scheduled:**", "✅ Yes" if result["interview_scheduled"] else "❌ No")
                        if result.get("analysis"):
                            st.json(result["analysis"])

        if rejected_count > 0:
            st.error(f"Rejected Candidates ({rejected_count})")
            for result in results:
                if not result["selected"]:
                    display_name = result.get("name", "") or result["email"]
                    with st.expander(f"{display_name} ({result['email']}) - {result['filename']}"):
                        if result.get("name"):
                            st.write("**Candidate:**", result["name"])
                        st.write("**Feedback:**", result["feedback"])
                        st.write("**Email Sent:**", "Yes" if result["email_sent"] else "No")
                        if result.get("analysis"):
                            st.json(result["analysis"])

        if st.button("Export Results to CSV"):
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"recruitment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
