import os
import base64
import numpy as np
import faiss
import openai
import streamlit as st
import tempfile
import re
from datetime import datetime, timedelta
import json
from email import message_from_bytes
import dateutil.parser
from typing import List, Dict, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openai import OpenAI
import speech_recognition as sr
import pyaudio
from pydub import AudioSegment
import io

# Gmail and Calendar API scopes
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/calendar'
]

# Initialize OpenAI client
client = OpenAI(api_key="your key")


# Embedding dimension for ada-002 embeddings
EMBEDDING_DIM = 1536

class EmailCalendarSystem:
    def __init__(self):
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.metadata_store = []
        self.service = None
        self.calendar_service = None
        
    def get_services(self):
        """Initialize Gmail and Calendar services."""
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'client_secret.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('gmail', 'v1', credentials=creds)
        self.calendar_service = build('calendar', 'v3', credentials=creds)
        return self.service, self.calendar_service

    def embed_text(self, email_text: str, max_chars: int = 6000) -> List[float]:
        """Create embedding for email text."""
        trimmed = email_text[:max_chars]
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=trimmed
        )
        return response.data[0].embedding

    def add_email_to_index(self, email_text: str, metadata: Dict):
        """Add email to FAISS index."""
        embedding = self.embed_text(email_text)
        self.index.add(np.array([embedding]))
        metadata_store_entry = {**metadata, "content": email_text}
        self.metadata_store.append(metadata_store_entry)

    def get_body(self, payload):
        """Extract email body from payload."""
        if 'parts' in payload:
            for part in payload['parts']:
                result = self.get_body(part)
                if result:
                    return result
        else:
            mime_type = payload.get("mimeType", "")
            body_data = payload.get("body", {}).get("data")
            if mime_type == "text/plain" and body_data:
                return base64.urlsafe_b64decode(body_data.encode('UTF-8')).decode('UTF-8')
            elif mime_type == "text/html" and body_data:
                html_content = base64.urlsafe_b64decode(body_data.encode('UTF-8')).decode('UTF-8')
                # Strip HTML tags for better text extraction
                import re
                clean_text = re.sub('<.*?>', '', html_content)
                return clean_text
        return ""

    def load_recent_emails(self, max_results: int = 30):
        """Load recent emails into the vector database."""
        if not self.service:
            self.get_services()
            
        results = self.service.users().messages().list(
            userId='me',
            labelIds=['INBOX'],
            q='-category:promotions -category:social',
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])
        
        if not messages:
            st.warning("No messages found.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, msg in enumerate(messages):
            msg_data = self.service.users().messages().get(
                userId='me', 
                id=msg['id'], 
                format='full'
            ).execute()
            
            payload = msg_data.get("payload", {})
            headers = payload.get("headers", [])

            subject = sender = date = None
            for h in headers:
                if h["name"] == "Subject":
                    subject = h["value"]
                elif h["name"] == "From":
                    sender = h["value"]
                elif h["name"] == "Date":
                    date = h["value"]

            content = self.get_body(payload).strip()

            if content:
                self.add_email_to_index(content, {
                    "from": sender, 
                    "subject": subject, 
                    "date": date,
                    "message_id": msg['id']
                })
            
            progress_bar.progress((i + 1) / len(messages))
            status_text.text(f"Processing email {i + 1}/{len(messages)}")
        
        status_text.text(f"Loaded {len(messages)} emails successfully!")

    def search_emails(self, query: str, k: int = 5) -> List[Dict]:
        """Search indexed emails by semantic similarity."""
        if self.index.ntotal == 0:
            return []
            
        query_vec = self.embed_text(query)
        D, I = self.index.search(np.array([query_vec]), k)
        results = []
        for idx in I[0]:
            if idx < len(self.metadata_store):
                results.append(self.metadata_store[idx])
        return results

    def transcribe_audio(self, audio_bytes) -> str:
        """Transcribe audio using OpenAI Whisper."""
        try:
            # Save audio bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            # Transcribe using OpenAI Whisper
            with open(tmp_file_path, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return transcript.text
        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
            return ""

    def extract_datetime_from_email(self, email_content: str) -> Optional[Dict]:
        """Extract date and time information from email content using OpenAI."""
        try:
            prompt = f"""
            Extract the meeting/interview date and time from the following email content.
            Return the information in JSON format with the following fields:
            - "date": Date in YYYY-MM-DD format
            - "time": Time in HH:MM format (24-hour)
            - "duration": Duration in minutes (estimate if not specified, default 60)
            - "timezone": Timezone if mentioned (default "UTC")
            - "location": Location if mentioned (default "")
            
            If no date/time is found, return null.
            
            Email content:
            {email_content}
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                datetime_info = json.loads(result)
                return datetime_info
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract manually
                return self.manual_datetime_extraction(email_content)
                
        except Exception as e:
            st.error(f"Error extracting datetime: {str(e)}")
            return None

    def manual_datetime_extraction(self, email_content: str) -> Optional[Dict]:
        """Manual datetime extraction as fallback."""
        # Common date patterns
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',    # YYYY/MM/DD
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
        ]
        
        # Common time patterns
        time_patterns = [
            r'\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\b',
            r'\b(\d{1,2}:\d{2})\b'
        ]
        
        found_date = None
        found_time = None
        
        for pattern in date_patterns:
            match = re.search(pattern, email_content, re.IGNORECASE)
            if match:
                found_date = match.group(1)
                break
        
        for pattern in time_patterns:
            match = re.search(pattern, email_content, re.IGNORECASE)
            if match:
                found_time = match.group(1)
                break
        
        if found_date:
            try:
                # Parse the date
                parsed_date = dateutil.parser.parse(found_date)
                result = {
                    "date": parsed_date.strftime("%Y-%m-%d"),
                    "time": found_time if found_time else "10:00",
                    "duration": 60,
                    "timezone": "UTC",
                    "location": ""
                }
                return result
            except:
                pass
        
        return None

    def create_calendar_event(self, email_data: Dict, datetime_info: Dict) -> bool:
        """Create calendar event from email data and datetime info."""
        try:
            if not self.calendar_service:
                self.get_services()
            
            # Parse datetime
            event_date = datetime_info.get("date")
            event_time = datetime_info.get("time", "10:00")
            duration = datetime_info.get("duration", 60)
            
            # Create datetime object
            start_datetime = datetime.strptime(f"{event_date} {event_time}", "%Y-%m-%d %H:%M")
            end_datetime = start_datetime + timedelta(minutes=duration)
            
            # Create event
            event = {
                'summary': email_data.get("subject", "Meeting from Email"),
                'location': datetime_info.get("location", ""),
                'description': f"Created from email from: {email_data.get('from', '')}\n\nEmail content:\n{email_data.get('content', '')[:500]}...",
                'start': {
                    'dateTime': start_datetime.isoformat(),
                    'timeZone': datetime_info.get("timezone", "UTC"),
                },
                'end': {
                    'dateTime': end_datetime.isoformat(),
                    'timeZone': datetime_info.get("timezone", "UTC"),
                },
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60},
                        {'method': 'popup', 'minutes': 10},
                    ],
                },
            }
            
            # Insert event
            event_result = self.calendar_service.events().insert(
                calendarId='primary',
                body=event
            ).execute()
            
            return True
            
        except Exception as e:
            st.error(f"Error creating calendar event: {str(e)}")
            return False

# Initialize the system
@st.cache_resource
def get_email_system():
    return EmailCalendarSystem()

def main():
    st.title("🎤 Voice-Enabled Email Calendar Assistant")
    st.write("Find emails using voice commands and create calendar invites automatically!")
    
    # Initialize system
    email_system = get_email_system()
    
    # Sidebar for loading emails
    with st.sidebar:
        st.header("📧 Email Management")
        
        if st.button("Load Recent Emails"):
            with st.spinner("Loading emails..."):
                email_system.load_recent_emails()
        
        st.write(f"📊 Total emails in database: {email_system.index.ntotal}")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Search Emails")
        
        # Text input option
        text_query = st.text_input("Enter your search query:")
        
        # Audio input option
        st.subheader("🎤 Or use voice search:")
        audio_bytes = st.audio_input("Record your search query")
        
        search_query = ""
        
        if audio_bytes:
            with st.spinner("Transcribing audio..."):
                search_query = email_system.transcribe_audio(audio_bytes.read())
                if search_query:
                    st.success(f"Transcribed: '{search_query}'")
        
        if text_query:
            search_query = text_query
    
    with col2:
        st.header("⚙️ Settings")
        max_results = st.slider("Max search results", 1, 10, 5)
    
    # Search and display results
    if search_query:
        with st.spinner("Searching emails..."):
            results = email_system.search_emails(search_query, max_results)
        
        if results:
            st.success(f"Found {len(results)} relevant emails:")
            
            # Display results with selection
            selected_email = None
            
            for i, result in enumerate(results):
                with st.expander(f"📧 {i+1}. {result.get('subject', 'No Subject')[:50]}..."):
                    st.write(f"**From:** {result.get('from', 'Unknown')}")
                    st.write(f"**Date:** {result.get('date', 'Unknown')}")
                    st.write(f"**Subject:** {result.get('subject', 'No Subject')}")
                    st.write(f"**Content Preview:** {result.get('content', '')[:300]}...")
                    
                    if st.button(f"Create Calendar Invite for Email {i+1}", key=f"create_{i}"):
                        selected_email = result
                        break
            
            # Process selected email for calendar creation
            if selected_email:
                st.header("📅 Creating Calendar Invite")
                
                with st.spinner("Extracting date and time information..."):
                    datetime_info = email_system.extract_datetime_from_email(selected_email.get('content', ''))
                
                if datetime_info:
                    st.success("✅ Date and time extracted successfully!")
                    
                    # Display extracted information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Extracted Information:**")
                        st.write(f"📅 Date: {datetime_info.get('date', 'Not found')}")
                        st.write(f"⏰ Time: {datetime_info.get('time', 'Not found')}")
                        st.write(f"⏱️ Duration: {datetime_info.get('duration', 60)} minutes")
                        st.write(f"🌍 Timezone: {datetime_info.get('timezone', 'UTC')}")
                        st.write(f"📍 Location: {datetime_info.get('location', 'Not specified')}")
                    
                    with col2:
                        st.write("**Email Information:**")
                        st.write(f"📧 Subject: {selected_email.get('subject', 'No Subject')}")
                        st.write(f"👤 From: {selected_email.get('from', 'Unknown')}")
                    
                    # Confirm calendar creation
                    if st.button("🗓️ Create Calendar Event", type="primary"):
                        with st.spinner("Creating calendar event..."):
                            success = email_system.create_calendar_event(selected_email, datetime_info)
                        
                        if success:
                            st.success("🎉 Calendar event created successfully!")
                            st.balloons()
                        else:
                            st.error("❌ Failed to create calendar event. Please check your credentials and try again.")
                else:
                    st.warning("⚠️ Could not extract date and time information from this email.")
                    st.write("Please check the email content and try again, or create the event manually.")
        else:
            st.info("No relevant emails found. Try a different search query.")

if __name__ == "__main__":
    main()