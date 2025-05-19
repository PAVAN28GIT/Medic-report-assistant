import streamlit as st
import requests
import os
from PIL import Image

# Backend API endpoints
BACKEND_UPLOAD_URL = "http://localhost:5001/upload_image" # Assuming backend runs on port 5001
BACKEND_CHAT_URL = "http://localhost:5001/chat"

st.set_page_config(layout="wide", page_title="Medical Report Generator & Chat")

# --- Styling ---
st.markdown("""
<style>
    .report-container {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .report-header {
        font-size: 24px;
        font-weight: bold;
        color: #2a7ba1; /* Corrected blue color for headings */
        margin-bottom: 15px;
        border-bottom: 2px solid #6c757d; /* Slightly darker border for headings */
        padding-bottom: 10px;
    }
    .report-section-header {
        font-size: 18px;
        font-weight: bold;
        color: #495057; /* Darker grey for sub-headings */
        margin-top: 15px;
        margin-bottom: 8px;
    }
    .report-content {
        font-size: 16px;
        line-height: 1.6;
        color: #444;
    }
    .chat-container {
        margin-top: 30px;
    }
    .user-message {
        text-align: right;
        background-color: #dcf8c6; /* Light green */
        color: #111; /* Darker text color */
        padding: 8px 12px;
        border-radius: 15px 15px 0 15px;
        margin-bottom: 10px;
        display: inline-block;
        max-width: 70%;
        float: right;
        clear: both;
    }
    .assistant-message {
        text-align: left;
        background-color: #f1f0f0; /* Light grey */
        color: #111; /* Darker text color */
        padding: 8px 12px;
        border-radius: 15px 15px 15px 0;
        margin-bottom: 10px;
        display: inline-block;
        max-width: 70%;
        float: left;
        clear: both;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'initial_report' not in st.session_state:
    st.session_state.initial_report = None
if 'expanded_report' not in st.session_state:
    st.session_state.expanded_report = None
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] 
if 'user_input_for_send' not in st.session_state: 
    st.session_state.user_input_for_send = ""

MIN_EXPANDED_REPORT_LENGTH = 100 # Threshold for considering an expanded report "valid"

# --- Main App Layout ---
st.title("Radiology Report Assistant")
st.markdown("Upload an X-ray image to generate a detailed medical report and chat with our AI assistant.")

# --- File Uploader and Report Generation ---
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.session_state.uploaded_image = uploaded_file
    st.image(uploaded_file, caption="Uploaded X-ray", width=300)

    if st.button("Generate Report", key="generate_report_button"):
        with st.spinner("Processing image and generating report... This may take a moment."):
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(BACKEND_UPLOAD_URL, files=files, timeout=120) 
                response.raise_for_status()  
                
                data = response.json()
                st.session_state.initial_report = data.get("initial_report")
                st.session_state.expanded_report = data.get("expanded_report")
                
                # Check if expanded report is substantial enough
                is_expanded_report_valid = (st.session_state.expanded_report and 
                                            st.session_state.expanded_report.strip() and 
                                            len(st.session_state.expanded_report.strip()) > MIN_EXPANDED_REPORT_LENGTH)

                if is_expanded_report_valid: 
                    st.session_state.report_generated = True
                    st.session_state.chat_history = [] 
                    st.success("Report generated successfully!")
                elif st.session_state.initial_report and st.session_state.initial_report.strip(): 
                    st.session_state.report_generated = True 
                    st.session_state.chat_history = []
                    error_message = data.get('error', 'Expanded report could not be generated or was too short.')
                    st.warning(f"Initial report generated, but comprehensive report failed or was incomplete: {error_message}")
                    st.session_state.expanded_report = None # Ensure it's None if not valid
                else: 
                    error_message = data.get('error', 'Unknown error from backend')
                    st.error(f"Failed to generate report. Backend error: {error_message}")
                    st.session_state.report_generated = False
                    st.session_state.initial_report = None # Clear if all failed
                    st.session_state.expanded_report = None

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to backend: {e}")
                st.session_state.report_generated = False
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.report_generated = False

# --- Display Reports ---
if st.session_state.report_generated:
    st.markdown("---")

    if st.session_state.initial_report and st.session_state.initial_report.strip():
        st.markdown("<div class='report-header'>Initial (Raw) Report</div>", unsafe_allow_html=True)
        st.text_area("Initial Report Content", st.session_state.initial_report, height=150, disabled=True, key="initial_report_display")
        st.markdown("---") 

    # Check for substantial expanded report again for display
    is_expanded_report_display_valid = (st.session_state.expanded_report and 
                                        st.session_state.expanded_report.strip() and 
                                        len(st.session_state.expanded_report.strip()) > MIN_EXPANDED_REPORT_LENGTH)

    if is_expanded_report_display_valid:
        st.markdown("<div class='report-header'>Comprehensive Medical Report</div>", unsafe_allow_html=True)
        st.markdown(st.session_state.expanded_report, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("<div class='report-header'>Chat with Medical Assistant</div>", unsafe_allow_html=True)
        st.markdown("Ask questions about the comprehensive report below.")

        chat_display_container = st.container()
        with chat_display_container:
            for role, message in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f"<div class='user-message'>{message}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='assistant-message'>{message}</div>", unsafe_allow_html=True)

        user_input = st.text_input("Your question:", key="chat_input", placeholder="Type your question here...", value=st.session_state.user_input_for_send)

        if st.button("Send", key="send_chat_button"):
            if user_input and is_expanded_report_display_valid: # Ensure chat only works with valid expanded report
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.user_input_for_send = "" 
                
                with chat_display_container: 
                     for role, message_text in st.session_state.chat_history:
                        if role == "user":
                            st.markdown(f"<div class='user-message'>{message_text}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='assistant-message'>{message_text}</div>", unsafe_allow_html=True)
                
                with st.spinner("Assistant is thinking..."):
                    payload = {
                        "question": user_input, 
                        "report_context": st.session_state.expanded_report
                    }
                    try:
                        chat_response = requests.post(BACKEND_CHAT_URL, json=payload, timeout=60)
                        chat_response.raise_for_status()
                        chat_data = chat_response.json()
                        assistant_reply = chat_data.get("answer", "Sorry, I couldn't get a response.")
                        st.session_state.chat_history.append(("assistant", assistant_reply))
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error communicating with chat backend: {e}")
                        st.session_state.chat_history.append(("assistant", "Error: Could not reach assistant."))
                    except Exception as e:
                        st.error(f"An unexpected error occurred during chat: {e}")
                        st.session_state.chat_history.append(("assistant", f"Error: {e}."))
                st.rerun() # CRITICAL: Ensure this is st.rerun()

            elif not is_expanded_report_display_valid:
                st.warning("Please generate a valid comprehensive report before asking questions.")
            else:
                st.warning("Please type a question.")
    elif st.session_state.initial_report and st.session_state.initial_report.strip(): 
        st.info("Comprehensive report could not be generated or was incomplete. Chat is unavailable.")

elif st.session_state.uploaded_image and not st.session_state.report_generated:
    st.info("Click 'Generate Report' to process the uploaded image.")

st.markdown("---")
st.markdown("Pavan Kumar K") 