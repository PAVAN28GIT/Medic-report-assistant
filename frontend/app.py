import streamlit as st
import requests
import os
from PIL import Image

# Backend API endpoints
BACKEND_UPLOAD_URL = "http://localhost:5001/upload_image" 
BACKEND_ANALYZE_UNCERTAINTY_URL = "http://localhost:5001/analyze_uncertainty"
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
    .sentence-uncertainty-item {
        margin-bottom: 8px; 
        padding: 4px 0; 
        color: #e0e0e0; 
        line-height: 1.5;
    }
    .uncertainty-score {
        font-weight: bold;
        color: #007bff; /* Blue score */
        margin-right: 10px;
        display: inline-block; /* Keep score and text on same line if possible */
    }
    .sentence-text {
        display: inline; /* Allow sentence text to flow normally */
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'uploaded_file_bytes' not in st.session_state: # Store the bytes of the uploaded file
    st.session_state.uploaded_file_bytes = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'initial_report_text' not in st.session_state: # Store the raw text string
    st.session_state.initial_report_text = None
if 'scored_sentences' not in st.session_state: 
    st.session_state.scored_sentences = []
if 'expanded_report' not in st.session_state:
    st.session_state.expanded_report = None
if 'reports_generated' not in st.session_state: # Main reports (initial & expanded)
    st.session_state.reports_generated = False
if 'uncertainty_analyzed' not in st.session_state:
    st.session_state.uncertainty_analyzed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] 
if 'user_input_for_send' not in st.session_state: 
    st.session_state.user_input_for_send = ""

MIN_EXPANDED_REPORT_LENGTH = 100 # Threshold for considering an expanded report "valid"

# --- Main App Layout ---
st.title("Radiology Report Assistant")
st.markdown("Upload an X-ray image to generate a detailed medical report and chat with our AI assistant.")

# --- File Uploader ---
# We handle the uploaded_file object directly now
uploaded_file_obj = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"], key="file_uploader")

if uploaded_file_obj is not None:
    # Display image immediately
    st.image(uploaded_file_obj, caption="Uploaded X-ray", width=300)
    # Store file bytes and name for later use (e.g., uncertainty analysis)
    st.session_state.uploaded_file_bytes = uploaded_file_obj.getvalue()
    st.session_state.uploaded_file_name = uploaded_file_obj.name

    # --- Generate Initial & Expanded Report Button ---
    if st.button("Generate Report", key="generate_reports_button"):
        if st.session_state.uploaded_file_bytes:
            with st.spinner("Generating reports this may take a while..."):
                files = {'file': (st.session_state.uploaded_file_name, st.session_state.uploaded_file_bytes, uploaded_file_obj.type)}
                try:
                    response = requests.post(BACKEND_UPLOAD_URL, files=files, timeout=120) 
                    response.raise_for_status()  
                    data = response.json()
                    
                    st.session_state.initial_report_text = data.get("initial_report")
                    st.session_state.expanded_report = data.get("expanded_report")
                    st.session_state.scored_sentences = [] # Reset if new report is generated
                    st.session_state.uncertainty_analyzed = False # Reset flag

                    is_expanded_report_valid = (st.session_state.expanded_report and 
                                                st.session_state.expanded_report.strip() and 
                                                len(st.session_state.expanded_report.strip()) > MIN_EXPANDED_REPORT_LENGTH)
                    
                    if st.session_state.initial_report_text or is_expanded_report_valid:
                        st.session_state.reports_generated = True
                        st.session_state.chat_history = [] 
                        st.success("Report generated!")
                        if not is_expanded_report_valid:
                            st.warning("Expanded report generation failed or was incomplete.")
                    else: 
                        error_message = data.get('error', 'Report generation failed.')
                        st.error(f"Report generation failed: {error_message}")
                        st.session_state.reports_generated = False
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting for report generation: {e}")
                    st.session_state.reports_generated = False
                except Exception as e:
                    st.error(f"An unexpected error during report generation: {e}")
                    st.session_state.reports_generated = False
        else:
            st.warning("Please upload an image first.")

# --- Display Reports & Uncertainty Section ---
if st.session_state.reports_generated:
    st.markdown("---")

    # Display Initial Report (plain text or with uncertainty)
    if st.session_state.uncertainty_analyzed and st.session_state.scored_sentences:
        st.markdown("<div class='report-header'>Initial Report with Sentence Uncertainty</div>", unsafe_allow_html=True)
        with st.container():
            for i, (sentence, score) in enumerate(st.session_state.scored_sentences):
                try: display_score = float(score) if score is not None else 0.0
                except (ValueError, TypeError): display_score = 0.0 
                st.markdown(f"<div class='sentence-uncertainty-item'><span class='sentence-text'>{sentence}</span> <span class='uncertainty-score'>({display_score:.4f})</span></div>", unsafe_allow_html=True)
    elif st.session_state.initial_report_text and st.session_state.initial_report_text.strip():
        st.markdown("<div class='report-header'>Initial (Raw) Report</div>", unsafe_allow_html=True)
        st.text_area("Initial Report Content", st.session_state.initial_report_text, height=150, disabled=True, key="initial_report_display_text")
    
    # Button to trigger uncertainty analysis
    if st.session_state.initial_report_text and not st.session_state.uncertainty_analyzed:
        if st.button("Analyze Sentence Uncertainty", key="analyze_uncertainty_button"):
            if st.session_state.uploaded_file_bytes:
                with st.spinner("Analyzing sentence uncertainty... This may take some time."):
                    files_for_ua = {'file': (st.session_state.uploaded_file_name, st.session_state.uploaded_file_bytes, uploaded_file_obj.type if uploaded_file_obj else 'application/octet-stream')}
                    try:
                        ua_response = requests.post(BACKEND_ANALYZE_UNCERTAINTY_URL, files=files_for_ua, timeout=180)
                        ua_response.raise_for_status()
                        ua_data = ua_response.json()
                        st.session_state.scored_sentences = ua_data.get("scored_sentences", [])
                        if st.session_state.scored_sentences:
                            st.session_state.uncertainty_analyzed = True
                            st.success("Sentence uncertainty analysis complete!")
                            st.rerun() # Rerun to update display with scores
                        else:
                            st.error(f"Uncertainty analysis returned no sentences. Error: {ua_data.get('error', '')}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error connecting for uncertainty analysis: {e}")
                    except Exception as e:
                        st.error(f"An error during uncertainty analysis: {e}")
            else:
                st.warning("Uploaded image data not found. Please re-upload.")
    elif st.session_state.uncertainty_analyzed:
        st.info("Sentence uncertainty has been analyzed for the initial report.")

    st.markdown("---") 

    # Display Expanded Report (if valid)
    is_expanded_report_display_valid = (st.session_state.expanded_report and 
                                        st.session_state.expanded_report.strip() and 
                                        len(st.session_state.expanded_report.strip()) > MIN_EXPANDED_REPORT_LENGTH)
    if is_expanded_report_display_valid:
        st.markdown("<div class='report-header'>Comprehensive Medical Report</div>", unsafe_allow_html=True)
        st.markdown(st.session_state.expanded_report, unsafe_allow_html=True)
            
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
            if user_input and is_expanded_report_display_valid: 
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
                st.rerun() 

            elif not is_expanded_report_display_valid:
                st.warning("Please generate a valid comprehensive report before asking questions.")
            else:
                st.warning("Please type a question.")
    elif st.session_state.initial_report_text and st.session_state.initial_report_text.strip(): 
        st.info("Comprehensive report generation failed or was incomplete. Chat is unavailable.")

elif uploaded_file_obj and not st.session_state.reports_generated: # Check uploaded_file_obj directly here
    st.info("Click 'Generate Report' to process the uploaded image.")

st.markdown("---")
st.markdown("Pavan Kumar K") 