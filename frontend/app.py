import streamlit as st
import requests
import os
from PIL import Image
import io # Needed for sending bytes as a file

# Backend API endpoints
BACKEND_UPLOAD_URL = "http://localhost:5001/upload_image" 
BACKEND_ANALYZE_UNCERTAINTY_URL = "http://localhost:5001/analyze_uncertainty"
BACKEND_VISUALIZE_ATTENTION_URL = "http://localhost:5001/visualize_attention"
BACKEND_CHAT_URL = "http://localhost:5001/chat"

st.set_page_config(layout="wide", page_title="Radiology Report AI Assistant - Attention Viz")

# --- Final Polish Styling ---
st.markdown("""
<style>
    /* --- General Page & Theme --- */
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        background-image: linear-gradient(135deg, #737db2 0%, #525c89 100%); /* Slightly desaturated, darker gradient */
        color: #e0e0e0; /* Lighter default text for dark bg */
        margin: 0; /* Remove default body margin */
        padding: 0; /* Remove default body padding */
    }
    .stApp {
        background-color: transparent; 
    }
    .main .block-container {
        padding-top: 1rem; /* Reduce default top padding */
        padding-bottom: 1rem;
    }

    h1 {
        color: #ffffff; 
        font-weight: 700;
        text-align: center;
        padding-top: 30px;
        padding-bottom: 10px; 
    }
    .subtitle {
        text-align:center; 
        font-size: 1.1em; 
        color: #c0c8d8; /* Lighter subtitle for dark bg */
        margin-bottom:25px;
        font-weight: 400;
    }
    .stSpinner > div > div {
        border-top-color: #82aaff; /* Lighter blue for spinner on dark bg */
    }

    /* --- Card Layout --- */
    .card {
        background-color: rgba(40, 45, 70, 0.7); /* Darker, translucent card background */
        backdrop-filter: blur(10px); 
        -webkit-backdrop-filter: blur(10px); 
        border-radius: 12px; 
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.25); 
        border: 1px solid rgba(255, 255, 255, 0.1); 
    }
    .upload-card-content { 
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 10px 0; /* Reduce padding if no image is shown */
    }

    /* --- Headers inside Cards --- */
    .report-header {
        font-size: 20px; 
        font-weight: 600;
        color: #e8eaf6; 
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2); 
        text-align: left;
    }

    /* --- Info/Warning/Error Message Styling --- */
    .stAlert {
        border-radius: 6px;
        padding: 12px;
        font-size: 1em;
        color: #1a1a2e; /* Dark text for alerts for contrast */
    }
    div[data-baseweb="toast"][data-testid="stNotification"] {
        background-color: #303952 !important; /* Darker info box */
        color: #e0e0e0 !important; /* Light text for info box */
        border-left: 4px solid #82aaff !important;
    }
    div[data-baseweb="toast"][data-testid="stNotification"] div[data-testid="stText"] {
         color: #e0e0e0 !important; 
    }

    /* --- Report Content Areas --- */
    .report-content-area { 
        background-color: rgba(20, 25, 50, 0.5); 
        border-radius: 6px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        min-height: 120px; 
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        color: #c0c8d8;
        font-size: 0.9em;
        white-space: pre-wrap; 
        word-wrap: break-word;
    }
    .expanded-report-content {
        padding-top: 10px;
        color: #d0d8e8; 
    }
    .expanded-report-content p, .expanded-report-content li {
        font-size: 1em;
        line-height: 1.65;
        color: #d0d8e8;
    }
    .expanded-report-content h1, .expanded-report-content h2, .expanded-report-content h3, 
    .expanded-report-content h4, .expanded-report-content strong {
        color: #e8eaf6; 
        font-weight: 600;
        margin-top: 0.8em;
        margin-bottom: 0.4em;
    }

    /* --- Sentence Uncertainty List --- */
    .sentence-uncertainty-item {
        margin-bottom: 8px;
        padding: 8px 0;
        color: #c0c8d8; 
        line-height: 1.55;
        border-bottom: 1px dashed rgba(255, 255, 255, 0.15); 
        display: flex; 
        align-items: flex-start;
    }
    .sentence-uncertainty-item:last-child {
        border-bottom: none;
    }
    .uncertainty-score {
        font-weight: 700;
        color: #82aaff; 
        margin-right: 10px;
        background-color: rgba(130, 170, 255, 0.15);
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.8em;
        white-space: nowrap; 
    }
    .sentence-text {
        flex-grow: 1; 
        color: #c0c8d8; 
    }

    /* --- Chat Interface --- */
    .chat-header {
        font-size: 18px;
        font-weight: 600;
        color: #e8eaf6;
        margin-bottom: 15px;
    }
    .user-message {
        background-image: linear-gradient(to right, #82aaff, #5a7fdc);
        color: #0A122A; /* Dark text on light blue gradient */
        padding: 10px 15px;
        border-radius: 16px 16px 3px 16px; 
        margin-bottom: 10px;
        max-width: 75%;
        float: right;
        clear: both;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        word-wrap: break-word;
    }
    .assistant-message {
        background-color: rgba(50, 60, 90, 0.8); 
        color: #d0d8e8;
        padding: 10px 15px;
        border-radius: 16px 16px 16px 3px; 
        margin-bottom: 10px;
        max-width: 75%;
        float: left;
        clear: both;
        box-shadow: 0 2px 5px rgba(0,0,0,0.15);
        word-wrap: break-word;
    }
    .stTextInput > div > div > input {
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 10px 12px;
        font-size: 0.95em;
        background-color: rgba(30, 35, 60, 0.7);
        color: #e0e0e0;
    }
    .stTextInput > div > div > input::placeholder {
        color: #9098b0;
    }
    .stButton > button {
        border-radius: 6px;
        background-image: linear-gradient(to right, #82aaff, #5a7fdc); /* Default gradient */
        color: #0A122A; /* Default dark text */
        padding: 10px 22px; 
        font-size: 1em; 
        font-weight: 600; 
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out, transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; /* Added background-color to transition */
        cursor: pointer;
    }
    .stButton > button:hover {
        background-image: none; 
        background-color: #4a69bd; 
        color: #000000; 
        font-size: 1.05em; 
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transform: translateY(-1px); 
    }
    .stButton > button:active {
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important; /* Inset shadow for pressed look */
        transform: translateY(1px) !important; /* Slight press down effect */
     }
    .heatmap-gallery img {
        width: 100%; /* Make heatmap images responsive within their container */
        max-width: 250px; /* Max size for each heatmap image */
        height: auto;
        margin: 5px;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization (ensure all relevant keys are present) ---
def init_session_state():
    defaults = {
        'uploaded_file_bytes': None,
        'uploaded_file_name': None,
        'uploaded_file_type': None,
        'original_image_filename': None,
        'initial_report_text': None,
        'scored_sentences': [],
        'expanded_report': None,
        'reports_generated': False,
        'uncertainty_analyzed': False,
        'attention_visualization_term': '',
        'attention_heatmaps': [],
        'chat_history': [],
        'user_input_for_send': ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

MIN_EXPANDED_REPORT_LENGTH = 100 

# --- Main App Layout ---
st.markdown("<h1><img src='https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/microscope.svg' style='height:36px; margin-right:10px; vertical-align:bottom; filter: invert(90%) sepia(15%) saturate(300%) hue-rotate(190deg) brightness(110%) contrast(95%);'>Radiology Report AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an X-ray image for AI-powered analysis, report generation, and interactive chat.</p>", unsafe_allow_html=True)

# --- Main Content Area in a Single Column for better flow control with cards ---
with st.container(): # Main container for all content cards
    # --- Upload and Generate Card ---
    with st.container():
        st.markdown("<div class='upload-card-content'>", unsafe_allow_html=True)
        if not st.session_state.reports_generated and not st.session_state.uploaded_file_bytes:
            st.markdown("<h2 class='report-header' style='text-align:center; width:100%; color: #e0e0e0;'>Upload X-Ray Image</h2>", unsafe_allow_html=True)
        
        # Center the file uploader using columns hack or by styling its container
        col_uploader_1, col_uploader_2, col_uploader_3 = st.columns([1,1.8,1])
        with col_uploader_2:
            uploaded_file_obj = st.file_uploader("X-ray image uploader for analysis", type=["png", "jpg", "jpeg"], key="file_uploader", label_visibility="collapsed")

        if uploaded_file_obj is not None:
            col_img_1, col_img_2, col_img_3 = st.columns([0.5,2,0.5])
            with col_img_2:
                 st.image(uploaded_file_obj, caption="Uploaded X-ray", use_container_width=True)
            
            st.session_state.uploaded_file_bytes = uploaded_file_obj.getvalue()
            st.session_state.uploaded_file_name = uploaded_file_obj.name
            st.session_state.uploaded_file_type = uploaded_file_obj.type

            if st.button("Generate Report", key="generate_reports_button", use_container_width=True):
                if st.session_state.uploaded_file_bytes:
                    with st.spinner("Generating reports... this may take a moment."):
                        files = {'file': (st.session_state.uploaded_file_name, st.session_state.uploaded_file_bytes, st.session_state.uploaded_file_type)}
                        try:
                            response = requests.post(BACKEND_UPLOAD_URL, files=files, timeout=120) 
                            response.raise_for_status()  
                            data = response.json()
                            
                            st.session_state.original_image_filename = data.get("original_image_filename")
                            st.session_state.initial_report_text = data.get("initial_report")
                            st.session_state.expanded_report = data.get("expanded_report")
                            st.session_state.scored_sentences = [] 
                            st.session_state.uncertainty_analyzed = False 
                            st.session_state.attention_heatmaps = []
                            st.session_state.attention_visualization_term = ''

                            is_expanded_report_valid = (st.session_state.expanded_report and 
                                                        st.session_state.expanded_report.strip() and 
                                                        len(st.session_state.expanded_report.strip()) > MIN_EXPANDED_REPORT_LENGTH)
                            
                            if st.session_state.initial_report_text or is_expanded_report_valid:
                                st.session_state.reports_generated = True
                                st.session_state.chat_history = [] 
                                st.success("Reports generated successfully!")
                                if not is_expanded_report_valid and st.session_state.initial_report_text:
                                    st.warning("Comprehensive report generation failed or was incomplete.")
                            else: 
                                error_message = data.get('error', 'Report generation failed.')
                                st.error(f"Report generation failed: {error_message}")
                                st.session_state.reports_generated = False
                        except requests.exceptions.RequestException as e:
                            st.error(f"Connection error for report generation: {e}")
                            st.session_state.reports_generated = False
                        except Exception as e:
                            st.error(f"Unexpected error during report generation: {e}")
                            st.session_state.reports_generated = False
                else:
                    st.warning("Please upload an image first to generate a report.")

    # --- Initial Report Card (Raw or with Uncertainty) ---
    if st.session_state.reports_generated and (st.session_state.initial_report_text or st.session_state.scored_sentences):
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if st.session_state.uncertainty_analyzed and st.session_state.scored_sentences:
                st.markdown("<h2 class='report-header'>Initial Report & Sentence Uncertainty</h2>", unsafe_allow_html=True)
                for i, (sentence, score) in enumerate(st.session_state.scored_sentences):
                    try: display_score = float(score) if score is not None else 0.0
                    except (ValueError, TypeError): display_score = 0.0 
                    st.markdown(f"<div class='sentence-uncertainty-item'><span class='sentence-text'>{sentence}</span> <span class='uncertainty-score'>({display_score:.4f})</span></div>", unsafe_allow_html=True)
            elif st.session_state.initial_report_text and st.session_state.initial_report_text.strip():
                st.markdown("<h2 class='report-header'>Initial (Raw) Report</h2>", unsafe_allow_html=True)
                st.markdown(f"<div class='report-content-area'>{st.session_state.initial_report_text}</div>", unsafe_allow_html=True)
            
            if st.session_state.initial_report_text and not st.session_state.uncertainty_analyzed:
                st.markdown("<br>", unsafe_allow_html=True) # Add some space before the button
                if st.button("Analyze Sentence Uncertainty", key="analyze_uncertainty_button", use_container_width=True):
                    if st.session_state.uploaded_file_bytes:
                        with st.spinner("Analyzing sentence uncertainty... This may take some time."):
                            files_for_ua = {'file': (st.session_state.uploaded_file_name, st.session_state.uploaded_file_bytes, st.session_state.uploaded_file_type)}
                            try:
                                ua_response = requests.post(BACKEND_ANALYZE_UNCERTAINTY_URL, files=files_for_ua, timeout=180)
                                ua_response.raise_for_status()
                                ua_data = ua_response.json()
                                st.session_state.scored_sentences = ua_data.get("scored_sentences", [])
                                if st.session_state.scored_sentences:
                                    st.session_state.uncertainty_analyzed = True
                                    st.success("Sentence uncertainty analysis complete!")
                                    st.rerun() 
                                else:
                                    st.error(f"Uncertainty analysis returned no data or an error occurred. Details: {ua_data.get('error', '')}")
                            except requests.exceptions.RequestException as e:
                                st.error(f"Connection error for uncertainty analysis: {e}")
                            except Exception as e:
                                st.error(f"Error during uncertainty analysis: {e}")
                    else:
                        st.warning("Uploaded image data not found. Please re-upload.")
            elif st.session_state.uncertainty_analyzed:
                st.markdown("<br>", unsafe_allow_html=True)
                st.success("Sentence uncertainty has been analyzed for this report.")
            st.markdown("</div>", unsafe_allow_html=True) # Close Initial Report Card

    # --- Attention Visualization Section ---
    if st.session_state.initial_report_text and st.session_state.original_image_filename:
        st.markdown("--- Pavan Kumar K ") # Visual separator
        st.markdown("<h3 class='report-header'>Attention Map Visualization</h3>", unsafe_allow_html=True)
        medical_term_input = st.text_input("Enter medical term to visualize attention:", key="medical_term_viz", value=st.session_state.attention_visualization_term)
        if st.button("Visualize Attention for Term", key="visualize_attention_button", use_container_width=True):
            if medical_term_input.strip():
                st.session_state.attention_visualization_term = medical_term_input.strip()
                with st.spinner(f"Generating attention heatmaps for '{st.session_state.attention_visualization_term}'..."):
                    payload = {
                        "image_filename": st.session_state.original_image_filename,
                        "medical_term": st.session_state.attention_visualization_term
                    }
                    try:
                        viz_response = requests.post(BACKEND_VISUALIZE_ATTENTION_URL, json=payload, timeout=120)
                        viz_response.raise_for_status()
                        viz_data = viz_response.json()
                        st.session_state.attention_heatmaps = viz_data.get("heatmaps", [])
                        if st.session_state.attention_heatmaps:
                            st.success(viz_data.get("message", "Heatmaps generated!"))
                        else:
                            st.warning(viz_data.get("message", "No heatmaps generated or term not found."))
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection error for attention visualization: {e}")
                        st.session_state.attention_heatmaps = []
                    except Exception as e:
                        st.error(f"Error during attention visualization: {e}")
                        st.session_state.attention_heatmaps = []
            else:
                st.warning("Please enter a medical term to visualize.")

        if st.session_state.attention_heatmaps:
            st.markdown(f"Displaying attention for: **{st.session_state.attention_visualization_term}**")
            # Display heatmaps in columns for better layout if multiple are returned
            # For simplicity, let's display up to 3 in a row.
            cols = st.columns(min(len(st.session_state.attention_heatmaps), 3))
            for i, heatmap_url in enumerate(st.session_state.attention_heatmaps):
                with cols[i % 3]: # Cycle through columns
                    # The URL from backend should be directly usable if Flask serves static files correctly
                    st.image(heatmap_url, caption=f"Heatmap {i+1}", use_container_width=True, output_format='PNG') 

    # --- Expanded Report & Chat Card (Combined or Separate) ---
    if st.session_state.reports_generated:
        is_expanded_report_display_valid = (st.session_state.expanded_report and 
                                            st.session_state.expanded_report.strip() and 
                                            len(st.session_state.expanded_report.strip()) > MIN_EXPANDED_REPORT_LENGTH)
        if is_expanded_report_display_valid:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h2 class='report-header'>Comprehensive Medical Report</h2>", unsafe_allow_html=True)
                st.markdown(f"<div class='expanded-report-content'>{st.session_state.expanded_report}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True) # Close Expanded Report Card

            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='chat-header'>Chat with Medical Assistant</h3>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:0.95em; color:#b0b8c0; margin-top:-15px; margin-bottom:20px;'>Ask questions about the comprehensive report.</p>", unsafe_allow_html=True)
                
                # Chat History Display
                chat_messages_container = st.container() # Explicit container for chat messages
                # Consider setting a max height for scrolling: style='max-height: 400px; overflow-y: auto; margin-bottom: 15px;'
                with chat_messages_container:
                    for role, message in st.session_state.chat_history:
                        if role == "user":
                            st.markdown(f"<div class='user-message'>{message}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='assistant-message'>{message}</div>", unsafe_allow_html=True)
                
                # Chat Input - Placed after messages for natural flow
                user_input_placeholder = "Type your question here..."
                # Provide a descriptive label for accessibility, hide it visually
                user_input = st.text_input("User question input for chat", key="chat_input", placeholder=user_input_placeholder, value=st.session_state.user_input_for_send, label_visibility="collapsed")

                if st.button("Send", key="send_chat_button", use_container_width=True):
                    if user_input and is_expanded_report_display_valid: 
                        st.session_state.chat_history.append(("user", user_input))
                        st.session_state.user_input_for_send = "" 
                        # No immediate redraw here, let st.rerun() handle it after API call
                        
                        with st.spinner("Assistant is thinking..."):
                            payload = {"question": user_input, "report_context": st.session_state.expanded_report}
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
                st.markdown("</div>", unsafe_allow_html=True) # Close Chat Card

        elif st.session_state.initial_report_text and st.session_state.initial_report_text.strip(): 
             with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.info("Comprehensive report generation failed or was incomplete. Chat is unavailable.")
                st.markdown("</div>", unsafe_allow_html=True)

if not uploaded_file_obj and not st.session_state.reports_generated:
    with st.container():
        st.info("Upload an image to begin the analysis process.")

st.markdown("<br><p style='text-align:center; color: #c0c8d8; font-size: 0.9em; padding: 20px 0;'> - MedicReport AI Assistant</p>", unsafe_allow_html=True) 