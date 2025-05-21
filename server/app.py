from flask import Flask, request, jsonify
import os
import sys
from werkzeug.utils import secure_filename
import uuid # For unique temporary filenames if needed, though re-upload is planned

# Add utils directory to sys.path to import inference and other necessary modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

try:
    from inference import generate_report as get_initial_report
    # Import the uncertainty analysis function
    from uncertaintity_estimate_from_scratch import analyze_image as get_uncertainty_analysis
except ImportError as e:
    print(f"Error importing modules from utils: {e}")
    get_initial_report = None
    get_uncertainty_analysis = None # Add placeholder

from google import genai

app = Flask(__name__)

# Configure Gemini API Key - Using the key from expanded_report.py
GEMINI_API_KEY = "AIzaSyCdJH3uSBwWKcyrKBlMTyI3NhJ1x4TK_ak"

if not GEMINI_API_KEY:
    print("Critical Error: GEMINI_API_KEY is not set in the code.")
    gemini_client = None
else:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini client initialized successfully.") # Added for confirmation
    except Exception as e:
        print(f"Failed to initialize Gemini client with the provided API key: {e}")
        gemini_client = None


# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_expanded_report_from_text(base_text_report):
    """
    Generates an expanded medical report using Gemini based on the initial report text.
    Uses the older genai.Client() method.
    """
    if not gemini_client:
        return "Error: Gemini client not initialized. Please check API key and server logs."
    if not base_text_report:
        return "Error: No base report text provided."

    prompt = f"""Directly generate a comprehensive expansion of the following radiology report, starting with the section '1. Explanation of Complex Terms'. Do not include any introductory phrases or conversational preamble before the first section. Ensure the output strictly follows the structure:

1. **Explanation of Complex Terms**: For each medical term in the report, provide a clear, concise explanation.
2. **Expanded Findings**: Rephrase the findings in professional and easy-to-understand medical language. Add relevant technical interpretation as needed.
3. **Impression**: Summarize the clinical interpretation of the findings (4–5 sentences).
4. **Diagnosis**: Suggest 4-5 possible diagnoses based on the findings.
5. **Recommendations**: Suggest 4-5 follow-up actions, tests, or treatments.

Radiology Report:
\\\"\\\"\\\"{base_text_report}\\\"\\\"\\\"

Generate a well-structured response under each section with 4–6 sentences where appropriate.
"""
    try:
        response = gemini_client.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    except AttributeError as ae:
        if 'generate_content' in str(ae) and hasattr(gemini_client, 'models') and hasattr(gemini_client.models, 'generate_content'):
            try:
                response = gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                return response.text
            except Exception as e_inner:
                print(f"Error generating expanded report with Gemini (client.models.generate_content): {e_inner}")
                return f"Error generating expanded report: {str(e_inner)}"
        else:
            print(f"Error generating expanded report with Gemini (AttributeError): {ae}")
            return f"Error generating expanded report: {str(ae)}"

    except Exception as e:
        print(f"Error generating expanded report with Gemini: {e}")
        return f"Error generating expanded report: {str(e)}"


@app.route('/upload_image', methods=['POST'])
def handle_upload_image_for_reports(): # Renamed for clarity
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # We still save it temporarily for inference, then delete.
        # The frontend will re-send for uncertainty if needed.
        filename = secure_filename(file.filename) 
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
        try:
            file.save(temp_image_path)
        except Exception as e:
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

        if not get_initial_report:
            if os.path.exists(temp_image_path): os.remove(temp_image_path)
            return jsonify({"error": "Initial inference module not loaded."}), 500

        initial_report_tuple = get_initial_report(image_path=temp_image_path)
        if not (initial_report_tuple and initial_report_tuple[0] is not None):
            if os.path.exists(temp_image_path): os.remove(temp_image_path)
            return jsonify({"error": "Failed to generate initial report from image."}), 500
        
        initial_report_text_for_expansion = initial_report_tuple[0]

        if initial_report_text_for_expansion.startswith("ings:"):
            initial_report_text_for_expansion = "Findings:" + initial_report_text_for_expansion[len("ings:"):]
        elif initial_report_text_for_expansion.startswith("ings "): 
            initial_report_text_for_expansion = "Findings: " + initial_report_text_for_expansion[len("ings "):]
        elif initial_report_text_for_expansion.lower().startswith("findings:") and not initial_report_text_for_expansion.startswith("Findings:"):
            initial_report_text_for_expansion = "Findings:" + initial_report_text_for_expansion[len("findings:"):]
        elif initial_report_text_for_expansion.lower().startswith("impression:") and not initial_report_text_for_expansion.startswith("Impression:"):
            initial_report_text_for_expansion = "Impression:" + initial_report_text_for_expansion[len("impression:"):]

        # Generate expanded report using the (potentially corrected) initial report
        expanded_report_text = generate_expanded_report_from_text(initial_report_text_for_expansion)

        # Clean up the uploaded file after processing
        if os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except Exception as e:
                print(f"Warning: Could not remove uploaded file {temp_image_path}: {e}")

        response_payload = {
            "initial_report": initial_report_text_for_expansion, # This is the report for expansion
            "expanded_report": None,
        }

        if "Error generating expanded report" in expanded_report_text or not expanded_report_text:
            response_payload["error"] = expanded_report_text or "Failed to generate expanded report."
            return jsonify(response_payload), 500 # Send error if expansion failed
        
        response_payload["expanded_report"] = expanded_report_text
        return jsonify(response_payload)

    return jsonify({"error": "File type not allowed"}), 400

@app.route('/analyze_uncertainty', methods=['POST'])
def handle_analyze_uncertainty():
    if 'file' not in request.files:
        return jsonify({"error": "No file part for uncertainty analysis"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file for uncertainty analysis"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Save image temporarily for this specific analysis
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
        try:
            file.save(temp_image_path)
        except Exception as e:
            return jsonify({"error": f"Failed to save file for uncertainty: {str(e)}"}), 500

        scored_sentences_for_display = []
        if get_uncertainty_analysis:
            try:
                print(f"Starting on-demand uncertainty analysis for {temp_image_path}")
                scored_sentences_for_display = get_uncertainty_analysis(image_path=temp_image_path, num_samples=10)
                print(f"On-demand uncertainty analysis completed. Got {len(scored_sentences_for_display)} scored sentences.")
            except Exception as ua_exc:
                print(f"Error during on-demand uncertainty analysis: {ua_exc}")
                # Clean up even if analysis fails
                if os.path.exists(temp_image_path): 
                    try: os.remove(temp_image_path)
                    except Exception as e_rem: print(f"Warning: Could not remove temp file {temp_image_path} after ua error: {e_rem}")
                return jsonify({"error": f"Uncertainty analysis failed: {str(ua_exc)}", "scored_sentences": []}), 500
        else:
            if os.path.exists(temp_image_path): 
                try: os.remove(temp_image_path)
                except Exception as e_rem: print(f"Warning: Could not remove temp file {temp_image_path} as ua module not loaded: {e_rem}")
            return jsonify({"error": "Uncertainty analysis module not loaded.", "scored_sentences": []}), 500
        
        if os.path.exists(temp_image_path):
            try: os.remove(temp_image_path)
            except Exception as e: print(f"Warning: Could not remove temp file {temp_image_path} after ua success: {e}")

        return jsonify({"scored_sentences": scored_sentences_for_display})
    
    return jsonify({"error": "File type not allowed for uncertainty analysis"}), 400

@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    if not gemini_client:
        return jsonify({"error": "Gemini client not initialized. Please check API key and server logs."}), 500
        
    data = request.get_json()
    if not data or 'question' not in data or 'report_context' not in data:
        return jsonify({"error": "Missing question or report_context in request"}), 400

    user_question = data['question']
    report_context = data['report_context']

    qa_prompt = f"""
You are a helpful medical assistant. Based on the following expanded medical report, answer the user\'s question in a clear, medically accurate, and concise manner.

Expanded Report:
\"\"\"{report_context}\"\"\"

User Question:
\"\"\"{user_question}\"\"\"
"""
    try:
        response = gemini_client.generate_content( 
            model="gemini-2.0-flash",
            contents=qa_prompt
        )
        return jsonify({"answer": response.text})
    except AttributeError as ae:
        if 'generate_content' in str(ae) and hasattr(gemini_client, 'models') and hasattr(gemini_client.models, 'generate_content'):
            try:
                response = gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=qa_prompt
                )
                return jsonify({"answer": response.text})
            except Exception as e_inner:
                print(f"Error in chat with Gemini (client.models.generate_content): {e_inner}")
                return jsonify({"error": f"Error processing chat request: {str(e_inner)}"}), 500
        else:
            print(f"Error in chat with Gemini (AttributeError): {ae}")
            return jsonify({"error": f"Error processing chat request: {str(ae)}"}), 500
    except Exception as e:
        print(f"Error in chat with Gemini: {e}")
        return jsonify({"error": f"Error processing chat request: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001) 