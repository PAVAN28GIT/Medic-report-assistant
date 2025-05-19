from flask import Flask, request, jsonify
import os
import sys
from werkzeug.utils import secure_filename

# Add utils directory to sys.path to import inference and other necessary modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

try:
    from inference import generate_report as get_initial_report
except ImportError as e:
    print(f"Error importing modules from utils: {e}")
    get_initial_report = None

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
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(image_path)
        except Exception as e:
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

        if not get_initial_report:
            if os.path.exists(image_path): os.remove(image_path)
            return jsonify({"error": "Inference module not loaded correctly."}), 500

        initial_report_tuple = get_initial_report(image_path=image_path)
        if not (initial_report_tuple and initial_report_tuple[0] is not None):
            if os.path.exists(image_path): os.remove(image_path)
            return jsonify({"error": "Failed to generate initial report from image."}), 500
        
        initial_report_text = initial_report_tuple[0]

        # Correct the beginning of the initial_report_text if necessary
        if initial_report_text.startswith("ings:"):
            initial_report_text = "Findings:" + initial_report_text[len("ings:"):]

        expanded_report_text = generate_expanded_report_from_text(initial_report_text)

        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"Warning: Could not remove uploaded file {image_path}: {e}")

        if "Error generating expanded report" in expanded_report_text or not expanded_report_text:
             return jsonify({
                "initial_report": initial_report_text,
                "expanded_report": None,
                "error": expanded_report_text or "Failed to generate expanded report."
            }), 500

        return jsonify({
            "initial_report": initial_report_text,
            "expanded_report": expanded_report_text
        })

    return jsonify({"error": "File type not allowed"}), 400

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