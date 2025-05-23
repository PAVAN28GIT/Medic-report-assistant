from flask import Flask, request, jsonify, url_for
import os
import sys
from werkzeug.utils import secure_filename
import uuid

# --- Add project root to sys.path to allow imports from utils ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# --- End sys.path modification ---

try:
    # Ensure the function in inference.py is named generate_report_data
    from utils.inference import generate_report_data, tokenizer as inference_tokenizer
    from utils.uncertaintity_estimate_from_scratch import analyze_image as get_uncertainty_analysis
    # Import the new visualization function
    from utils.visualize_attention import process_and_visualize_medical_terms 
except ImportError as e:
    print(f"[SERVER_ERROR] Error importing modules from utils: {e}.")
    generate_report_data = None
    get_uncertainty_analysis = None
    process_and_visualize_medical_terms = None
    inference_tokenizer = None

from google import genai

app = Flask(__name__, static_folder='static') # Ensure static folder is correctly named

# Configure Gemini API Key
GEMINI_API_KEY = "AIzaSyCdJH3uSBwWKcyrKBlMTyI3NhJ1x4TK_ak" # Keep your actual key
gemini_client = None # For older API style
gemini_model_for_expansion_name = "gemini-1.5-flash-latest" # Default, can be changed

# Define the base URL for your backend server
# This assumes your Flask backend is running on localhost:5001
# In a production environment, this would be your actual domain/IP and port.
BACKEND_BASE_URL = "http://localhost:5001"

if not GEMINI_API_KEY:
    print("[SERVER_CRITICAL] GEMINI_API_KEY is not set.")
else:
    try:
        # Prioritize the older genai.Client() as it was working for the user
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("[SERVER_INFO] Gemini initialized successfully with genai.Client() API.")
    except AttributeError:
        print("[SERVER_WARN] genai.Client() not found or failed. Trying newer genai.configure API...")
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            # Ensure a compatible model name for the newer API if this path is taken
            gemini_model_instance = genai.GenerativeModel('gemini-1.5-flash-latest') 
            print("[SERVER_INFO] Gemini initialized successfully with new API (genai.configure).")
            # Adapt gemini_client to use the new model instance if needed, or handle calls separately
            # For simplicity, if new API works, we'll redefine how generate_gemini_content works
            # This part would need a flag or different function for the new API style.
            # However, the primary goal now is to fix the regression, so we focus on genai.Client working.
        except Exception as e_new_api:
            print(f"[SERVER_CRITICAL] Failed to initialize Gemini with new API as well: {e_new_api}")
            gemini_client = None # Ensure it's None if all attempts fail
    except Exception as e_client_init:
        print(f"[SERVER_CRITICAL] Failed to initialize Gemini with genai.Client(): {e_client_init}")
        gemini_client = None

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'server', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configuration for generated heatmaps
HEATMAP_WEB_ROOT = "generated_heatmaps"  # Relative to static folder for URL
ABSOLUTE_HEATMAP_SAVE_DIR = os.path.join(app.static_folder, HEATMAP_WEB_ROOT)
os.makedirs(ABSOLUTE_HEATMAP_SAVE_DIR, exist_ok=True)


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_gemini_content(prompt_text):
    if not gemini_client:
        return "Error: Gemini client (older API) is not initialized."
    try:
        # Use client.models.generate_content as per user's working expanded_report.py
        response = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=prompt_text)
        return response.text 
    except Exception as e:
        print(f"[SERVER_ERROR] Error in generate_gemini_content with client.models: {e}")
        return f"Error generating content with Gemini (client.models): {str(e)}"

def generate_expanded_report_from_text(base_text_report):
    if not base_text_report:
        return "Error: No base report text provided for expansion."
    prompt = f"""Directly generate a comprehensive expansion of the following radiology report, starting with the section '1. Explanation of Complex Terms'. Do not include any introductory phrases or conversational preamble before the first section. Ensure the output strictly follows the structure:

1. **Explanation of Complex Terms**: For each medical term in the report, provide a clear, concise explanation.
2. **Expanded Findings**: Rephrase the findings in professional and easy-to-understand medical language. Add relevant technical interpretation as needed.
3. **Impression**: Summarize the clinical interpretation of the findings (4–5 sentences).
4. **Diagnosis**: Suggest 4-5 possible diagnoses based on the findings.
5. **Recommendations**: Suggest 4-5 follow-up actions, tests, or treatments.

Radiology Report:
\"\"\"{base_text_report}\"\"\"

Generate a well-structured response under each section with 4–6 sentences where appropriate.
"""
    return generate_gemini_content(prompt)


@app.route('/upload_image', methods=['POST'])
def handle_upload_image_for_reports():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        # Save to UPLOAD_FOLDER with original name for potential reuse by visualize_attention
        # This file will NOT be deleted by this endpoint immediately.
        # A cleanup strategy for UPLOAD_FOLDER might be needed for long-running apps.
        persistent_image_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        try:
            file.save(persistent_image_path)
        except Exception as e:
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

        if generate_report_data is None:
            # No need to delete persistent_image_path here if main inference fails
            return jsonify({"error": "Core inference module not loaded."}), 500

        initial_report_text, _, _, _ = generate_report_data(image_path=persistent_image_path)
        
        if not initial_report_text:
            return jsonify({"error": "Failed to generate initial report from image."}), 500
                
        expanded_report_text = generate_expanded_report_from_text(initial_report_text)

        response_payload = {
            "message": "Image processed successfully.",
            "original_image_filename": original_filename, # Send back original filename
            "initial_report": initial_report_text,
            "expanded_report": None
        }
        if "Error generating expanded report" in expanded_report_text or not expanded_report_text:
            response_payload["error_expanding"] = expanded_report_text or "Failed to generate expanded report."
        else:
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
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
        try:
            file.save(temp_image_path)
        except Exception as e:
            return jsonify({"error": f"Failed to save file for uncertainty: {str(e)}"}), 500

        scored_sentences_for_display = []
        if get_uncertainty_analysis:
            try:
                print(f"Starting on-demand uncertainty analysis for {temp_image_path}")
                # Assuming get_uncertainty_analysis is adapted to take image_path
                scored_sentences_for_display = get_uncertainty_analysis(image_path=temp_image_path, num_samples=10) # Pass image path
                print(f"On-demand uncertainty analysis completed. Got {len(scored_sentences_for_display)} scored sentences.")
            except Exception as ua_exc:
                print(f"Error during on-demand uncertainty analysis: {ua_exc}")
                if os.path.exists(temp_image_path): os.remove(temp_image_path)
                return jsonify({"error": f"Uncertainty analysis failed: {str(ua_exc)}", "scored_sentences": []}), 500
        else:
            if os.path.exists(temp_image_path): os.remove(temp_image_path)
            return jsonify({"error": "Uncertainty analysis module not loaded.", "scored_sentences": []}), 500
        
        if os.path.exists(temp_image_path): os.remove(temp_image_path)
        return jsonify({"scored_sentences": scored_sentences_for_display})
    
    return jsonify({"error": "File type not allowed for uncertainty analysis"}), 400


# --- NEW ENDPOINT for Attention Visualization ---
@app.route('/visualize_attention', methods=['POST'])
def visualize_attention_endpoint():
    data = request.get_json()
    image_filename_original = data.get('image_filename') 
    medical_term = data.get('medical_term')

    if not image_filename_original or not medical_term:
        return jsonify({"error": "Missing image_filename or medical_term"}), 400

    image_path_on_server = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_filename_original))
    base_img_name_for_paths = os.path.splitext(secure_filename(image_filename_original))[0]

    if not os.path.exists(image_path_on_server):
        return jsonify({"error": f"Image '{image_filename_original}' not found on server."}), 404

    if not all([generate_report_data, process_and_visualize_medical_terms, inference_tokenizer]):
        return jsonify({"error": "Visualization components not loaded on server."}), 500

    _report_text, raw_ids, attentions, img_pil = generate_report_data(image_path=image_path_on_server)

    if not img_pil or attentions is None or raw_ids is None: 
        return jsonify({"error": "Failed to get necessary data for attention visualization."}), 500

    heatmap_web_paths_relative_to_web_heatmap_root = process_and_visualize_medical_terms(
        img_pil, attentions, raw_ids,
        inference_tokenizer,
        [medical_term], 
        base_img_name_for_paths, 
        absolute_heatmaps_root_dir=ABSOLUTE_HEATMAP_SAVE_DIR 
    )

    if not heatmap_web_paths_relative_to_web_heatmap_root:
        return jsonify({"message": f"Term '{medical_term}' not found or no heatmaps generated.", "heatmaps": []})

    final_heatmap_urls = []
    for rel_path in heatmap_web_paths_relative_to_web_heatmap_root:
        # rel_path is like "image8000/pneumothorax_occ1_token0_pos19.png"
        # We construct the full URL by prepending the backend base URL and the static path component.
        # The path from HEATMAP_WEB_ROOT onwards is what url_for would typically generate as the filename part.
        full_url = f"{BACKEND_BASE_URL}/static/{HEATMAP_WEB_ROOT}/{rel_path.replace(os.sep, '/')}"
        final_heatmap_urls.append(full_url)
        
    print(f"[SERVER /visualize_attention] Returning full heatmap URLs: {final_heatmap_urls}")
    return jsonify({"heatmaps": final_heatmap_urls, "term": medical_term, "message": f"Attention heatmaps generated for '{medical_term}'."})


@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    if not gemini_model_for_expansion_name: # Check the model used for expansion
        return jsonify({"error": "Gemini model for chat not initialized."}), 500
        
    data = request.get_json()
    if not data or 'question' not in data or 'report_context' not in data:
        return jsonify({"error": "Missing question or report_context"}), 400

    user_question = data['question']
    report_context = data['report_context']
    qa_prompt = f"""
You are a helpful medical assistant. Based on the following expanded medical report, answer the user's question in a clear, medically accurate, and concise manner.

Expanded Report:
\"\"\"{report_context}\"\"\"

User Question:
\"\"\"{user_question}\"\"\"
"""
    try:
        answer_text = generate_gemini_content(qa_prompt)
        if "Error:" in answer_text:
            return jsonify({"error": answer_text}), 500
        return jsonify({"answer": answer_text})
    except Exception as e:
        print(f"Error in chat with Gemini: {e}")
        return jsonify({"error": f"Error processing chat request: {str(e)}"}), 500

if __name__ == '__main__':
    if not all([generate_report_data, process_and_visualize_medical_terms, inference_tokenizer, get_uncertainty_analysis, gemini_client is not None]):
        print("[SERVER_CRITICAL] Core modules or the primary Gemini client (genai.Client) failed to load. Server cannot start effectively.")
    else:
        print(f"[SERVER_INFO] Uploads retained in: {UPLOAD_FOLDER}")
        print(f"[SERVER_INFO] Heatmaps saved to: {ABSOLUTE_HEATMAP_SAVE_DIR}")
        print(f"[SERVER_INFO] Heatmaps served from: {app.static_folder}/{HEATMAP_WEB_ROOT}")
        app.run(host='0.0.0.0', port=5001, debug=True) 