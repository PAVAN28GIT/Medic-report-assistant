# utils/inference.py
import torch
from transformers import (
    CLIPProcessor,
    GPT2Tokenizer,
    VisionEncoderDecoderModel
)
from PIL import Image
from . import config # Explicit relative import
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure global tokenizer is available if exported or used by other functions in this module
# This instance is also used by visualize_attention.py when it imports `tokenizer as inference_tokenizer`
try:
    tokenizer = GPT2Tokenizer.from_pretrained(config.TOKENIZER_SAVE_PATH)
    print(f"[INFERENCE] Global tokenizer loaded successfully from {config.TOKENIZER_SAVE_PATH}")
except Exception as e:
    print(f"[INFERENCE_ERROR] Failed to load global tokenizer: {e}")
    tokenizer = None # Fallback, will cause issues if other modules rely on it.

print(f"[INFERENCE] Using device from config: {config.DEVICE}")

# --- Load Models and Processors (Loaded once when module is imported) ---
print(f"[INFERENCE] Loading processor from: {config.PROCESSOR_SAVE_PATH}")
processor = CLIPProcessor.from_pretrained(config.PROCESSOR_SAVE_PATH)

print(f"[INFERENCE] Loading tokenizer from: {config.TOKENIZER_SAVE_PATH}")
tokenizer = GPT2Tokenizer.from_pretrained(config.TOKENIZER_SAVE_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("[INFERENCE] Set tokenizer pad_token to eos_token for generation.")

print(f"[INFERENCE] Loading trained model from: {config.DECODER_SAVE_PATH}")
model = VisionEncoderDecoderModel.from_pretrained(config.DECODER_SAVE_PATH).to(config.DEVICE)
model.eval()

def generate_report_data(image_path: str, max_length: int = 128, num_beams: int = 4):
    """
    Generates a medical report for a given X-ray image path.
    Returns the report text, raw output IDs (tensor), cross-attentions (tuple), and PIL image.
    Loads model, processor, tokenizer internally for each call for simplicity in this example.
    For high performance, consider loading these once and passing them as arguments.
    """
    pil_image = None
    try:
        pil_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"[INFERENCE_ERROR] Image not found at {image_path}")
        return None, None, None, None
    except Exception as e:
        print(f"[INFERENCE_ERROR] Error loading image {image_path}: {e}")
        return None, None, None, None

    try:
        # Load components for this specific call
        # print(f"[INFERENCE] Loading processor from: {config.PROCESSOR_SAVE_PATH}")
        current_processor = CLIPProcessor.from_pretrained(config.PROCESSOR_SAVE_PATH)
        # print(f"[INFERENCE] Loading tokenizer from: {config.TOKENIZER_SAVE_PATH}")
        current_tokenizer = GPT2Tokenizer.from_pretrained(config.TOKENIZER_SAVE_PATH)
        # print(f"[INFERENCE] Loading model from: {config.DECODER_SAVE_PATH}")
        current_model = VisionEncoderDecoderModel.from_pretrained(config.DECODER_SAVE_PATH).to(config.DEVICE)
        current_model.eval()
    except Exception as load_e:
        print(f"[INFERENCE_ERROR] Failed to load model/processor/tokenizer: {load_e}")
        return None, None, None, pil_image # Return image if loading failed

    pixel_values = current_processor(images=pil_image, return_tensors="pt").pixel_values.to(config.DEVICE)
    
    report_text = None
    raw_output_ids = None
    cross_attentions = None

    try:
        with torch.no_grad():
            generate_output = current_model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=current_tokenizer.pad_token_id if current_tokenizer.pad_token_id is not None else current_tokenizer.eos_token_id,
                eos_token_id=current_tokenizer.eos_token_id,
                bos_token_id=current_tokenizer.bos_token_id,
                return_dict_in_generate=True, 
                output_attentions=True,       
            )
            raw_output_ids = generate_output.sequences 
            cross_attentions = generate_output.cross_attentions if hasattr(generate_output, 'cross_attentions') else None
            report_text_list = current_tokenizer.batch_decode(raw_output_ids, skip_special_tokens=True)
            report_text = report_text_list[0].strip() if report_text_list else ""

            if report_text.lower().startswith("ings:"):
                report_text = "Findings:" + report_text[len("ings:"):]
            elif report_text.lower().startswith("ings "):
                 report_text = "Findings: " + report_text[len("ings "):]
            elif report_text.lower().startswith("findings:") and not report_text.startswith("Findings:"):
                report_text = "Findings:" + report_text[len("findings:"):]
            elif report_text.lower().startswith("impression:") and not report_text.startswith("Impression:"):
                report_text = "Impression:" + report_text[len("impression:"):]

    except Exception as e:
        print(f"[INFERENCE_ERROR] Error during generation/decoding: {e}")
        return None, None, None, pil_image 

    return report_text, raw_output_ids, cross_attentions, pil_image

if __name__ == "__main__":
    print("[INFERENCE_MAIN] Running basic inference test...")
    # Example usage of this module if run directly
    # test_image_path = "image8000.jpg" # Or any other test image in the utils folder
    test_image_path = os.path.join(os.path.dirname(__file__), "image8000.jpg")


    if not os.path.exists(test_image_path):
        print(f"[INFERENCE_MAIN_ERROR] Test image not found at: {test_image_path}")
        print("Please place image8000.jpg in the utils directory or update the path.")
    else:
        report, ids, attentions, img = generate_report_data(test_image_path)

        print("\n--- Generated Report (from inference.py) ---")
        if report:
            print(report)
        else:
            print("Report generation failed.")

        if attentions:
            print(f"\n--- Cross-Attentions data was returned (Length: {len(attentions)} steps).")
        else:
            print("\n--- Cross-Attentions data was NOT returned.")
        print("-------------------------------------------")

# image8000 : 
# 1. Low lung volumes accentuate the bronchovascular markings. 
# 2. Subtle right base opacity most likely represents combination of atelectasis and vascular structures, although infection is not excluded in the appropriate clinical setting. 
# 3. Suggest dedicated PA and lateral views when patient able for better evaluation. 
# 4. No pleural effusion is seen. 
# 5. There is no pneumothorax. 