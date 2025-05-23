import torch
import numpy as np
from transformers import CLIPProcessor, GPT2Tokenizer, VisionEncoderDecoderModel
from PIL import Image
from nltk.tokenize import sent_tokenize
from collections import Counter
from scipy.stats import entropy
import nltk
import time
from . import config
import os

# --- Initialization ---
try:
    nltk.download("punkt", quiet=True)
except Exception as nltk_e:
    print(f"[UNCERTAINTY_WARN] Failed to download nltk.punkt: {nltk_e}. Sentence tokenization might fail if not already available.")

# --- Configuration (using your project's config) ---
PROCESSOR_PATH = config.PROCESSOR_SAVE_PATH
TOKENIZER_PATH = config.TOKENIZER_SAVE_PATH
MODEL_PATH = config.DECODER_SAVE_PATH
DEVICE = config.DEVICE

# --- Load components ---
print("[UNCERTAINTY_INFO] Loading processor, tokenizer, and model for uncertainty module...")
try:
    processor = CLIPProcessor.from_pretrained(PROCESSOR_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    print("[UNCERTAINTY_INFO] Uncertainty components loaded successfully.")
except Exception as load_e:
    print(f"[UNCERTAINTY_ERROR] Failed to load components for uncertainty module: {load_e}")
    processor = None
    tokenizer = None
    model = None

# --- Enable dropout at inference ---
if model:
    for module_component in model.modules():
        if module_component.__class__.__name__.startswith('Dropout'):
            module_component.train()
else:
    print("[UNCERTAINTY_WARN] Model not loaded, cannot enable MC Dropout globally for uncertainty module.")

# --- Generate N stochastic reports ---
def generate_mc_samples(image_path, num_samples=5, max_length=128):
    if not all([processor, tokenizer, model]):
        print("[UNCERTAINTY_ERROR] MC Sampling cannot proceed: one or more critical components (processor, tokenizer, model) failed to load.")
        return []
        
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"[UNCERTAINTY_ERROR] Image not found: {image_path}")
        return []
    except Exception as img_e:
        print(f"[UNCERTAINTY_ERROR] Error loading image {image_path}: {img_e}")
        return []

    inputs = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

    for module_component in model.modules():
        if module_component.__class__.__name__.startswith('Dropout'):
            module_component.train()

    reports = []
    for i in range(num_samples):
        with torch.no_grad():
            output_ids = model.generate(
                inputs,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        reports.append(decoded)
    return reports

# --- Compute entropy per sentence position ---
def sentence_level_uncertainty(reports):
    if not reports:
        return []
    sentence_lists = [sent_tokenize(r) for r in reports if r]
    if not any(sentence_lists):
        return []

    max_len = max(len(lst) for lst in sentence_lists if lst) if sentence_lists else 0
    if max_len == 0:
        return []

    for lst in sentence_lists:
        lst.extend(["<PAD_SENTENCE>"] * (max_len - len(lst)))

    sentence_columns = list(zip(*sentence_lists))
    sentence_scores = []

    for idx, group in enumerate(sentence_columns):
        valid_sentences_in_group = [s for s in group if s != "<PAD_SENTENCE>"]
        display_sentence = next((s for s in group if s != "<PAD_SENTENCE>"), group[0] if group else "[No Sentence]")

        if not valid_sentences_in_group:
            norm_ent = 1.0
        else:
            counts = Counter(valid_sentences_in_group)
            probs = np.array(list(counts.values())) / len(valid_sentences_in_group)
            ent = entropy(probs)
            norm_ent = ent / np.log(len(counts)) if len(counts) > 1 else 0.0

        sentence_scores.append((display_sentence, norm_ent))
    return sentence_scores

# --- Full Pipeline ---
def analyze_image(image_path, num_samples=5):
    if not all([processor, tokenizer, model]):
        print("[UNCERTAINTY_ERROR] Full analysis cannot proceed: critical components not loaded.")
        return []
    samples = generate_mc_samples(image_path, num_samples=num_samples)
    if not samples:
        print("[UNCERTAINTY_ERROR] Failed to generate MC samples for full analysis.")
        return []
    scored_sentences = sentence_level_uncertainty(samples)
    return scored_sentences

# --- Main ---
if __name__ == "__main__":
    if not all([processor, tokenizer, model]):
        print("[UNCERTAINTY_MAIN_ERROR] Cannot run standalone test: critical components failed to load during module import.")
    else:
        test_image_path = os.path.join(os.path.dirname(__file__), "image1.jpg") 
        if not os.path.exists(test_image_path):
            print(f"[UNCERTAINTY_MAIN_ERROR] Test image not found: {test_image_path}.")
        else:
            analyze_image(test_image_path, num_samples=10)
