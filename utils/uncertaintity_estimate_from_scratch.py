import torch
import numpy as np
from transformers import CLIPProcessor, GPT2Tokenizer, VisionEncoderDecoderModel
from PIL import Image
from nltk.tokenize import sent_tokenize
from collections import Counter
from scipy.stats import entropy
import nltk
import time
import config
import os

# --- Initialization ---
nltk.download("punkt", quiet=True)

# --- Configuration (using your project's config) ---
PROCESSOR_PATH = config.PROCESSOR_SAVE_PATH
TOKENIZER_PATH = config.TOKENIZER_SAVE_PATH
MODEL_PATH = config.DECODER_SAVE_PATH
DEVICE = config.DEVICE

# --- Load components ---
print("[INFO] Loading processor, tokenizer, and model...")
processor = CLIPProcessor.from_pretrained(PROCESSOR_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# --- Enable dropout at inference ---
def enable_mc_dropout(m):
    for module in m.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

# --- Generate N stochastic reports ---
def generate_mc_samples(image_path, num_samples=5, max_length=128):
    print(f"[INFO] Loading and preprocessing image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"[ERROR] Image not found: {image_path}")
        return []
    inputs = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

    enable_mc_dropout(model)

    reports = []
    print(f"[INFO] Starting MC sampling with {num_samples} iterations...")
    for i in range(num_samples):
        print(f"  → Sampling {i+1}/{num_samples}")
        start_time = time.time()
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
        print(f"    ✓ Report {i+1} generated in {time.time() - start_time:.2f} sec")
    return reports

# --- Compute entropy per sentence position ---
def sentence_level_uncertainty(reports):
    if not reports:
        print("[WARN] No reports generated, cannot compute uncertainty.")
        return []
    print("[INFO] Tokenizing and computing uncertainty across sentences...")
    sentence_lists = [sent_tokenize(r) for r in reports if r]
    if not any(sentence_lists):
        print("[WARN] All reports were empty after sentence tokenization.")
        return []

    max_len = max(len(lst) for lst in sentence_lists if lst) if sentence_lists else 0
    if max_len == 0:
        print("[WARN] No sentences found in any report.")
        return []

    for lst in sentence_lists:
        lst.extend(["<PAD_SENTENCE>"] * (max_len - len(lst)))

    sentence_columns = list(zip(*sentence_lists))
    sentence_scores = []

    print("    Sentence Group Distributions:")
    for idx, group in enumerate(sentence_columns):
        valid_sentences_in_group = [s for s in group if s != "<PAD_SENTENCE>"]
        if not valid_sentences_in_group:
            norm_ent = 1.0
            display_sentence = group[0] if group[0] != "<PAD_SENTENCE>" else "[Padded Position]"
            print(f"      Group {idx+1}: All padding. Uncertainty set to {norm_ent:.4f}")
        else:
            counts = Counter(valid_sentences_in_group)
            probs = np.array(list(counts.values())) / len(valid_sentences_in_group)
            ent = entropy(probs)
            norm_ent = ent / np.log(len(counts)) if len(counts) > 1 else 0.0
            display_sentence = next((s for s in group if s != "<PAD_SENTENCE>"), "[Error: No valid sentence]")
            print(f"      Group {idx+1}: Counts={counts}, Probs={probs}, Entropy={ent:.4f}, NormEntropy={norm_ent:.4f}")

        sentence_scores.append((display_sentence, norm_ent))

    return sentence_scores

# --- Full Pipeline ---
def analyze_image(image_path, num_samples=5):
    print("\n[INFO] Starting uncertainty analysis...")
    total_start_time = time.time()

    samples = generate_mc_samples(image_path, num_samples=num_samples)

    if not samples:
        print("[ERROR] Failed to generate MC samples.")
        return []

    print("\n[INFO] Analyzing uncertainty...")
    scored_sentences = sentence_level_uncertainty(samples)

    print("\n--- Sentence-Level Uncertainty Report ---")
    for i, (sent, score) in enumerate(scored_sentences, 1):
        print(f"Sentence {i}: ({score:.4f}) {sent}")

    print(f"\n[INFO] Total analysis time: {time.time() - total_start_time:.2f} seconds")
    return scored_sentences

# --- Main ---
if __name__ == "__main__":
    image_path = "image1.jpg"
    if not os.path.exists(image_path):
        print(f"[ERROR] Test image not found: {image_path}. Please place a test image or update the path.")
    else:
        analyze_image(image_path, num_samples=10)
