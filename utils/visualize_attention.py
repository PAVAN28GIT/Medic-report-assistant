import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from transformers import GPT2Tokenizer

# Use relative import for config if it's in the same package (utils)
from . import config 
# Assuming inference.py is in the same directory or its path is handled
# If inference.py is also in utils, this could be: from .inference import ...
# For now, let server.app.py handle adding project root to sys.path for this import
from utils.inference import generate_report_data, tokenizer as inference_tokenizer

# Configure Matplotlib for non-interactive backend suitable for servers
plt.switch_backend('Agg')

def plot_attention_heatmap(image: Image, attentions_tuple: tuple, attention_tuple_idx: int,
                         current_word: str = "token", layer_idx: int = -1,
                         output_filename_prefix: str | None = None) -> str | None:
    """
    Processes and plots the cross-attention heatmap for a specific token generation step.
    Saves the plot if output_filename_prefix is provided and returns the filename.
    Returns:
        str | None: The filename of the saved plot, or None if not saved or error.
    """
    if not attentions_tuple or not (0 <= attention_tuple_idx < len(attentions_tuple)):
        # print(f"[PLOT_ERROR] Attention data not available or index {attention_tuple_idx} is out of bounds.")
        return None

    try:
        attention_tensor_for_step = attentions_tuple[attention_tuple_idx]
        if not (0 <= abs(layer_idx) < len(attention_tensor_for_step)):
            # print(f"[PLOT_ERROR] Layer index {layer_idx} out of bounds.")
            return None
        attention_tensor = attention_tensor_for_step[layer_idx]
        
        avg_attention = attention_tensor.mean(dim=1)
        token_attention_vector = avg_attention[0, -1, :].cpu()

        num_encoder_tokens = token_attention_vector.shape[0]
        spatial_attention = None
        grid_size = 0

        if num_encoder_tokens == 50: # ViT-B/32 specific patch count + CLS
            spatial_attention = token_attention_vector[1:] 
            grid_size = 7
        else: # Attempt generic square grid deduction
            temp_spatial_tokens = num_encoder_tokens - 1 if num_encoder_tokens > 0 else 0
            sqrt_tokens_minus_cls = np.sqrt(temp_spatial_tokens) if temp_spatial_tokens > 0 else 0
            sqrt_tokens_direct = np.sqrt(num_encoder_tokens) if num_encoder_tokens > 0 else 0

            if temp_spatial_tokens > 0 and sqrt_tokens_minus_cls == int(sqrt_tokens_minus_cls):
                grid_size = int(sqrt_tokens_minus_cls)
                spatial_attention = token_attention_vector[1:]
            elif num_encoder_tokens > 0 and sqrt_tokens_direct == int(sqrt_tokens_direct):
                grid_size = int(sqrt_tokens_direct)
                spatial_attention = token_attention_vector
            else:
                return None
        
        if spatial_attention is None or spatial_attention.nelement() == 0 or spatial_attention.shape[0] != grid_size*grid_size:
            return None

        attention_map_grid = spatial_attention.reshape(grid_size, grid_size).numpy()
        
        img_w, img_h = image.size
        try:
            resampling_method = Image.Resampling.BILINEAR
        except AttributeError: # Fallback for older Pillow versions
            resampling_method = Image.BILINEAR 
        attention_map_resized = Image.fromarray(attention_map_grid).resize((img_w, img_h), resampling_method)

        fig, ax = plt.subplots(figsize=(5, 5)) # Adjusted for potentially smaller web display
        ax.imshow(image)
        ax.imshow(np.array(attention_map_resized), cmap='viridis', alpha=0.6)
        title_text = f"Attention: '{current_word}' (Step {attention_tuple_idx})"
        ax.set_title(title_text, fontsize=7)
        ax.axis('off')
        
        saved_path = None
        if output_filename_prefix:
            try:
                full_saved_path = f"{output_filename_prefix}.png"
                plt.savefig(full_saved_path, bbox_inches='tight', dpi=100) # Control DPI
                # print(f"    [PLOT] Saved attention map to {full_saved_path}")
                saved_path = full_saved_path
            except Exception as save_e:
                print(f"    [PLOT_ERROR] Failed to save heatmap: {save_e}")
        plt.close(fig) 
        return saved_path
    except Exception as e:
        print(f"[PLOT_ERROR] Error during attention visualization for '{current_word}': {e}")
        return None

def find_sublist_indices(main_list, sub_list):
    indices = []
    len_sub = len(sub_list)
    if len_sub == 0: return []
    for i in range(len(main_list) - len_sub + 1):
        if main_list[i:i+len_sub] == sub_list:
            indices.append(i)
    return indices

def process_and_visualize_medical_terms(original_image: Image, full_attentions_data: tuple, 
                                        generated_sequence_ids_tensor: torch.Tensor, 
                                        tokenizer_for_terms: GPT2Tokenizer, # Use the passed tokenizer
                                        medical_terms_to_visualize: list, 
                                        base_image_filename: str, 
                                        absolute_heatmaps_root_dir: str) -> list:
    if not all([original_image, full_attentions_data, generated_sequence_ids_tensor is not None, tokenizer_for_terms is not None]):
        print("[VIS_ERROR] Missing necessary data for visualizing attention.")
        return []

    image_specific_disk_save_dir = os.path.join(absolute_heatmaps_root_dir, base_image_filename)
    os.makedirs(image_specific_disk_save_dir, exist_ok=True)

    report_token_ids_list = generated_sequence_ids_tensor[0].tolist() 
    saved_heatmap_web_paths = [] 

    for term_string in medical_terms_to_visualize:
        term_token_ids = tokenizer_for_terms.encode(term_string, add_special_tokens=False)
        if not term_token_ids: continue

        occurrence_start_indices = find_sublist_indices(report_token_ids_list, term_token_ids)
        if not occurrence_start_indices:
            term_token_ids_with_space = tokenizer_for_terms.encode(" " + term_string, add_special_tokens=False)
            if term_token_ids_with_space != term_token_ids:
                occurrence_start_indices = find_sublist_indices(report_token_ids_list, term_token_ids_with_space)
                if occurrence_start_indices: term_token_ids = term_token_ids_with_space
            if not occurrence_start_indices: continue
        
        for occ_num, start_idx_in_report in enumerate(occurrence_start_indices):
            for token_offset_in_term, term_token_id_val in enumerate(term_token_ids):
                actual_token_idx_in_report = start_idx_in_report + token_offset_in_term
                attention_data_idx = actual_token_idx_in_report -1 # Attention index is often 0-based for generated tokens

                if not (0 <= attention_data_idx < len(full_attentions_data)):
                    continue
                
                decoded_token = tokenizer_for_terms.decode([term_token_id_val])
                term_safe = term_string.replace(' ', '_').replace('/', '-').replace(':', '') # Sanitize for filename
                
                filename_only = f"{term_safe}_occ{occ_num+1}_token{token_offset_in_term}_pos{actual_token_idx_in_report}"
                absolute_save_file_prefix = os.path.join(image_specific_disk_save_dir, filename_only)
                current_heatmap_web_path = os.path.join(base_image_filename, f"{filename_only}.png")

                saved_plot_filepath = plot_attention_heatmap(
                    original_image, full_attentions_data,
                    attention_tuple_idx=attention_data_idx,
                    current_word=f'{decoded_token} (in "{term_string}")',
                    output_filename_prefix=absolute_save_file_prefix
                )
                if saved_plot_filepath:
                    saved_heatmap_web_paths.append(current_heatmap_web_path)
    return saved_heatmap_web_paths

if __name__ == "__main__":
    if inference_tokenizer is None:
        print("[VIS_MAIN_ERROR] Inference tokenizer could not be loaded. Cannot run standalone test.")
    else:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        test_image_path = os.path.join(current_script_dir, "image8000.jpg")
        base_img_name = os.path.splitext(os.path.basename(test_image_path))[0]
        SEARCH_WORD = "catheter"

        if not os.path.exists(test_image_path):
            print(f"[VIS_MAIN_ERROR] Test image not found: {test_image_path}")
        else:
            report_text, raw_ids, attentions, img = generate_report_data(test_image_path)
            if img and attentions and raw_ids is not None and report_text:
                standalone_heatmap_root = os.path.join(current_script_dir, "attention_heatmaps_test")
                heatmap_web_paths = process_and_visualize_medical_terms(
                    img, attentions, raw_ids,
                    inference_tokenizer, # Use the imported tokenizer
                    [SEARCH_WORD], 
                    base_img_name,
                    absolute_heatmaps_root_dir=standalone_heatmap_root
                )
                if heatmap_web_paths:
                    print(f"\n[VIS_MAIN] Heatmaps generated for '{SEARCH_WORD}'. Saved in {os.path.join(standalone_heatmap_root, base_img_name)}")
                else:
                    print(f"[VIS_MAIN] No heatmaps generated for '{SEARCH_WORD}'.")
            else:
                print("[VIS_MAIN_ERROR] Could not run visualization due to missing data from inference.")
