# inference.py
import torch
from transformers import (
    CLIPProcessor,
    GPT2Tokenizer,
    VisionEncoderDecoderModel # Load combined model
)
from PIL import Image
import config
import os
import numpy as np  # Keep numpy for potential attention processing
import matplotlib.pyplot as plt # <<< ADDED for plotting
# import matplotlib.pyplot as plt # Needed for visualization
# from PIL import Image # Needed for visualization

print(f"Using device: {config.DEVICE}")

# --- Load Models and Processors ---
print(f"Loading processor from: {config.PROCESSOR_SAVE_PATH}")
processor = CLIPProcessor.from_pretrained(config.PROCESSOR_SAVE_PATH)

print(f"Loading tokenizer from: {config.TOKENIZER_SAVE_PATH}")
tokenizer = GPT2Tokenizer.from_pretrained(config.TOKENIZER_SAVE_PATH)

print(f"Loading trained model from: {config.DECODER_SAVE_PATH}")
# Load the combined VisionEncoderDecoderModel saved during decoder training
model = VisionEncoderDecoderModel.from_pretrained(config.DECODER_SAVE_PATH).to(config.DEVICE)
model.eval() # Set to evaluation mode


def generate_report(image_path: str, max_length=128, num_beams=4):
    """
    Generates a medical report for a given X-ray image path.
    Also returns the raw cross-attention weights and the loaded image if available.

    Args:
        image_path (str): Path to the input X-ray image.
        max_length (int): Maximum length of the generated report.
        num_beams (int): Number of beams for beam search generation.

    Returns:
        tuple(str | None, tuple | None, Image | None): A tuple containing:
            - str | None: The generated medical report, or None on error.
            - tuple | None: The raw cross-attention tensors, or None if not available/error.
            - Image | None: The loaded PIL Image object, or None on error.
    """
    image = None
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None, None, None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None

    print(f"Processing image: {image_path}")
    # Process image using CLIP processor
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(config.DEVICE)

    print("Generating report...")
    cross_attentions = None
    report = None
    try:
        with torch.no_grad():
            # Generate output IDs using beam search
            # Add return_dict_in_generate=True and output_attentions=True
            generate_output = model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                return_dict_in_generate=True, # <<< KEEP THIS
                output_attentions=True,       # <<< KEEP THIS
            )
            output_ids = generate_output.sequences # Generated token IDs

            # Extract cross-attentions (might be None if not applicable)
            cross_attentions = generate_output.cross_attentions if hasattr(generate_output, 'cross_attentions') else None

        print("Decoding generated IDs...")
        # Decode the generated IDs to text
        report = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    except Exception as e:
        print(f"Error during generation or decoding: {e}")
        return None, None, image # Return image even if generation fails

    # Return report, raw cross-attention tensors, and the image object
    return report, cross_attentions, image


def plot_attention_heatmap(image: Image, attentions: tuple, token_idx_to_vis: int = -1, layer_idx: int = -1):
    """
    Processes and plots the cross-attention heatmap.

    Args:
        image (Image): The original PIL image.
        attentions (tuple): The raw cross_attentions output from model.generate.
        token_idx_to_vis (int): The index of the generated token step for which to visualize attention. Defaults to -1 (last token).
        layer_idx (int): The index of the decoder layer from which to visualize attention. Defaults to -1 (last layer).
    """
    if not attentions or len(attentions) <= abs(token_idx_to_vis):
        print("Attention data not available or not long enough for the selected token index.")
        return

    print(f"Processing attention for token step {token_idx_to_vis}, layer {layer_idx}...")
    try:
        # Structure: Tuple[Tuple[Tensor(batch, heads, seq_len_dec, seq_len_enc)]]
        # attentions[token_step][layer_index]
        attention_tensor = attentions[token_idx_to_vis][layer_idx] # Get attentions for the specific step and layer
        print(f"  Debug: Raw attention_tensor shape: {attention_tensor.shape}")

        # Average across heads (dim=1)
        # Shape before: (beam_batch_size, num_heads, num_decoder_tokens_at_step, num_encoder_tokens)
        # Shape after: (beam_batch_size, num_decoder_tokens_at_step, num_encoder_tokens)
        avg_attention = attention_tensor.mean(dim=1)
        print(f"  Debug: avg_attention shape (after mean over heads): {avg_attention.shape}")

        # Select attention from the first beam (index 0) and the last generated token position (-1) TO the encoder tokens
        # Shape: (num_encoder_tokens,)
        token_attention_vector = avg_attention[0, -1, :].cpu() # Select beam 0, last decoder token
        print(f"  Debug: token_attention_vector shape: {token_attention_vector.shape}")

        # --- Reshape based on Encoder Patch Grid ---
        num_encoder_tokens = token_attention_vector.shape[0] # This should now be 50
        print(f"  Debug: Calculated num_encoder_tokens: {num_encoder_tokens}")

        # For ViT-B/32, encoder output is num_patches + 1 (CLS token). Patches = (224/32)^2 = 49. So 50 tokens.
        # We usually ignore the CLS token attention for visualization.
        if num_encoder_tokens == 50: # Specific check for ViT-B/32 + CLS token
             spatial_attention = token_attention_vector[1:] # Skip the first token (CLS)
             grid_size = 7 # int(np.sqrt(49))
             print(f"Detected ViT-B/32 structure (49 patches + CLS). Visualizing spatial attention ({grid_size}x{grid_size}).")
        else:
             # Attempt a generic square grid if possible
             sqrt_tokens = np.sqrt(num_encoder_tokens)
             if sqrt_tokens == int(sqrt_tokens):
                 grid_size = int(sqrt_tokens)
                 spatial_attention = token_attention_vector
                 print(f"Assuming a square attention grid ({grid_size}x{grid_size}).")
             else:
                  print(f"Cannot reshape encoder tokens ({num_encoder_tokens}) to a square grid. Visualization skipped.")
                  return

        # Reshape to grid: (grid_size, grid_size)
        attention_map_grid = spatial_attention.reshape(grid_size, grid_size).numpy() # Convert to numpy

        # --- Upscale and Plot ---
        img_w, img_h = image.size # Get original image size

        # Upscale attention map smoothly
        # Use Image.Resampling.BILINEAR (or .BICUBIC)
        attention_map_resized = Image.fromarray(attention_map_grid).resize((img_w, img_h), Image.Resampling.BILINEAR)

        # Plotting
        fig, ax = plt.subplots()
        ax.imshow(image) # Show the original image
        im = ax.imshow(np.array(attention_map_resized), cmap='viridis', alpha=0.5) # Overlay heatmap with transparency
        ax.set_title(f"Cross-Attention Heatmap (Token Step: {token_idx_to_vis}, Layer: {layer_idx})")
        ax.axis('off') # Hide axes
        fig.colorbar(im, ax=ax) # Add a color bar
        plt.show() # Display the plot window

    except Exception as e:
        print(f"Error during attention visualization: {e}")


if __name__ == "__main__":
    # --- Example Usage ---
    test_image_path = "image8000.jpg" # Replace with your test image path

    if not os.path.exists(test_image_path):
        print(f"Test image not found at: {test_image_path}")
        print("Please update the path in inference.py for testing.")
    else:
        # Generate report and get attentions + original image
        generated_report, attentions, original_image = generate_report(test_image_path)

        print("--- Generated Report ---")
        if generated_report:
            print(generated_report)
        else:
            print("Report generation failed.")

        print("--- Cross-Attention Visualization --- ")
        if attentions and original_image:
             # Call the plotting function
             plot_attention_heatmap(original_image, attentions)
        elif not original_image:
             print("Cannot visualize attention, image failed to load.")
        else:
             print("Cross-attention data was not returned by the model or generation failed.")
        print("-----------------------------------")
