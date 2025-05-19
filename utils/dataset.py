# dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import config

class MedicalReportDataset(Dataset):
    def __init__(self, hf_dataset, image_dir, clip_processor, decoder_tokenizer, split='train'):
        """
        Args:
            hf_dataset: Hugging Face dataset object containing 'image', 'findings', 'impression'.
                        Assumes 'image' column might contain relative paths or image objects.
            image_dir: Base directory where images are stored (if 'image' column has paths).
            clip_processor: The CLIP processor for images and text.
            decoder_tokenizer: The GPT-2 tokenizer for the reports.
            split: 'train', 'validation', or 'test'.
        """
        self.dataset = hf_dataset[split]
        self.image_dir = image_dir
        self.clip_processor = clip_processor
        self.decoder_tokenizer = decoder_tokenizer

        # Ensure decoder tokenizer has a padding token
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
            print("Set decoder pad_token to eos_token.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # --- Load Image ---
        image_input = item[config.IMAGE_COLUMN]
        if isinstance(image_input, str): # If it's a path
             # Construct the full path carefully based on your dataset structure
             # This might need adjustment depending on how paths are stored in your CSV/dataset
            image_path = os.path.join(self.image_dir, image_input)
            try:
                image = Image.open(image_path).convert("RGB")
            except FileNotFoundError:
                print(f"Warning: Image not found at {image_path}. Skipping sample or returning None.")
                # Handle appropriately - maybe return None and filter in DataLoader collate_fn
                # For now, let's raise an error if essential
                raise FileNotFoundError(f"Image not found: {image_path}")
            except Exception as e:
                print(f"Warning: Error loading image {image_path}: {e}")
                raise # Re-raise other loading errors
        elif isinstance(image_input, Image.Image):
             image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image format in dataset: {type(image_input)}")


        # --- Prepare Text (Combine Findings and Impression) ---
        findings = item.get(config.FINDINGS_COLUMN, "") or "" # Handle None or empty
        impression = item.get(config.IMPRESSION_COLUMN, "") or "" # Handle None or empty
        report_text = f"Findings: {findings} Impression: {impression}"

        # --- Process for CLIP (Image + Combined Text) ---
        # We process text here mainly to align with image features during CLIP fine-tuning
        clip_inputs = self.clip_processor(
            text=[report_text], # Process as a list for consistency
            images=image,
            return_tensors="pt",
            padding="max_length", # Pad to max length defined by CLIP model
            truncation=True,
            max_length=self.clip_processor.tokenizer.model_max_length # Use processor's tokenizer max length
        )

        # Squeeze batch dimensions added by processor
        clip_pixel_values = clip_inputs['pixel_values'].squeeze(0)
        clip_input_ids = clip_inputs['input_ids'].squeeze(0)
        clip_attention_mask = clip_inputs['attention_mask'].squeeze(0)

        # --- Process for Decoder (Target Report Text) ---
        # Tokenize the report text for the decoder target
        decoder_encodings = self.decoder_tokenizer(
            report_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=config.MAX_TEXT_LENGTH # Use config max length
        )

        decoder_input_ids = decoder_encodings['input_ids'].squeeze(0)
        decoder_attention_mask = decoder_encodings['attention_mask'].squeeze(0)

        # Decoder labels are the input_ids shifted (handled by model if labels provided)
        # We provide input_ids as labels, model handles shifting internally
        labels = decoder_input_ids.clone()
        # Mask padding tokens in labels
        labels[labels == self.decoder_tokenizer.pad_token_id] = -100


        return {
            "clip_pixel_values": clip_pixel_values,
            "clip_input_ids": clip_input_ids,
            "clip_attention_mask": clip_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels # Labels for the decoder LM task
        }
