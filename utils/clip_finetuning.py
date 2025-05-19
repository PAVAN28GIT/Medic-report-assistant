# clip_finetuning.py
import torch
from transformers import CLIPModel, CLIPProcessor, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset # Or your custom loading function
from dataset import MedicalReportDataset
import config
import os

def main():
    print(f"Using device: {config.DEVICE}")

    # --- Load Dataset ---
    # Replace with your actual dataset loading logic
    # Example: Load from Hub
    # Or load from local files (implement this part)
    # hf_dataset = load_my_local_dataset(...)
    # For demonstration, let's assume hf_dataset is loaded and has 'train', 'validation' splits
    # Make sure it has 'image', 'findings', 'impression' columns
    print("Loading dataset...")
    # This is a placeholder - replace with your actual data loading
    # For example: hf_dataset = load_dataset("csv", data_files={"train": "path/to/train.csv", ...})
    # hf_dataset = None # REPLACE THIS
    try:
        hf_dataset = load_dataset("itsanmolgupta/mimic-cxr-cleaned-old")
        print("Dataset loaded successfully.")
        print(f"Dataset structure: {hf_dataset}")

        # # --- Select a subset for testing --- 
        # print("Selecting the first 5000 training samples for a test run...")
        # hf_dataset['train'] = hf_dataset['train'].select(range(5000))
        # # If using validation, select a subset too (e.g., first 500)
        # # if 'validation' in hf_dataset:
        # #     hf_dataset['validation'] = hf_dataset['validation'].select(range(500))
        # print(f"Using subset: {hf_dataset}")
        # # ------------------------------------

    except Exception as e:
        print(f"Error loading dataset 'itsanmolgupta/mimic-cxr-cleaned-old': {e}")
        print("Please ensure you are logged into Hugging Face CLI (`huggingface-cli login`) if required.")
        return # Exit if dataset loading fails

    if hf_dataset is None:
        print("Error: Dataset not loaded. Please configure dataset loading in clip_finetuning.py")
        return

    # --- Load CLIP Model and Processor ---
    print(f"Loading CLIP model: {config.CLIP_MODEL_NAME}")
    model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(config.DEVICE)
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)

    # --- Load Decoder Tokenizer (needed for dataset) ---
    # Although not trained here, the dataset needs it for preparing decoder inputs
    decoder_tokenizer = AutoTokenizer.from_pretrained(config.DECODER_MODEL_NAME)
    if decoder_tokenizer.pad_token is None:
         decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    # --- Create Datasets ---
    print("Creating datasets...")
    train_dataset = MedicalReportDataset(hf_dataset, config.IMAGE_DIR, processor, decoder_tokenizer, split='train')
    # Add validation dataset if available
    # eval_dataset = MedicalReportDataset(hf_dataset, config.IMAGE_DIR, processor, decoder_tokenizer, split='validation')

    # --- Training Arguments ---
    # Uses Hugging Face Trainer API for convenience
    training_args = TrainingArguments(
        output_dir="./clip_training_output",
        num_train_epochs=config.EPOCHS_CLIP,
        per_device_train_batch_size=config.BATCH_SIZE,
        # per_device_eval_batch_size=config.BATCH_SIZE, # Uncomment if using eval_dataset
        learning_rate=config.LEARNING_RATE_CLIP,
        warmup_steps=0,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_strategy="epoch",
        # evaluation_strategy="epoch", # Uncomment if using eval_dataset
        load_best_model_at_end=False, # Set to True if using evaluation
        remove_unused_columns=False, # Important as we have custom outputs
        report_to="none", # Disable wandb/tensorboard unless configured
        dataloader_num_workers=4, # Adjust based on your system
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
    )

    # --- Custom Trainer for CLIP Loss ---
    # The default Trainer expects 'labels', but CLIPModel returns 'loss' directly
    class ClipTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Prepare inputs for CLIPModel forward pass
            clip_model_inputs = {
                "pixel_values": inputs["clip_pixel_values"],
                "input_ids": inputs["clip_input_ids"],
                "attention_mask": inputs["clip_attention_mask"],
                "return_loss": True,
            }
            outputs = model(**clip_model_inputs)
            loss = outputs.loss # CLIPModel directly returns contrastive loss
            return (loss, outputs) if return_outputs else loss

    # --- Initialize Trainer ---
    trainer = ClipTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset, # Uncomment if using eval_dataset
        # data_collator=None, # Default collator should work
    )

    # --- Train ---
    print("Starting CLIP fine-tuning...")
    trainer.train()
    print("CLIP fine-tuning finished.")

    # --- Save Model and Processor ---
    print(f"Saving fine-tuned CLIP model to {config.CLIP_SAVE_PATH}")
    os.makedirs(config.CLIP_SAVE_PATH, exist_ok=True)
    model.save_pretrained(config.CLIP_SAVE_PATH)

    print(f"Saving CLIP processor to {config.PROCESSOR_SAVE_PATH}")
    os.makedirs(config.PROCESSOR_SAVE_PATH, exist_ok=True)
    processor.save_pretrained(config.PROCESSOR_SAVE_PATH)
    print("CLIP model and processor saved.")

if __name__ == "__main__":
    main()
