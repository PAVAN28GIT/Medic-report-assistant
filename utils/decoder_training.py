# decoder_training.py
import torch
from transformers import (
    CLIPModel,
    CLIPProcessor,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    VisionEncoderDecoderModel, # Use this instead of EncoderDecoderModel
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    CLIPVisionModel # Import CLIPVisionModel specifically
)
from datasets import load_dataset # Or your custom loading function
from dataset import MedicalReportDataset
import config
import os

# --- Custom Data Collator ---
# Default collator might not handle our specific dictionary structure well
def collate_fn(batch):
    pixel_values = torch.stack([item['clip_pixel_values'] for item in batch])
    decoder_input_ids = torch.stack([item['decoder_input_ids'] for item in batch])
    decoder_attention_mask = torch.stack([item['decoder_attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'pixel_values': pixel_values, # Input to the encoder (CLIP vision)
        'decoder_input_ids': decoder_input_ids, # Input to the decoder
        'decoder_attention_mask': decoder_attention_mask,
        'labels': labels # Target labels for the decoder
    }

def main():
    # --- Load Fine-tuned Vision Encoder --- 
    # Load ONLY the vision part from the saved CLIP model directory
    print(f"Loading fine-tuned CLIP *Vision* model from: {config.CLIP_SAVE_PATH}")
    clip_vision_model = CLIPVisionModel.from_pretrained(config.CLIP_SAVE_PATH).to(config.DEVICE)
    print("CLIP Vision model loaded.")

    # --- Load Processor --- (Still needed for data preparation)
    print(f"Loading CLIP processor from: {config.PROCESSOR_SAVE_PATH}")
    clip_processor = CLIPProcessor.from_pretrained(config.PROCESSOR_SAVE_PATH)


    # --- Load Decoder Components ---
    print(f"Loading Decoder Tokenizer: {config.DECODER_MODEL_NAME}")
    decoder_tokenizer = GPT2Tokenizer.from_pretrained(config.DECODER_MODEL_NAME)
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
        print("Set decoder pad_token to eos_token.")

    print(f"Loading Decoder Model: {config.DECODER_MODEL_NAME}")
    decoder_lm_head_model = GPT2LMHeadModel.from_pretrained(
        config.DECODER_MODEL_NAME,
        pad_token_id=decoder_tokenizer.pad_token_id,
        add_cross_attention=True
    ).to(config.DEVICE)


    # --- Combine into VisionEncoderDecoderModel directly ---
    print("Instantiating VisionEncoderDecoderModel with loaded components...")
    model = VisionEncoderDecoderModel(encoder=clip_vision_model, decoder=decoder_lm_head_model)

    # Freeze encoder weights AFTER model creation
    print("Freezing weights of the encoder part (CLIP Vision Model)...")
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Configure the model for generation
    model.config.decoder_start_token_id = decoder_tokenizer.bos_token_id
    model.config.pad_token_id = decoder_tokenizer.pad_token_id
    model.config.eos_token_id = decoder_tokenizer.eos_token_id
    # Important: Link vocab size if decoder's different from config
    model.config.vocab_size = model.config.decoder.vocab_size


    # --- Load Dataset ---
    print("Loading dataset...")
    # This is a placeholder - replace with your actual data loading
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
        exit() # Exit if dataset loading fails

    if hf_dataset is None:
        print("Error: Dataset not loaded. Please configure dataset loading in decoder_training.py")
        exit()

    # --- Create Datasets ---
    print("Creating datasets...")
    train_dataset = MedicalReportDataset(hf_dataset, config.IMAGE_DIR, clip_processor, decoder_tokenizer, split='train')
    # Add validation dataset if available
    # eval_dataset = MedicalReportDataset(hf_dataset, config.IMAGE_DIR, clip_processor, decoder_tokenizer, split='validation')


    # --- Training Arguments ---
    # Use Seq2SeqTrainingArguments for EncoderDecoder models
    training_args = Seq2SeqTrainingArguments(
        output_dir="./decoder_training_output",
        num_train_epochs=config.EPOCHS_DECODER,
        per_device_train_batch_size=config.BATCH_SIZE,
        # per_device_eval_batch_size=config.BATCH_SIZE, # Uncomment if using eval_dataset
        learning_rate=config.LEARNING_RATE_DECODER,
        warmup_steps=0,
        weight_decay=0.01,
        logging_dir='./logs_decoder',
        logging_steps=100,
        save_strategy="epoch",
        # evaluation_strategy="epoch", # Uncomment if using eval_dataset
        predict_with_generate=True, # Needed for Seq2Seq models if evaluating
        load_best_model_at_end=False, # Set to True if using evaluation
        remove_unused_columns=False, # We need our custom columns
        report_to="none",
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),
    )

    # --- Initialize Trainer ---
    # Use Seq2SeqTrainer for EncoderDecoder models
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset, # Uncomment if using eval_dataset
        tokenizer=decoder_tokenizer, # Pass tokenizer for generate if needed
        data_collator=collate_fn,
    )

    # --- Train ---
    print("Starting Decoder training...")
    trainer.train()
    print("Decoder training finished.")

    # --- Save Decoder Model and Tokenizer ---
    print(f"Saving trained Decoder model to {config.DECODER_SAVE_PATH}")
    os.makedirs(config.DECODER_SAVE_PATH, exist_ok=True)
    model.save_pretrained(config.DECODER_SAVE_PATH) # Saves both encoder (frozen) and decoder

    print(f"Saving Decoder tokenizer to {config.TOKENIZER_SAVE_PATH}")
    os.makedirs(config.TOKENIZER_SAVE_PATH, exist_ok=True)
    decoder_tokenizer.save_pretrained(config.TOKENIZER_SAVE_PATH)
    print("Decoder model and tokenizer saved.")

if __name__ == "__main__":
    main() # Call the main function
 