#!/usr/bin/env python3
"""
Production-Ready DeBERTa PII Token Classification Fine-Tuning Script.

Features:
- Optuna Hyperparameter Search (optimises lr, batch size, and epochs).
- Model Initialization Hook (model_init) for clean trials.
- Epoch Checkpointing & Resiliency (auto-resumes from unexpected crashes).
- Dynamic Label Mismatch Protection (ignore_mismatched_sizes=True).
- Weighted Loss Handling (downweights background 'O' labels to balance PII representation).
- Exception Handling with checkpoint auto-detection.

Prerequisites:
    pip install torch transformers datasets accelerate optuna
"""

import os
import glob
import logging
import torch
from typing import Dict, List, Any
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pii_deberta_training")

# ----------------------------------------------------------------------
# 1. Dataset & Label Configuration (Placeholder)
# ----------------------------------------------------------------------
# Define the path or name of the pre-trained DeBERTa base model
MODEL_NAME = "Isotonic/deberta-v3-base_finetuned_ai4privacy_v2"
OUTPUT_DIR = "./models/finetuned-deberta"

# Define your custom label mapping. 
# Make sure to include both B- and I- prefixes for your PII entities.
LABEL_LIST = [
    "O",
    "B-NAME", "I-NAME",
    "B-EMAIL", "I-EMAIL",
    "B-PHONE", "I-PHONE",
    "B-SSN", "I-SSN",
    "B-ADDRESS", "I-ADDRESS",
    # Add your own domain-specific PII labels here
]

ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}
NUM_LABELS = len(LABEL_LIST)


# ----------------------------------------------------------------------
# 2. Weighted Loss Handler
# ----------------------------------------------------------------------
class WeightedPIITrainer(Trainer):
    """
    Custom Trainer that implements class-weighted Cross-Entropy loss.
    PII samples are typically highly imbalanced, with the background 'O' label
    dominating the token distribution. This subclass dynamically balances
    the loss to ensure correct PII detection.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Create class weights dynamically based on current label config length
        num_classes = model.config.num_labels
        device = logits.device
        
        # Default all weights to 1.0 (PII tokens)
        class_weights = torch.ones(num_classes, dtype=torch.float32, device=device)
        
        # Find index of the background class 'O' dynamically
        label2id = getattr(model.config, "label2id", {})
        o_id = label2id.get("O", 0)
        
        # Apply 0.1 weight to background 'O' labels to mitigate imbalance
        if o_id < num_classes:
            class_weights[o_id] = 0.1
            
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=-100  # PyTorch standard for padded token masking
        )
        
        # Flatten logits and compute loss
        loss = loss_fct(logits.view(-1, num_classes), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


# ----------------------------------------------------------------------
# 3. Model Initialization (model_init) Hook
# ----------------------------------------------------------------------
def model_init() -> AutoModelForTokenClassification:
    """
    Instantiates a clean copy of the model for Optuna hyperparameter trials.
    Incorporates label mismatch protection to dynamically size the classification head.
    Wraps the model with a LoRA adapter using peft.
    """
    from peft import LoraConfig, get_peft_model
    logger.info(f"Initializing clean model weights from: {MODEL_NAME}")
    
    # Configure custom label mappings
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    # Dynamic size protection: ignore mismatched weights (e.g., classifier head)
    # when loading pre-trained weights into a model with a different label count.
    base_model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False
    )
    
    peft_config = LoraConfig(
        task_type="TOKEN_CLS",
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query_proj", "value_proj"],
        modules_to_save=["classifier"]
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    return model


# ----------------------------------------------------------------------
# 4. Checkpoint Detection Helper
# ----------------------------------------------------------------------
def get_latest_checkpoint(directory: str) -> str:
    """
    Scans the output directory and returns the path to the latest checkpoint directory,
    or None if no checkpoints are found.
    """
    if not os.path.exists(directory):
        return None
        
    checkpoints = glob.glob(os.path.join(directory, "checkpoint-*"))
    if not checkpoints:
        return None
        
    # Sort by the checkpoint index (e.g. checkpoint-123 -> 123)
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest_checkpoint = checkpoints[-1]
    logger.info(f"Auto-detected existing checkpoint for resumption: {latest_checkpoint}")
    return latest_checkpoint


# ----------------------------------------------------------------------
# 5. Hyperparameter Space Definition (Optuna)
# ----------------------------------------------------------------------
def hp_space_definition(trial) -> Dict[str, Any]:
    """
    Specifies the parameter search grid for Optuna.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
    }


# ----------------------------------------------------------------------
# 6. Main Execution Logic
# ----------------------------------------------------------------------
def main():
    # Placeholder: Initialize your tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Placeholder dataset block:
    # Under a production setup, replace these lists with your tokenized dataset dicts
    # or load them using the Hugging Face `datasets` library.
    # Note: Tokens are lists of string words, labels are lists of integer label IDs.
    mock_train_dataset = [
        {
            "input_ids": tokenizer("My name is John Doe and my email is john@email.com")["input_ids"],
            # Replace with real labels (e.g., 0 for 'O', B-NAME id, etc.) matching input_ids length
            "labels": [0] * len(tokenizer("My name is John Doe and my email is john@email.com")["input_ids"])
        }
    ]
    mock_eval_dataset = mock_train_dataset  # Simple validation mirror for trial evaluation

    # Set up data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Establish training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir="./logs",
        logging_steps=10,
        
        # Resilience & Checkpointing Strategy
        save_strategy="epoch",       # Save a checkpoint folder after every epoch
        save_total_limit=2,          # Keep last 2 checkpoint directories to limit disk size
        eval_strategy="epoch",  # Evaluate after every epoch for hyperparameter optimization
        
        # Resource settings
        disable_tqdm=False,
        report_to="none"             # Disable default metric logging (e.g. wandb, tensorboard)
    )

    # Initialize the custom weighted trainer using model_init hook
    trainer = WeightedPIITrainer(
        args=training_args,
        train_dataset=mock_train_dataset,
        eval_dataset=mock_eval_dataset,
        data_collator=data_collator,
        model_init=model_init,
        tokenizer=tokenizer
    )

    # Exception-guarded hyperparameter search / training initiation
    try:
        logger.info("Starting hyperparameter tuning with Optuna...")
        
        # Run Optuna hyperparameter optimization
        best_run = trainer.hyperparameter_search(
            direction="minimize",
            backend="optuna",
            hp_space=hp_space_definition,
            n_trials=3  # Adjust trials for production tuning
        )
        logger.info(f"Optimal parameters identified: {best_run.hyperparameters}")
        
        # Apply the best identified hyperparameters to the trainer arguments
        for param_name, param_value in best_run.hyperparameters.items():
            setattr(training_args, param_name, param_value)
            
        # Re-initialize trainer using best trial hyperparams
        trainer = WeightedPIITrainer(
            args=training_args,
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
            data_collator=data_collator,
            model_init=model_init,
            tokenizer=tokenizer
        )

        # Detect any existing checkpoints in output directory
        checkpoint_path = get_latest_checkpoint(OUTPUT_DIR)
        
        logger.info("Initiating model training loop...")
        
        try:
            # Execute training (resuming from checkpoint if one exists)
            trainer.train(resume_from_checkpoint=checkpoint_path or False)
        except Exception as e:
            if checkpoint_path:
                logger.warning(
                    f"Failed to resume from checkpoint '{checkpoint_path}' ({e}). "
                    "Cleaning up checkpoint and retrying training from scratch..."
                )
                try:
                    import shutil
                    if os.path.exists(checkpoint_path):
                        shutil.rmtree(checkpoint_path, ignore_errors=True)
                except Exception as cleanup_err:
                    logger.warning(f"Could not clean up checkpoint directory: {cleanup_err}")
                
                # Re-initialize trainer using best trial hyperparams
                trainer = WeightedPIITrainer(
                    args=training_args,
                    train_dataset=mock_train_dataset,
                    eval_dataset=mock_eval_dataset,
                    data_collator=data_collator,
                    model_init=model_init,
                    tokenizer=tokenizer
                )
                trainer.train(resume_from_checkpoint=False)
            else:
                raise e
        
        # Save the finalized model
        trainer.save_model(OUTPUT_DIR)
        trainer.model.config.save_pretrained(OUTPUT_DIR)
        logger.info(f"Training completed successfully. Fine-tuned model saved to: {OUTPUT_DIR}")

    except Exception as e:
        logger.critical(
            f"Training process aborted due to an exception: {e}",
            exc_info=True
        )
        
        # Perform checkpoint sweep for recovery assistance
        checkpoint_path = get_latest_checkpoint(OUTPUT_DIR)
        if checkpoint_path:
            logger.info(
                f"Recovery Assist: A valid checkpoint was detected at '{checkpoint_path}'. "
                "You can safely restart the training loop with 'resume_from_checkpoint=True' "
                "to resume execution."
            )
        else:
            logger.warning("Recovery Assist: No active checkpoints were detected in the target directory.")


if __name__ == "__main__":
    main()
