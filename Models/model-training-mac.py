import os
# Disable the MPS memory watermark, if needed (use with caution)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

# Import PEFT modules for LoRA
from peft import LoraConfig, get_peft_model, TaskType


def load_and_combine_datasets(agent_file, parsing_file):
    """
    Load two JSON files and combine them.
    Assumes each JSON file contains a list of examples.
    Each example should be a dict with at least a "prompt" key and optionally a "completion" key.
    """
    agent_dataset = load_dataset("json", data_files=agent_file, split="train")
    parsing_dataset = load_dataset("json", data_files=parsing_file, split="train")
    combined_dataset = concatenate_datasets([agent_dataset, parsing_dataset])
    return combined_dataset


def preprocess_function(examples, tokenizer, max_length=1024):
    """
    Preprocess each training example.
    Expects each example to have 'prompt' and optionally 'completion'.
    If a prompt or completion is None, it is replaced with an empty string.
    The final text is: prompt + EOS token + completion.
    
    For causal LM training, we also set the labels equal to the tokenized input.
    """
    inputs = []
    for i in range(len(examples["prompt"])):
        # Retrieve prompt; if None, set to empty string, otherwise strip spaces.
        prompt_val = examples["prompt"][i]
        if prompt_val is None:
            prompt_val = ""
        else:
            prompt_val = prompt_val.strip()
        
        # Do the same for the completion if present.
        if "completion" in examples:
            completion_val = examples["completion"][i]
            if completion_val is None:
                completion_val = ""
            else:
                completion_val = completion_val.strip()
        else:
            completion_val = ""
        
        # Concatenate prompt and completion using the EOS token as a delimiter.
        full_text = prompt_val + tokenizer.eos_token + completion_val
        inputs.append(full_text)
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    # Set labels so that the Trainer can compute the loss.
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


def main():
    # File paths for your JSON training files.
    agent_file = "training-datasets/Agent-training.json"
    parsing_file = "training-datasets/parsing_instructions.json"
    
    # The base model to fine-tune.
    model_id = "meta-llama/Llama-3.2-1B"
    
    # --- Device Setup ---
    # Check for CUDA first. If not available, then for MPS (Apple Silicon), else use CPU.
    if torch.cuda.is_available():
        device = "cuda"
        use_fp16 = True
        use_bf16 = False
        print("Using CUDA GPU:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = "mps"
        use_fp16 = False
        use_bf16 = True  # Use bf16 for MPS
        print("Using MPS device (Apple GPU)")
    else:
        device = "cpu"
        use_fp16 = False
        use_bf16 = False
        print("No GPU found. Using CPU (performance will be lower).")
    
    # --- Step 1. Load and Combine Datasets ---
    print("Loading and combining datasets...")
    dataset = load_and_combine_datasets(agent_file, parsing_file)
    print("Combined dataset size:", len(dataset))
    
    # --- Step 2. Load the Tokenizer and Model ---
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # If no pad token exists, set it to the EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.resize_token_embeddings(len(tokenizer))
    
    # --- Integrate LoRA using PEFT ---
    # Define LoRA configuration for causal language modeling.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # For causal language model tasks
        r=8,                          # LoRA rank (adjust as needed)
        lora_alpha=32,                # Scaling factor (adjust as needed)
        lora_dropout=0.1,             # Dropout for regularization
    )
    # Wrap the model with LoRA adapters; most original weights are frozen.
    model = get_peft_model(model, lora_config)
    print("Model wrapped with LoRA adapters:")
    model.print_trainable_parameters()
    
    # --- Step 3. Preprocess the Dataset ---
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # --- Step 4. Set Up Training Arguments ---
    training_args = TrainingArguments(
        output_dir="./llama_finetuned",
        overwrite_output_dir=True,
        num_train_epochs=10,  # Adjust the number of epochs as needed.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=use_fp16,
        bf16=use_bf16,  # Use bf16 when on MPS
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        no_cuda=False,
        remove_unused_columns=False,
    )
    
    # --- Step 5. Create the Trainer and Fine-Tune the Model ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    print("Starting fine-tuning on", device, "...")
    trainer.train()
    
    # --- Step 6. Save the Fine-Tuned Model and Tokenizer ---
    output_dir = "./llama_finetuned_final"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Fine-tuned model saved to {output_dir}")
    
    # --- Step 7. Test the Fine-Tuned Model with a Generation Pipeline ---
    # For generation, if a GPU (CUDA or MPS) is available, set device index to 0.
    gen_device = 0 if device in ["cuda", "mps"] else -1
    generation_pipeline = transformers.pipeline(
        "text-generation",
        model=output_dir,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16} if device in ["cuda", "mps"] else {},
        device=gen_device,
    )
    
    # Define sample conversation messages and convert them into a single prompt.
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"}
    ]
    prompt = "\n".join([f'{m["role"]}: {m["content"]}' for m in messages])
    
    outputs = generation_pipeline(prompt, max_new_tokens=256)
    print("Generated output:")
    print(outputs[0]["generated_text"])

if __name__ == "__main__":
    main()