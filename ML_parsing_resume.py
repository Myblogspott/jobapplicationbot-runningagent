"""
Author: Sai Raghavendra Maddula
Full Resume Parsing Script Using LayoutLM & Nvidia Dynamo
-----------------------------------------------------------
This script does the following:
1. Loads the resume parsing dataset from Hugging Face.
2. Loads an ATS-friendly resume template from a local PDF.
3. Preprocesses the dataset (tokenizes text, aligns bounding boxes and labels).
4. Initializes the LayoutLM model for token classification.
5. Optimizes the model using Nvidia Dynamo for GPU acceleration.
6. Sets up training using Hugging Face's Trainer.
7. Trains and evaluates the model.
8. (Optional) Runs inference on your uploaded ATS-friendly template.

Required packages:
    pip install transformers datasets seqeval pymupdf dynamo
    (Ensure Nvidia Dynamo is installed per its GitHub instructions: https://github.com/ai-dynamo/dynamo)

Update the variable 'template_pdf_path' with the path to your resume template PDF.


"""

import os
import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import (
    LayoutLMTokenizer,
    LayoutLMForTokenClassification,
    TrainingArguments,
    Trainer,
)
import fitz  # PyMuPDF for PDF processing

# Import Nvidia Dynamo for GPU acceleration.
# Make sure it is installed and configured correctly.
import dynamo

# -------------------------
# 1. Load the Resume Parsing Dataset
# -------------------------
print("Loading the resume parsing dataset from Hugging Face...")
dataset = load_dataset("shandilyabh/resume_parsing")
print("Dataset splits available:", list(dataset.keys()))

# Extract label names from the dataset.
labels_list = dataset["train"].features["labels"].feature.names
num_labels = len(labels_list)
id2label = {i: label for i, label in enumerate(labels_list)}
label2id = {label: i for i, label in enumerate(labels_list)}

# -------------------------
# 2. Load the ATS-Friendly Resume Template from a PDF
# -------------------------
# Provide the path to your ATS-friendly resume template PDF file.
template_pdf_path = "path/to/your/template.pdf"  # <<<--- UPDATE THIS PATH

def load_resume_template(file_path):
    """
    Loads a PDF resume template and extracts text and bounding box information.
    For demonstration, this function extracts words and their bounding boxes for each page.
    """
    if not os.path.exists(file_path):
        print("Template file not found at:", file_path)
        return None

    doc = fitz.open(file_path)
    pages_data = []
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        # Extract words along with bounding boxes.
        # Each tuple: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        words_data = page.get_text("words")
        extracted_words = []
        bboxes = []
        for word_tuple in words_data:
            x0, y0, x1, y1, word, block_no, line_no, word_no = word_tuple
            extracted_words.append(word)
            bbox = [int(x0), int(y0), int(x1), int(y1)]
            bboxes.append(bbox)
        # For the template, we may not have annotated labels.
        # Here we assign a dummy label (-100) for each word to ignore them during training.
        pages_data.append({
            "words": extracted_words,
            "bboxes": bboxes,
            "labels": [-100] * len(extracted_words)
        })
    return pages_data

print("Loading ATS-friendly resume template from:", template_pdf_path)
template_data = load_resume_template(template_pdf_path)
if template_data:
    print("Loaded template data for", len(template_data), "page(s).")
else:
    print("No template data loaded. Please check the template path.")

# -------------------------
# 3. Initialize Tokenizer and LayoutLM Model
# -------------------------
model_name = "microsoft/layoutlm-base-uncased"
print("Initializing tokenizer and model:", model_name)
tokenizer = LayoutLMTokenizer.from_pretrained(model_name)
model = LayoutLMForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# -------------------------
# 4. Optimize Model using Nvidia Dynamo for GPU Acceleration
# -------------------------
print("Optimizing model with Nvidia Dynamo for GPU acceleration...")
model = dynamo.optimize(model)

# -------------------------
# 5. Preprocess the Dataset for Token Classification
# -------------------------
def preprocess_data(examples):
    """
    Tokenizes words with their bounding boxes and aligns labels for token classification.
    Each example is expected to have 'words', 'bboxes', and 'labels'.
    """
    tokenized_inputs = tokenizer(
        examples["words"],
        boxes=examples["bboxes"],
        truncation=True,
        padding="max_length",
        is_split_into_words=True,
    )
    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens get label -100 (ignored in loss)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

print("Preprocessing the training dataset...")
tokenized_dataset = dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
print("Dataset preprocessing complete.")

# -------------------------
# 6. Set Up Training Arguments and Evaluation Metrics
# -------------------------
training_args = TrainingArguments(
    output_dir="./layoutlm_resume_parser",
    evaluation_strategy="steps",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    remove_unused_columns=False,
)

metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# -------------------------
# 7. Initialize the Trainer
# -------------------------
# Use the "validation" split if it exists; otherwise, fall back to "test"
eval_split = "validation" if "validation" in tokenized_dataset else "test"
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset[eval_split],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# -------------------------
# 8. Train and Evaluate the Model
# -------------------------
print("Starting model training...")
trainer.train()
print("Training complete. Evaluating the model...")
results = trainer.evaluate()
print("Evaluation results:", results)

# -------------------------
# 9. (Optional) Run Inference on the Uploaded Template
# -------------------------
if template_data:
    print("Running inference on the ATS-friendly resume template...")
    for idx, page in enumerate(template_data):
        print(f"\nProcessing page {idx + 1}:")
        inputs = tokenizer(
            page["words"],
            boxes=page["bboxes"],
            truncation=True,
            padding="max_length",
            is_split_into_words=True,
            return_tensors="pt"
        )
        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
        word_ids = inputs.word_ids(batch_index=0)
        predicted_labels = []
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and predictions[i] != -100:
                predicted_labels.append(id2label[predictions[i]])
        print("Predicted labels for this page:", predicted_labels)

print("\nScript execution completed.")