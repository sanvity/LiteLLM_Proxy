#!/usr/bin/env python3
import json
import os
import random
import sys

# Ensure workspace is in sys.path
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from synthetic_data import SyntheticDataEngine

def check_overlap(new_samples, training_file="synthetic_dataset.json"):
    try:
        with open(training_file, "r") as f:
            training_data = json.load(f)
        training_texts = [item["text"].lower() for item in training_data]
    except:
        training_texts = []
    
    overlaps = []
    for sample in new_samples:
        new_text = sample["text"].lower()
        for train_text in training_texts:
            words_new = set(new_text.split())
            words_train = set(train_text.split())
            if words_new and words_train:
                overlap = len(words_new & words_train) / len(words_new | words_train)
                if overlap > 0.5:
                    overlaps.append((sample["text"], train_text, overlap))
                    break
    return overlaps

def main():
    random.seed(42)
    
    test_domain_pool = [
        "insurance claim description",
        "legal contract clause", 
        "medical intake form notes",
        "call center audio transcript",
        "internal corporate memo",
        "financial audit report footnote",
        "product review forum post",
        "academic research survey response",
        "government tax filing notes",
        "real estate lease agreement text",
        "database migration query log",
        "software system trace log"
    ]
    
    test_labels = ["NAME", "SSN", "EMAIL", "PHONE", "ADDRESS"]
    
    # Dynamically discover custom fine-tuned labels from active local model config
    custom_labels = []
    config_file = os.path.join(workspace_dir, "models", "finetuned-deberta", "config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            id2label = cfg.get("id2label", {})
            for k, v in id2label.items():
                if int(k) >= 111:
                    lbl = v
                    if lbl.startswith("B-") or lbl.startswith("I-"):
                        lbl = lbl[2:]
                    lbl = lbl.upper()
                    if lbl not in custom_labels and lbl != "O":
                        custom_labels.append(lbl)
        except Exception as e:
            print(f"Error detecting custom labels: {e}")
            
    if custom_labels:
        print(f"Detected custom fine-tuned classes: {custom_labels}")
        test_labels.extend(custom_labels)
        
    print("=" * 70)
    print(" SCALED HELD-OUT TEST SET GENERATION (NON-INTERACTIVE)")
    print("=" * 70)
    print(f"Labels to generate: {', '.join(test_labels)}")
    print(f"Domain pool size: {len(test_domain_pool)}")
    print("=" * 70)
    
    engine = SyntheticDataEngine()
    
    # Generate scaled test set of 100 samples
    num_samples = 100
    print(f"\nGenerating {num_samples} samples...")
    
    # Generate the dataset
    dataset = engine.generate_dataset(
        num_samples=num_samples,
        target_labels=test_labels,
        model="groq/llama-3.1-8b-instant",
        domain_pool=test_domain_pool,
        batch_size=10,
        similarity_threshold=0.85
    )
    
    # Check overlaps
    overlaps = check_overlap(dataset)
    if overlaps:
        print(f"\n[WARNING] Found {len(overlaps)} potential overlaps in generated dataset")
    else:
        print("\n[OK] No overlaps detected in generated dataset")
        
    # Save directly to the test set file
    output_file = "test_dataset_heldout.json"
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"\n[OK] Saved {len(dataset)} samples to {output_file}")
    
    # Show stats
    label_counts = {}
    for sample in dataset:
        for ent in sample.get("entities", []):
            lbl = ent.get("label", "UNKNOWN").upper()
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            
    print("\nLabel distribution:")
    for lbl, count in sorted(label_counts.items()):
        print(f"  {lbl}: {count}")

if __name__ == "__main__":
    main()
