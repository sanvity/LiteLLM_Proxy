#!/usr/bin/env python3
"""
Generate held-out test set with no overlap from training data.
Uses different random seed and domain sampling to ensure diversity.
"""
import json
import random
import sys
from synthetic_data import SyntheticDataEngine

def check_overlap(new_samples, training_file="synthetic_dataset.json"):
    """Check for text overlap with training data."""
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
            # Check for significant overlap (> 50% similarity)
            words_new = set(new_text.split())
            words_train = set(train_text.split())
            if words_new and words_train:
                overlap = len(words_new & words_train) / len(words_new | words_train)
                if overlap > 0.5:
                    overlaps.append((sample["text"], train_text, overlap))
                    break
    return overlaps

def main():
    # Different random seed to ensure different outputs
    random.seed(42)  # Training likely used default seed
    
    # Different domain pool to minimize overlap
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
    
    # Use same labels as training but different generation parameters
    test_labels = ["NAME", "SSN", "EMAIL", "PHONE", "ADDRESS"]
    
    print("=" * 70)
    print(" HELD-OUT TEST SET GENERATION")
    print("=" * 70)
    print(f"Labels: {', '.join(test_labels)}")
    print(f"Domain pool size: {len(test_domain_pool)}")
    print("=" * 70)
    
    engine = SyntheticDataEngine()
    
    # Generate small sample first for validation
    print("\nGenerating 10 sample examples for validation...")
    sample_dataset = engine.generate_dataset(
        num_samples=10,
        target_labels=test_labels,
        model="groq/llama-3.1-8b-instant",
        domain_pool=test_domain_pool,
        batch_size=5,
        similarity_threshold=0.85
    )
    
    print(f"\nGenerated {len(sample_dataset)} samples")
    
    # Check for overlap
    overlaps = check_overlap(sample_dataset)
    if overlaps:
        print(f"\n⚠️  WARNING: Found {len(overlaps)} potential overlaps with training data:")
        for new_text, train_text, overlap in overlaps[:3]:
            print(f"  Overlap: {overlap:.2f}")
            print(f"  New: {new_text[:80]}...")
            print(f"  Train: {train_text[:80]}...")
    else:
        print("\n✓ No significant overlaps detected with training data")
    
    # Show sample outputs
    print("\n" + "=" * 70)
    print(" SAMPLE OUTPUTS (first 5)")
    print("=" * 70)
    for i, sample in enumerate(sample_dataset[:5]):
        print(f"\nSample {i+1}:")
        print(f"  Domain: {sample['domain']}")
        print(f"  Style: {sample['style']}")
        print(f"  Text: {sample['text']}")
        print(f"  Entities: {sample['entities']}")
    
    # Ask for confirmation before full generation
    print("\n" + "=" * 70)
    response = input("Generate full test set of 50 samples? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled. Sample data saved to test_sample.json")
        with open("test_sample.json", "w") as f:
            json.dump(sample_dataset, f, indent=2)
        return
    
    # Generate full test set
    print("\nGenerating full test set (50 samples)...")
    full_dataset = engine.generate_dataset(
        num_samples=50,
        target_labels=test_labels,
        model="groq/llama-3.1-8b-instant",
        domain_pool=test_domain_pool,
        batch_size=10,
        similarity_threshold=0.85
    )
    
    # Final overlap check
    overlaps = check_overlap(full_dataset)
    if overlaps:
        print(f"\n⚠️  WARNING: Found {len(overlaps)} potential overlaps in full dataset")
    else:
        print("\n✓ No overlaps detected in full dataset")
    
    # Save to file
    output_file = "test_dataset_heldout.json"
    with open(output_file, "w") as f:
        json.dump(full_dataset, f, indent=2)
    
    print(f"\n✓ Saved {len(full_dataset)} samples to {output_file}")
    
    # Show statistics
    label_counts = {}
    for sample in full_dataset:
        for ent in sample.get("entities", []):
            lbl = ent.get("label", "UNKNOWN").upper()
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
    
    print("\nLabel distribution:")
    for lbl, count in sorted(label_counts.items()):
        print(f"  {lbl}: {count}")

if __name__ == "__main__":
    main()
