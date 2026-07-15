#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Any

from synthetic_data import SyntheticDataEngine

def print_progress(current: int, total: int, stats: Dict[str, Any]):
    """Prints a beautiful live progress update in the terminal."""
    percent = (current / total) * 100
    bar_length = 30
    filled_length = int(bar_length * current // total)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    
    # Extract latest stats
    healed = stats.get("healed_entities", 0)
    attempts = stats.get("total_attempts", 0)
    parsed = stats.get("successful_parses", 0)
    
    # Print status line
    sys.stdout.write(
        f"\rProgress: [{bar}] {current}/{total} ({percent:.1f}%) | "
        f"Attempts: {attempts} | Parsed: {parsed} | Healed Spans: {healed}"
    )
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(
        description="Diversity-Driven Synthetic Data Engine CLI Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=10,
        help="Number of synthetic data samples to generate"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="groq/llama-3.1-8b-instant",
        help="LiteLLM-compatible model identifier to route to"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="synthetic_dataset.json",
        help="Target filepath to save the generated dataset"
    )
    parser.add_argument(
        "--labels", "-l",
        type=str,
        default="NAME,EMAIL,PHONE,SSN,PAN,ADDRESS",
        help="Comma-separated list of target NER entity labels"
    )
    
    args = parser.parse_args()
    
    # Parse labels list
    labels_list = [lbl.strip().upper() for lbl in args.labels.split(",") if lbl.strip()]
    if not labels_list:
        print("Error: No valid labels provided.")
        sys.exit(1)
        
    print("=" * 70)
    print(" DIVERSITY-DRIVEN SYNTHETIC DATA ENGINE")
    print("=" * 70)
    print(f"Target Model:  {args.model}")
    print(f"Sample Count:  {args.num_samples}")
    print(f"Output File:   {args.output}")
    print(f"Target Labels: {', '.join(labels_list)}")
    print("=" * 70)
    
    engine = SyntheticDataEngine()
    
    print("\n[Engine] Starting generation loop. Calling APIs...")
    sys.stdout.write("Progress: [------------------------------] 0/" + str(args.num_samples) + " (0.0%)")
    sys.stdout.flush()
    
    try:
        dataset = engine.generate_dataset(
            num_samples=args.num_samples,
            target_labels=labels_list,
            model=args.model,
            progress_callback=print_progress
        )
        
        # Write to JSON file
        print("\n\n[Engine] Saving generated dataset to file...")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
            
        print(f"[Engine] Success! Saved {len(dataset)} samples to '{args.output}'.")
        
        # Display summary statistics
        print("\n" + "=" * 30 + " GENERATION SUMMARY " + "=" * 30)
        domain_counts = {}
        label_counts = {}
        
        for sample in dataset:
            domain = sample.get("domain", "Unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            for ent in sample.get("entities", []):
                lbl = ent.get("label", "Unknown").upper()
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
                
        print("\nDomain Distribution:")
        for dom, count in sorted(domain_counts.items()):
            pct = (count / len(dataset)) * 100
            print(f"  • {dom:<15}: {count} ({pct:.1f}%)")
            
        print("\nLabel Distribution:")
        total_ents = sum(label_counts.values())
        if total_ents > 0:
            for lbl, count in sorted(label_counts.items()):
                pct = (count / total_ents) * 100
                print(f"  • {lbl:<15}: {count} ({pct:.1f}%)")
        else:
            print("  No entities generated (all decoy samples).")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n[Engine] Generation interrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[Engine] Critical execution failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
