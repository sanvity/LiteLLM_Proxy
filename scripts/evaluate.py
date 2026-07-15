#!/usr/bin/env python3
"""
Evaluation script for comparing base vs fine-tuned DeBERTa PII models.
Uses overlap-based entity matching and generates comprehensive HTML report.
"""
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Add guardrails to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from guardrails.deberta_pii_guardrail import canonicalize_label, get_ner_model_path, load_ner_pipeline

class ModelEvaluator:
    def __init__(self, test_set_path: str = "test_dataset_heldout.json", model_path: str = None):
        self.test_set_path = test_set_path
        self.test_data = self.load_test_data()
        self.base_model = None
        self.finetuned_model = None
        self.threshold = 0.4
        self.model_path = model_path
        
    def load_test_data(self) -> List[Dict]:
        """Load held-out test set."""
        with open(self.test_set_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Filter out hard negatives and neutral samples for evaluation
        return [item for item in data if item.get("entities") and not item.get("is_hard_negative")]
    
    def load_models(self):
        """Load both base and fine-tuned models using existing pipeline."""
        from transformers import pipeline
        
        print("Loading base model: Isotonic/deberta-v3-base_finetuned_ai4privacy_v2")
        self.base_model = pipeline(
            "token-classification",
            model="Isotonic/deberta-v3-base_finetuned_ai4privacy_v2",
            aggregation_strategy="simple"
        )
        
        finetuned_path = self.model_path if self.model_path else get_ner_model_path()
        print(f"Loading fine-tuned model from path: {finetuned_path}")
        self.finetuned_model = load_ner_pipeline(finetuned_path)
        
    def run_inference(self, model, text: str) -> List[Dict]:
        """Run inference on a single text sample."""
        results = model(text)
        entities = []
        for r in results:
            score = float(r.get("score", 0.0))
            if score < self.threshold:
                continue
                
            model_label = r.get("entity_group", "")
            label = canonicalize_label(model_label)
            
            entities.append({
                "label": label,
                "start": int(r["start"]),
                "end": int(r["end"]),
                "score": score,
                "text": r.get("word", "")
            })
        
        return sorted(entities, key=lambda e: e["start"])
    
    def calculate_iou(self, ent1: Dict, ent2: Dict) -> float:
        """Calculate Intersection over Union for two entity spans."""
        start1, end1 = ent1["start"], ent1["end"]
        start2, end2 = ent2["start"], ent2["end"]
        
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection = max(0, intersection_end - intersection_start)
        
        union_start = min(start1, start2)
        union_end = max(end1, end2)
        union = union_end - union_start
        
        return intersection / union if union > 0 else 0.0
    
    def match_entities(self, predictions: List[Dict], ground_truth: List[Dict]) -> Tuple[List[Tuple], List, List]:
        """
        Match predictions to ground truth using IoU ≥ 0.5.
        Returns: (matched_pairs, false_positives, false_negatives)
        """
        matched_pairs = []
        used_predictions = set()
        used_ground_truth = set()
        
        # Sort by score for greedy matching
        sorted_preds = sorted(enumerate(predictions), key=lambda x: -x[1].get("score", 0))
        
        for pred_idx, pred in sorted_preds:
            if pred_idx in used_predictions:
                continue
                
            best_match = None
            best_iou = 0.0
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in used_ground_truth:
                    continue
                    
                iou = self.calculate_iou(pred, gt)
                if iou >= 0.5 and iou > best_iou:
                    best_iou = iou
                    best_match = (pred_idx, gt_idx, iou)
            
            if best_match:
                pred_idx, gt_idx, iou = best_match
                matched_pairs.append((predictions[pred_idx], ground_truth[gt_idx], iou))
                used_predictions.add(pred_idx)
                used_ground_truth.add(gt_idx)
        
        false_positives = [predictions[i] for i in range(len(predictions)) if i not in used_predictions]
        false_negatives = [ground_truth[i] for i in range(len(ground_truth)) if i not in used_ground_truth]
        
        return matched_pairs, false_positives, false_negatives
    
    def calculate_metrics(self, model_results: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics per entity type and aggregated."""
        entity_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
        total_tokens = 0
        
        # Calculate character-level sequence accuracy (TP+TN)/(TP+TN+FP+FN)
        total_chars = 0
        correct_chars = 0
        
        for sample_result in model_results:
            text = sample_result["text"]
            predictions = sample_result["predictions"]
            ground_truth = sample_result["ground_truth"]
            
            n = len(text)
            total_chars += n
            
            gt_chars = ["O"] * n
            pred_chars = ["O"] * n
            
            for ent in ground_truth:
                start, end = ent["start"], ent["end"]
                lbl = canonicalize_label(ent["label"])
                for i in range(max(0, start), min(n, end)):
                    gt_chars[i] = lbl
                    
            for ent in predictions:
                start, end = ent["start"], ent["end"]
                lbl = canonicalize_label(ent["label"])
                for i in range(max(0, start), min(n, end)):
                    pred_chars[i] = lbl
                    
            correct_chars += sum(1 for i in range(n) if gt_chars[i] == pred_chars[i])
            
            # Count tokens for TN calculation
            total_tokens += len(text.split())
            
            matched, fp, fn = self.match_entities(predictions, ground_truth)
            
            # Process matched pairs
            for pred, gt, iou in matched:
                pred_label = canonicalize_label(pred["label"])
                gt_label = canonicalize_label(gt["label"])
                
                if pred_label == gt_label:
                    entity_metrics[gt_label]["tp"] += 1
                else:
                    # Label mismatch - count as FP for pred label, FN for gt label
                    entity_metrics[pred_label]["fp"] += 1
                    entity_metrics[gt_label]["fn"] += 1
                
                entity_metrics[gt_label]["support"] += 1
            
            # Process false positives
            for fp_ent in fp:
                fp_label = canonicalize_label(fp_ent["label"])
                entity_metrics[fp_label]["fp"] += 1
            
            # Process false negatives
            for fn_ent in fn:
                fn_label = canonicalize_label(fn_ent["label"])
                entity_metrics[fn_label]["fn"] += 1
                entity_metrics[fn_label]["support"] += 1
        
        # Calculate precision, recall, F1
        metrics_summary = {}
        for label, counts in entity_metrics.items():
            tp, fp, fn, support = counts["tp"], counts["fp"], counts["fn"], counts["support"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics_summary[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": support
            }
        
        # Calculate aggregated metrics
        total_tp = sum(m["tp"] for m in entity_metrics.values())
        total_fp = sum(m["fp"] for m in entity_metrics.values())
        total_fn = sum(m["fn"] for m in entity_metrics.values())
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        # Macro average
        macro_precision = np.mean([m["precision"] for m in metrics_summary.values()])
        macro_recall = np.mean([m["recall"] for m in metrics_summary.values()])
        macro_f1 = np.mean([m["f1"] for m in metrics_summary.values()])
        
        accuracy = correct_chars / total_chars if total_chars > 0 else 1.0
        
        return {
            "per_entity": metrics_summary,
            "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
            "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
            "accuracy": accuracy
        }
    
    def evaluate_model(self, model) -> Tuple[Dict, List[Dict]]:
        """Evaluate a single model on the test set."""
        results = []
        
        for sample in self.test_data:
            text = sample["text"]
            ground_truth = sample["entities"]
            
            predictions = self.run_inference(model, text)
            
            results.append({
                "text": text,
                "ground_truth": ground_truth,
                "predictions": predictions,
                "domain": sample.get("domain", "Unknown")
            })
        
        metrics = self.calculate_metrics(results)
        return metrics, results
    
    def get_confusion_matrix(self, base_results: List[Dict], finetuned_results: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """Generate confusion matrix for entity type misclassification."""
        all_labels = set()
        
        # Collect all labels
        for result in base_results + finetuned_results:
            for ent in result["ground_truth"] + result["predictions"]:
                all_labels.add(canonicalize_label(ent["label"]))
        
        labels = sorted(list(all_labels))
        label_to_idx = {label: i for i, label in enumerate(labels)}
        
        # Build confusion matrix from fine-tuned results
        cm = np.zeros((len(labels), len(labels)))
        
        for result in finetuned_results:
            matched, fp, fn = self.match_entities(result["predictions"], result["ground_truth"])
            
            for pred, gt, iou in matched:
                pred_label = canonicalize_label(pred["label"])
                gt_label = canonicalize_label(gt["label"])
                
                if pred_label in label_to_idx and gt_label in label_to_idx:
                    cm[label_to_idx[gt_label], label_to_idx[pred_label]] += 1
        
        return cm, labels
    
    def generate_error_analysis(self, base_results: List[Dict], finetuned_results: List[Dict]) -> Dict[str, List]:
        """Generate concrete error examples."""
        fixed_misses = []
        new_false_positives = []
        
        for base_res, ft_res in zip(base_results, finetuned_results):
            base_matched, base_fp, base_fn = self.match_entities(base_res["predictions"], base_res["ground_truth"])
            ft_matched, ft_fp, ft_fn = self.match_entities(ft_res["predictions"], ft_res["ground_truth"])
            
            # Cases where fine-tuning fixed misses
            for fn_ent in base_fn:
                # Check if this was caught in fine-tuned
                was_fixed = False
                for ft_pred in ft_res["predictions"]:
                    if self.calculate_iou(fn_ent, ft_pred) >= 0.5:
                        was_fixed = True
                        break
                
                if was_fixed and len(fixed_misses) < 5:
                    fixed_misses.append({
                        "text": base_res["text"],
                        "entity": fn_ent,
                        "domain": base_res["domain"]
                    })
            
            # Cases where fine-tuning introduced new false positives
            for fp_ent in ft_fp:
                # Check if this was NOT in base false positives
                was_in_base = False
                for base_pred in base_res["predictions"]:
                    if self.calculate_iou(fp_ent, base_pred) >= 0.5:
                        was_in_base = True
                        break
                
                if not was_in_base and len(new_false_positives) < 5:
                    new_false_positives.append({
                        "text": ft_res["text"],
                        "entity": fp_ent,
                        "domain": ft_res["domain"]
                    })
        
        return {"fixed_misses": fixed_misses, "new_false_positives": new_false_positives}
    
    def generate_html_report(self, base_metrics: Dict, ft_metrics: Dict, cm: np.ndarray, labels: List[str], error_analysis: Dict):
        """Generate comprehensive HTML report."""
        table_rows = ""
        entity_labels = []
        base_f1_scores = []
        ft_f1_scores = []
        
        all_entities = set(base_metrics["per_entity"].keys()) | set(ft_metrics["per_entity"].keys())
        
        for entity in sorted(all_entities):
            base_ent = base_metrics["per_entity"].get(entity, {"precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": 0, "fn": 0, "support": 0})
            ft_ent = ft_metrics["per_entity"].get(entity, {"precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": 0, "fn": 0, "support": 0})
            
            delta = ft_ent["f1"] - base_ent["f1"]
            delta_class = "delta-pos" if delta > 0 else ("delta-neg" if delta < 0 else "delta-neutral")
            delta_str = f"{delta:+.3f}" if delta != 0 else "0.000"
            
            entity_labels.append(entity)
            base_f1_scores.append(base_ent["f1"])
            ft_f1_scores.append(ft_ent["f1"])
            
            table_rows += f"""
            <tr>
                <td class="entity-name">{entity}</td>
                <td>
                    <div class="compare-value">
                        <span class="val-base">{base_ent["precision"]:.3f}</span>
                        <span class="val-arrow">→</span>
                        <span class="val-ft">{ft_ent["precision"]:.3f}</span>
                    </div>
                </td>
                <td>
                    <div class="compare-value">
                        <span class="val-base">{base_ent["recall"]:.3f}</span>
                        <span class="val-arrow">→</span>
                        <span class="val-ft">{ft_ent["recall"]:.3f}</span>
                    </div>
                </td>
                <td>
                    <div class="compare-value">
                        <span class="val-base">{base_ent["f1"]:.3f}</span>
                        <span class="val-arrow">→</span>
                        <span class="val-ft">{ft_ent["f1"]:.3f}</span>
                    </div>
                </td>
                <td>
                    <span class="delta-badge {delta_class}">{delta_str}</span>
                </td>
                <td class="cell-support">{ft_ent["support"]}</td>
            </tr>
            """
            
        fixed_misses_html = ""
        for example in error_analysis["fixed_misses"]:
            fixed_misses_html += f"""
            <div class="example-card success-card">
                <div class="card-meta">
                    <span class="badge badge-domain">Domain: {example["domain"]}</span>
                    <span class="badge badge-fixed">Fixed Miss</span>
                </div>
                <div class="card-body">"{example["text"]}"</div>
                <div class="card-footer">
                    <span>Missed entity caught by Fine-tuned: <b>{example["entity"]["label"]}</b> ({example["entity"].get("text", "")})</span>
                </div>
            </div>
            """
            
        new_false_positives_html = ""
        for example in error_analysis["new_false_positives"]:
            new_false_positives_html += f"""
            <div class="example-card error-card">
                <div class="card-meta">
                    <span class="badge badge-domain">Domain: {example["domain"]}</span>
                    <span class="badge badge-fp">New False Positive</span>
                </div>
                <div class="card-body">"{example["text"]}"</div>
                <div class="card-footer">
                    <span>Fine-tuned predicted: <b>{example["entity"]["label"]}</b> ({example["entity"]["text"]}) [score: {example["entity"]["score"]:.3f}]</span>
                </div>
            </div>
            """
            
        base_macro_f1 = base_metrics["macro"]["f1"]
        ft_macro_f1 = ft_metrics["macro"]["f1"]
        delta = ft_macro_f1 - base_macro_f1
        delta_class = "delta-pos" if delta > 0 else ("delta-neg" if delta < 0 else "delta-neutral")
        delta_str = f"{delta:+.3f}"
        
        base_accuracy = base_metrics.get("accuracy", 0.0)
        ft_accuracy = ft_metrics.get("accuracy", 0.0)
        delta_acc = ft_accuracy - base_accuracy
        delta_acc_class = "delta-pos" if delta_acc > 0 else ("delta-neg" if delta_acc < 0 else "delta-neutral")
        delta_acc_str = f"{delta_acc:+.2%}"
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DeBERTa PII Model Evaluation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --border-color: #e2e8f0;
            --primary: #4f46e5;
            --primary-light: #e0e7ff;
            --success: #10b981;
            --success-light: #ecfdf5;
            --danger: #ef4444;
            --danger-light: #fef2f2;
            --warning: #f59e0b;
            --warning-light: #fef3c7;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            line-height: 1.4;
            padding: 20px 10px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 80%);
            pointer-events: none;
        }}
        
        .header-title {{
            font-size: 1.6rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .header-sub {{
            font-size: 0.85rem;
            color: #c7d2fe;
            font-weight: 400;
        }}
        
        .section-title {{
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: 20px 0 10px 0;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 5px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 12px;
            margin-bottom: 15px;
        }}
        
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 12px 16px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        
        .card-label {{
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }}
        
        .card-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 4px;
            display: flex;
            align-items: baseline;
            gap: 10px;
        }}
        
        .compare-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        
        .delta-badge {{
            display: inline-flex;
            align-items: center;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 9999px;
        }}
        
        .delta-pos {{
            background-color: var(--success-light);
            color: var(--success);
        }}
        
        .delta-neg {{
            background-color: var(--danger-light);
            color: var(--danger);
        }}
        
        .delta-neutral {{
            background-color: #f1f5f9;
            color: #64748b;
        }}
        
        .tabs {{
            display: flex;
            border-bottom: 2px solid var(--border-color);
            margin-bottom: 15px;
            gap: 8px;
        }}
        
        .tab {{
            padding: 8px 16px;
            font-weight: 500;
            font-size: 0.85rem;
            color: var(--text-secondary);
            border: none;
            background: none;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s ease;
            margin-bottom: -2px;
        }}
        
        .tab:hover {{
            color: var(--primary);
        }}
        
        .tab.active {{
            color: var(--primary);
            border-bottom-color: var(--primary);
            font-weight: 600;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .table-container {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            overflow: hidden;
            margin-bottom: 15px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}
        
        th {{
            background-color: #f8fafc;
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 10px 14px;
            border-bottom: 2px solid var(--border-color);
        }}
        
        td {{
            padding: 10px 14px;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.85rem;
        }}
        
        tr:last-child td {{
            border-bottom: none;
        }}
        
        tr:hover td {{
            background-color: #f8fafc;
        }}
        
        .entity-name {{
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .compare-value {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .val-base {{
            color: var(--text-secondary);
        }}
        
        .val-arrow {{
            color: #cbd5e1;
            font-size: 0.85rem;
        }}
        
        .val-ft {{
            font-weight: 600;
            color: var(--primary);
        }}
        
        .cell-support {{
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
        }}
        
        .chart-box {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 15px;
        }}
        
        .charts-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 15px;
        }}
        
        .charts-row .chart-box {{
            margin-bottom: 0;
        }}
        
        @media (max-width: 1024px) {{
            .charts-row {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .examples-list {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 12px;
        }}
        
        .example-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        
        .success-card {{
            border-left: 5px solid var(--success);
        }}
        
        .error-card {{
            border-left: 5px solid var(--danger);
        }}
        
        .card-meta {{
            display: flex;
            gap: 10px;
            margin-bottom: 12px;
        }}
        
        .badge {{
            display: inline-flex;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }}
        
        .badge-domain {{
            background-color: var(--primary-light);
            color: var(--primary);
        }}
        
        .badge-fixed {{
            background-color: var(--success-light);
            color: var(--success);
        }}
        
        .badge-fp {{
            background-color: var(--danger-light);
            color: var(--danger);
        }}
        
        .card-body {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            color: var(--text-primary);
            background-color: #f8fafc;
            padding: 14px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            margin-bottom: 12px;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        
        .card-footer {{
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="header-title">🛡️ DeBERTa PII Model Evaluation Report</h1>
            <div class="header-sub">Robust validation of base vs fine-tuned classification heads on held-out test sets</div>
        </div>
        
        <div class="metrics-grid">
            <div class="card">
                <div>
                    <div class="card-label">Macro F1</div>
                    <div class="card-value">
                        <span>{ft_macro_f1:.3f}</span>
                        <span class="delta-badge {delta_class}">{delta_str}</span>
                    </div>
                </div>
                <div class="compare-label">Base model: {base_macro_f1:.3f}</div>
            </div>
            
            <div class="card">
                <div>
                    <div class="card-label">Macro Precision</div>
                    <div class="card-value">
                        <span>{ft_metrics["macro"]["precision"]:.3f}</span>
                        <span class="delta-badge {('delta-pos' if (ft_metrics['macro']['precision'] - base_metrics['macro']['precision']) > 0 else ('delta-neg' if (ft_metrics['macro']['precision'] - base_metrics['macro']['precision']) < 0 else 'delta-neutral'))}">
                            {ft_metrics["macro"]["precision"] - base_metrics["macro"]["precision"]:+.3f}
                        </span>
                    </div>
                </div>
                <div class="compare-label">Base model: {base_metrics["macro"]["precision"]:.3f}</div>
            </div>
            
            <div class="card">
                <div>
                    <div class="card-label">Macro Recall</div>
                    <div class="card-value">
                        <span>{ft_metrics["macro"]["recall"]:.3f}</span>
                        <span class="delta-badge {('delta-pos' if (ft_metrics['macro']['recall'] - base_metrics['macro']['recall']) > 0 else ('delta-neg' if (ft_metrics['macro']['recall'] - base_metrics['macro']['recall']) < 0 else 'delta-neutral'))}">
                            {ft_metrics["macro"]["recall"] - base_metrics["macro"]["recall"]:+.3f}
                        </span>
                    </div>
                </div>
                <div class="compare-label">Base model: {base_metrics["macro"]["recall"]:.3f}</div>
            </div>
            
            <div class="card">
                <div>
                    <div class="card-label">Classification Accuracy</div>
                    <div class="card-value">
                        <span>{ft_accuracy:.2%}</span>
                        <span class="delta-badge {delta_acc_class}">{delta_acc_str}</span>
                    </div>
                </div>
                <div class="compare-label">Base model: {base_accuracy:.2%}</div>
            </div>
            
            <div class="card">
                <div>
                    <div class="card-label">Micro F1</div>
                    <div class="card-value">
                        <span>{ft_metrics["micro"]["f1"]:.3f}</span>
                        <span class="delta-badge {('delta-pos' if (ft_metrics['micro']['f1'] - base_metrics['micro']['f1']) > 0 else ('delta-neg' if (ft_metrics['micro']['f1'] - base_metrics['micro']['f1']) < 0 else 'delta-neutral'))}">
                            {ft_metrics["micro"]["f1"] - base_metrics["micro"]["f1"]:+.3f}
                        </span>
                    </div>
                </div>
                <div class="compare-label">Base model: {base_metrics["micro"]["f1"]:.3f}</div>
            </div>
            
            <div class="card">
                <div>
                    <div class="card-label">Micro Precision</div>
                    <div class="card-value">
                        <span>{ft_metrics["micro"]["precision"]:.3f}</span>
                        <span class="delta-badge {('delta-pos' if (ft_metrics['micro']['precision'] - base_metrics['micro']['precision']) > 0 else ('delta-neg' if (ft_metrics['micro']['precision'] - base_metrics['micro']['precision']) < 0 else 'delta-neutral'))}">
                            {ft_metrics["micro"]["precision"] - base_metrics["micro"]["precision"]:+.3f}
                        </span>
                    </div>
                </div>
                <div class="compare-label">Base model: {base_metrics["micro"]["precision"]:.3f}</div>
            </div>
            
            <div class="card">
                <div>
                    <div class="card-label">Micro Recall</div>
                    <div class="card-value">
                        <span>{ft_metrics["micro"]["recall"]:.3f}</span>
                        <span class="delta-badge {('delta-pos' if (ft_metrics['micro']['recall'] - base_metrics['micro']['recall']) > 0 else ('delta-neg' if (ft_metrics['micro']['recall'] - base_metrics['micro']['recall']) < 0 else 'delta-neutral'))}">
                            {ft_metrics["micro"]["recall"] - base_metrics["micro"]["recall"]:+.3f}
                        </span>
                    </div>
                </div>
                <div class="compare-label">Base model: {base_metrics["micro"]["recall"]:.3f}</div>
            </div>
            
            <div class="card">
                <div>
                    <div class="card-label">Total Test Support</div>
                    <div class="card-value">
                        <span>{sum(m["support"] for m in ft_metrics["per_entity"].values())}</span>
                    </div>
                </div>
                <div class="compare-label">Total labeled PII instances evaluated</div>
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="switchTab(event, 'tab-summary')">📈 Summary Dashboard</button>
            <button class="tab" onclick="switchTab(event, 'tab-errors')">❌ Error Analysis</button>
        </div>
        
        <div id="tab-summary" class="tab-content active">
            <div class="charts-row">
                <div class="chart-box">
                    <div id="f1-chart"></div>
                </div>
                <div class="chart-box">
                    <div id="confusion-chart"></div>
                </div>
            </div>
            
            <div class="table-container" style="margin-top: 30px;">
                <table>
                    <thead>
                        <tr>
                            <th>Entity Type</th>
                            <th>Precision (Base → FT)</th>
                            <th>Recall (Base → FT)</th>
                            <th>F1 Score (Base → FT)</th>
                            <th>F1 Delta</th>
                            <th>Support</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="tab-errors" class="tab-content">
            <h3 style="margin-bottom:15px; font-weight:600; font-size:1.15rem; color:var(--text-primary);">✅ Fixed Misses (Fine-tuned Caught)</h3>
            <div class="examples-list" style="margin-bottom:40px;">
                {fixed_misses_html or "<p style='color:var(--text-secondary);'>No examples found.</p>"}
            </div>
            
            <h3 style="margin-bottom:15px; font-weight:600; font-size:1.15rem; color:var(--text-primary);">❌ New False Positives Introduced</h3>
            <div class="examples-list">
                {new_false_positives_html or "<p style='color:var(--text-secondary);'>No examples found.</p>"}
            </div>
        </div>
    </div>
    
    <script>
        function switchTab(evt, tabId) {{
            var tabContent = document.getElementsByClassName("tab-content");
            for (var i = 0; i < tabContent.length; i++) {{
                tabContent[i].classList.remove("active");
            }}
            
            var tabs = document.getElementsByClassName("tab");
            for (var i = 0; i < tabs.length; i++) {{
                tabs[i].classList.remove("active");
            }}
            
            document.getElementById(tabId).classList.add("active");
            evt.currentTarget.classList.add("active");
            
            window.dispatchEvent(new Event('resize'));
        }}
        
        var f1Trace1 = {{
            x: {entity_labels},
            y: {base_f1_scores},
            name: 'Base Model',
            type: 'bar',
            marker: {{color: '#94a3b8'}}
        }};
        var f1Trace2 = {{
            x: {entity_labels},
            y: {ft_f1_scores},
            name: 'Fine-tuned Model',
            type: 'bar',
            marker: {{color: '#4f46e5'}}
        }};
        var f1Layout = {{
            title: {{
                text: 'F1 Score Comparison by Entity Type',
                font: {{ family: "'Inter', sans-serif", size: 16, weight: 'bold' }}
            }},
            font: {{ family: "'Inter', sans-serif" }},
            barmode: 'group',
            xaxis: {{title: 'Entity Type', tickangle: -25}},
            yaxis: {{title: 'F1 Score', range: [0, 1.05]}},
            margin: {{l: 50, r: 50, b: 100, t: 80, pad: 4}},
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)'
        }};
        Plotly.newPlot('f1-chart', [f1Trace1, f1Trace2], f1Layout);
        
        var zData = {cm.tolist()};
        var xLabels = {labels};
        var yLabels = {labels};
        
        var annotations = [];
        var maxVal = 0;
        for (var i = 0; i < zData.length; i++) {{
            for (var j = 0; j < zData[i].length; j++) {{
                if (zData[i][j] > maxVal) maxVal = zData[i][j];
            }}
        }}

        for (var i = 0; i < yLabels.length; i++) {{
            for (var j = 0; j < xLabels.length; j++) {{
                var val = zData[i][j];
                var textColor = val > (maxVal / 2) ? '#ffffff' : '#1f2937';
                annotations.push({{
                    x: xLabels[j],
                    y: yLabels[i],
                    text: val > 0 ? val.toString() : '',
                    font: {{
                        family: "'Inter', sans-serif",
                        size: 10,
                        color: textColor,
                        weight: 'bold'
                    }},
                    showarrow: false
                }});
            }}
        }}

        var confusionData = [{{
            z: zData,
            x: xLabels,
            y: yLabels,
            type: 'heatmap',
            colorscale: [
                [0.0, '#ffffff'],
                [0.1, '#e0e7ff'],
                [0.5, '#4f46e5'],
                [1.0, '#1e1b4b']
            ],
            showscale: true,
            colorbar: {{
                title: 'Count',
                titleside: 'top'
            }},
            hoverinfo: 'text',
            text: zData.map(function(row, i) {{
                return row.map(function(val, j) {{
                    return 'True Label: ' + yLabels[i] + '<br>Predicted Label: ' + xLabels[j] + '<br>Count: ' + val;
                }});
            }})
        }}];
        
        var confusionLayout = {{
            title: {{
                text: 'Entity Type Confusion Matrix (Fine-tuned)',
                font: {{ family: "'Inter', sans-serif", size: 16, weight: 'bold' }}
            }},
            font: {{ family: "'Inter', sans-serif" }},
            xaxis: {{
                title: 'Predicted Label',
                tickangle: -45
            }},
            yaxis: {{title: 'True Label'}},
            margin: {{l: 150, r: 50, b: 120, t: 80, pad: 4}},
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            annotations: annotations
        }};
        Plotly.newPlot('confusion-chart', confusionData, confusionLayout);
    </script>
</body>
</html>"""
        
        return html

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate DeBERTa model.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model to evaluate")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation reports")
    args = parser.parse_args()

    print("=" * 70)
    print(" DEBERTa PII MODEL EVALUATION")
    print("=" * 70)
    
    # Resolve test set path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_set_path = os.path.abspath(os.path.join(script_dir, "..", "data", "test_dataset_heldout.json"))
    
    evaluator = ModelEvaluator(test_set_path=test_set_path, model_path=args.model_path)
    print(f"Loaded {len(evaluator.test_data)} test samples")
    
    print("\nLoading models...")
    evaluator.load_models()
    
    print("\nEvaluating base model...")
    base_metrics, base_results = evaluator.evaluate_model(evaluator.base_model)
    print(f"Base Model Macro F1: {base_metrics['macro']['f1']:.3f}")
    
    print("\nEvaluating fine-tuned model...")
    ft_metrics, ft_results = evaluator.evaluate_model(evaluator.finetuned_model)
    print(f"Fine-tuned Model Macro F1: {ft_metrics['macro']['f1']:.3f}")
    
    print("\nGenerating confusion matrix...")
    cm, labels = evaluator.get_confusion_matrix(base_results, ft_results)
    
    print("\nGenerating error analysis...")
    error_analysis = evaluator.generate_error_analysis(base_results, ft_results)
    
    print("\nGenerating HTML report...")
    html_report = evaluator.generate_html_report(base_metrics, ft_metrics, cm, labels, error_analysis)
    
    default_out_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))
    out_dir = args.output_dir if args.output_dir else default_out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    output_file = os.path.join(out_dir, "evaluation_report.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_report)
    
    # Generate JSON data for frontend consumption
    print("\nGenerating JSON data for frontend...")
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "base_metrics": base_metrics,
        "finetuned_metrics": ft_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": labels,
        "error_analysis": error_analysis
    }
    
    json_file = os.path.join(out_dir, "evaluation_data.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
        
    # Also save to root directory for default frontend compatibility if output_dir is not root
    if os.path.abspath(out_dir) != default_out_dir:
        try:
            import shutil
            shutil.copy(output_file, os.path.join(default_out_dir, "evaluation_report.html"))
            shutil.copy(json_file, os.path.join(default_out_dir, "evaluation_data.json"))
        except Exception as copy_err:
            print(f"Warning: could not copy report/json to root dir: {copy_err}")
    
    print(f"\n[SUCCESS] Evaluation complete! Report saved to {output_file}")
    print(f"[SUCCESS] JSON data saved to {json_file}")
    print(f"  Base Model Macro F1: {base_metrics['macro']['f1']:.3f}")
    print(f"  Fine-tuned Model Macro F1: {ft_metrics['macro']['f1']:.3f}")
    print(f"  Improvement: {ft_metrics['macro']['f1'] - base_metrics['macro']['f1']:+.3f}")

if __name__ == "__main__":
    main()
