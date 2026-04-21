import json
import warnings
import collections
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
)

# --- CONFIGURATION ---
INPUT_FILE = "golden_merged_predictions.json"

# All 18 standard HIPAA Safe Harbor identifiers as defined in the
# de-identification prompt. Tag names map 1-to-1 to the official categories.
# This hardcoded schema ensures:
#   1. The confusion matrix always shows all columns/rows, even for rare
#      or zero-support classes, allowing fair cross-model comparison.
#   2. Any label predicted by a model that is NOT in this set is flagged
#      explicitly as a schema violation (rogue label), eliminating the
#      sklearn "y_pred contains classes not in y_true" UserWarning.
HIPAA_SCHEMA = [
    "ACCOUNT",       # Account numbers (bank accounts, IBAN) — HIPAA identifier #10
    "DATE",          # Dates (except year, unless patient age > 89)
    "DEVICE",        # Device identifiers and serial numbers
    "EMAIL",         # Email addresses
    "FAX",           # Fax numbers
    "HEALTHPLANID",  # Health plan beneficiary numbers
    "IP",            # IP addresses
    "LICENSE",       # Certificate / license / diploma numbers
    "LOCATION",      # Geographic subdivisions, hospital/university names
    "MEDICALID",     # Medical record / protocol / patient numbers
    "NAME",          # Patient, doctor, staff names (incl. titles)
    "OTHERID",       # SGK facility codes, system logs, user codes, other unique IDs
    "PHONE",         # Telephone numbers
    "PHOTO",         # Full-face photographs or comparable images
    "SSN",           # TC Kimlik No / Social Security Numbers
    "URL",           # Web URLs
    "VEHICLE",       # Vehicle identifiers, license plates
    "BIOMETRIC",     # Biometric identifiers (fingerprint, voice print)
]
# Sorted for consistent ordering in reports and confusion matrix axes
HIPAA_SCHEMA = sorted(HIPAA_SCHEMA)


def get_entity_set(result_list):
    """Converts a Label Studio result array (annotations or predictions) into a set of tuples."""
    entities = set()

    if not isinstance(result_list, list):
        return entities

    for item in result_list:
        val = item.get("value", {})
        start = val.get("start")
        end = val.get("end")
        labels = val.get("labels", [])
        label_category = labels[0] if labels else "UNKNOWN"

        if start is not None and end is not None:
            entities.add((start, end, label_category))

    return entities


def compute_per_class_metrics(y_true, y_pred, entity_labels):
    """
    Computes TP, FP, FN, TN, Precision, Recall, F1, and MCC for each entity
    label using one-vs-rest binarisation.

    For each label L:
        positive = L
        negative = everything else (all other entity types + None)

    TN is always large and well-defined because it includes every span that
    is neither truly L nor predicted as L — guaranteed non-zero in any
    realistic corpus.

    Returns a dict {label: {TP, FP, FN, TN, Precision, Recall, F1, MCC}}.
    MCC is 0.0 for labels entirely absent from ground truth.
    """
    metrics = {}
    for label in entity_labels:
        y_true_bin = np.array([1 if t == label else 0 for t in y_true])
        y_pred_bin = np.array([1 if p == label else 0 for p in y_pred])

        tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
        fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
        fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
        tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        mcc       = (matthews_corrcoef(y_true_bin, y_pred_bin)
                     if (tp + fp + fn + tn) > 0 else 0.0)

        metrics[label] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": precision, "Recall": recall, "F1": f1, "MCC": mcc,
        }
    return metrics


def compute_summary_stats(y_true, y_pred, entity_labels):
    """
    Computes valid summary statistics for span-union NER evaluation.

    WHY "Binary PHI vs None" and "Entity-Only" sections are excluded
    ─────────────────────────────────────────────────────────────────
    In span-union evaluation every row is built from:
        all_spans = true_spans.union(pred_spans)
    This makes a (true=None, pred=None) pair structurally impossible —
    every span originated from at least one side.

    Consequences:
    1. Entity-only filter removes zero rows => identical to Overall.
    2. Binary PHI vs None MCC / Balanced Accuracy: TN = 0 always =>
       balanced_accuracy = recall_PHI / 2 (~0.48 even for perfect models),
       MCC is degenerate / negative. Both are mathematically invalid.

    What remains valid:
    • Overall Accuracy / Balanced Accuracy / MCC / Kappa — None acts as
      an error signal, not a true negative.
    • Per-class metrics (one-vs-rest) — TN is always well-defined.
    • Binary PHI F1/P/R from classification_report — does not use TN.
    ─────────────────────────────────────────────────────────────────
    """
    stats = {}
    stats["overall_accuracy"]          = accuracy_score(y_true, y_pred)
    stats["overall_balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    stats["overall_mcc"]               = matthews_corrcoef(y_true, y_pred)
    stats["overall_kappa"]             = cohen_kappa_score(y_true, y_pred)
    stats["per_class_metrics"]         = compute_per_class_metrics(
                                             y_true, y_pred, entity_labels)
    return stats


def format_summary_stats(stats, model_name):
    """Formats the summary statistics dictionary into a readable string block."""

    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A "

    lines = [
        f"Summary Statistics — {model_name}",
        "=" * 70,
        "",
        "[ 1 ] OVERALL  (all spans, all labels incl. None)",
        f"  Accuracy          : {fmt(stats['overall_accuracy'])}",
        f"  Balanced Accuracy : {fmt(stats['overall_balanced_accuracy'])}",
        f"  MCC (multiclass)  : {fmt(stats['overall_mcc'])}",
        f"  Cohen's Kappa     : {fmt(stats['overall_kappa'])}",
        "",
        "[ 2 ] PER-CLASS METRICS  (one-vs-rest per HIPAA entity type)",
        f"  {'Label':<14} {'Prec':>6} {'Rec':>6} {'F1':>6} {'MCC':>6} |"
        f" {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>6}",
        "  " + "-" * 65,
    ]

    for label, m in stats["per_class_metrics"].items():
        lines.append(
            f"  {label:<14} {m['Precision']:>6.4f} {m['Recall']:>6.4f}"
            f" {m['F1']:>6.4f} {m['MCC']:>6.4f} |"
            f" {m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>6}"
        )

    lines.extend([
        "",
        "  Notes:",
        "  • In span-union evaluation (true=None, pred=None) never occurs.",
        "    'None' signals an error (missed or hallucinated span), not TN.",
        "  • Balanced Accuracy averages recall across all label classes,",
        "    reducing sensitivity to class-count imbalance.",
        "  • Per-class metrics use one-vs-rest binarisation; TN always valid.",
        "    MCC near 1.0 = perfect, 0 = random, -1 = always wrong.",
        "  • Binary PHI F1/Precision/Recall (masking report above) is the",
        "    primary privacy-safety metric — unaffected by TN degeneracy.",
        "=" * 70,
    ])
    return "\n".join(lines)


def main():
    print(f"Loading merged file from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(
            f"Error: '{INPUT_FILE}' not found. "
            "Please run ls_merge_gold_predictions.py first."
        )
        return

    model_y_true = collections.defaultdict(list)
    model_y_pred = collections.defaultdict(list)

    print("Evaluating models based on annotations and predictions in the merged file...")

    for task in data:
        annotations = task.get("annotations", [])
        if not annotations:
            continue

        true_result   = annotations[0].get("result", [])
        true_entities = get_entity_set(true_result)
        true_spans    = {(s, e): l for s, e, l in true_entities}

        predictions = task.get("predictions", [])
        for pred in predictions:
            model_name    = pred.get("model_version", "Unknown_Model")
            pred_entities = get_entity_set(pred.get("result", []))
            pred_spans    = {(s, e): l for s, e, l in pred_entities}

            all_spans = set(true_spans.keys()).union(set(pred_spans.keys()))
            for span in all_spans:
                model_y_true[model_name].append(true_spans.get(span, "None"))
                model_y_pred[model_name].append(pred_spans.get(span, "None"))

    if not model_y_true:
        print("No models evaluated. Please check if annotations and predictions exist.")
        return

    for model_name in model_y_true.keys():
        y_true = model_y_true[model_name]
        y_pred = model_y_pred[model_name]

        print("=" * 50)
        print(f"EVALUATION RESULTS FOR MODEL: {model_name}")
        print("=" * 50)

        # -- Schema-driven label resolution ------------------------------
        # entity_labels: schema labels that actually appear in this model's
        # data (true OR pred), keeping HIPAA_SCHEMA order for consistency.
        observed_entity_labels = (set(y_true) | set(y_pred)) - {"None"}
        entity_labels = [l for l in HIPAA_SCHEMA if l in observed_entity_labels]

        # Detect rogue labels — predicted by the model but not in schema.
        # Replaces the sklearn UserWarning with a clear, actionable message.
        rogue_labels = observed_entity_labels - set(HIPAA_SCHEMA)
        if rogue_labels:
            print(f"  [SCHEMA WARNING] Model '{model_name}' predicted label(s) not")
            print(f"  defined in the 18 HIPAA Safe Harbor schema: {sorted(rogue_labels)}")
            print(f"  These spans are counted as FP against every gold class.")
            print(f"  Action: verify the model prompt explicitly lists all 18 HIPAA")
            print(f"  tag names and instructs the model to use only those exact labels.\n")

        # Full label list for confusion matrix: all 18 HIPAA schema labels
        # always present (zero rows/cols for unseen classes), rogue labels
        # appended at the end if any, then "None" last.
        labels = HIPAA_SCHEMA + sorted(rogue_labels) + ["None"]

        # Suppress the sklearn UserWarning now that we handle it ourselves
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="y_pred contains classes not in y_true",
            )

            # -- Exact entity classification report ----------------------
            cls_report = classification_report(
                y_true, y_pred,
                labels=entity_labels + sorted(rogue_labels),
                zero_division=0,
            )
            print("--- Exact Entity Match Report ---")
            print(cls_report)

            # -- Binary masking report (F1/P/R — valid, no TN dependency)
            y_true_bin = ["PHI" if l != "None" else "None" for l in y_true]
            y_pred_bin = ["PHI" if l != "None" else "None" for l in y_pred]
            bin_report = classification_report(
                y_true_bin, y_pred_bin, labels=["PHI"], zero_division=0
            )
            print("--- Binary Masking Report (Any PHI vs None) ---")
            print(bin_report)

            # -- Summary statistics --------------------------------------
            stats       = compute_summary_stats(y_true, y_pred, entity_labels)
            stats_block = format_summary_stats(stats, model_name)
            print(stats_block)

        # -- Confusion matrix (tab-separated text) -----------------------
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (FP + FN + TP)

        header = "Tr\\Pr\t" + "\t".join(labels) + "\tTP\tFP\tFN\tTN\n"
        rows = []
        for i, label in enumerate(labels):
            row = [label] + cm[i].tolist() + [TP[i], FP[i], FN[i], TN[i]]
            rows.append("\t".join(map(str, row)))

        report  = f"Confusion Matrix Report - {model_name}\n========================\n\n"
        report += header + "\n".join(rows)
        report += f"\n\nExact Entity Classification Report:\n{cls_report}"
        report += f"\n\nBinary Masking Report (Any PHI vs None):\n{bin_report}"
        report += f"\n\n{stats_block}\n"

        safe_model_name = "".join([c if c.isalnum() else "_" for c in model_name])
        report_filename = f"confusion_matrix_report_{safe_model_name}.txt"
        with open(report_filename, "w", encoding="utf-8") as file:
            file.write(report)
        print(f"Saved textual report to {report_filename}")

        # -- Confusion matrix heat-map -----------------------------------
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        plot_filename = f"confusion_matrix_{safe_model_name}.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved confusion matrix plot to {plot_filename}\n")


if __name__ == "__main__":
    main()
