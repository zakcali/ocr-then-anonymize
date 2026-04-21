# Anonymization Statistics & Evaluation

This directory contains the tools and outputs for mathematically evaluating the performance of the local LLMs at detecting HIPAA Protected Health Information (PHI) compared to human "Gold Standard" annotations.

## Directory Structure

```text
statistics/
├── ls_analyze_annotations.py         # The core evaluation script
├── golden_merged_predictions.json    # Input dataset (Output from the label-studio pipeline)
└── analyzes/                         # Pre-computed evaluation results
    ├── console-output.txt            # The raw stdout from running the script
    ├── confusion_matrix_report_*.txt # Detailed text reports for each model
    └── confusion_matrix_*.png        # Heatmap visualizations of the confusion matrices
```

## How It Works (`ls_analyze_annotations.py`)

The evaluation script relies on **span-union evaluation**. It compares the precise character coordinates predicted by the models against the coordinates marked by the human labeler.

### Metrics Calculated:
1. **Exact Entity Match Report:** Evaluates the model's ability to not only find the PHI but also correctly classify it into the exact correct HIPAA category (e.g., Did it label a `PHONE` as a `PHONE`, or mistakenly as an `ACCOUNT`?). 
   * Provides Precision, Recall, and F1-Score per class.
2. **Binary Masking Report (Any PHI vs None):** This is the most critical metric for privacy safety. It ignores classification errors (e.g., calling a Phone number a Date) and only asks: *Did the model redact the sensitive span?*
3. **Summary Statistics:** Provides Overall Accuracy, Balanced Accuracy, Matthews Correlation Coefficient (MCC), and Cohen's Kappa.

### Handled Edge Cases:
* **Rogue Labels:** If a model hallucinates a tag outside the 18 standard HIPAA categories (e.g., inventing `<DOCTOR>`), the script intercepts this, flags it as a schema warning, and counts the spans as False Positives against the Gold Standard, eliminating generic scikit-learn warnings.
* **True Negative (TN) Degeneracy:** In span-union NER evaluation, a "True Negative" is mathematically undefined at the dataset level (you cannot count the infinite number of words that *aren't* PHI). The script specifically handles this to prevent artificially inflated accuracy scores.

## Reviewing the Results

All pre-computed results for the evaluated models are located in the `analyzes/` folder.

- **For a quick overview:** Read the `console-output.txt` file.
- **For deep dives into specific model failures:** Check the `confusion_matrix_report_[model_name].txt` files to see exactly where the model produced False Positives (over-redacting) or False Negatives (leaking PHI).
- **For visual representation:** Open the `.png` heatmaps to visually identify which HIPAA categories models confuse the most (e.g., frequently confusing `MEDICALID` with `OTHERID`).

## Usage

If you run new models and generate a new `golden_merged_predictions.json` file (via the scripts in the `label-studio/` directory), you can recalculate the statistics by running:

```bash
# Run the evaluation
python ls_analyze_annotations.py >console-output.txt

# Folder
You may move the output files to analyzes folder
```

The script will automatically parse the JSON, evaluate every model present within it, and drop the new `.txt` and `.png` files into your working directory.
