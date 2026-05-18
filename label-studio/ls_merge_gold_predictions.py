import json

# --- CONFIGURATION ---
PREDICTIONS_FILE = "predictions.json"
GOLDEN_FILE = "golden-standard.json"
OUTPUT_FILE = "golden_merged_predictions.json"

def get_entity_set_golden(label_list):
    """Converts a Label Studio MIN-JSON label array into a list of tuples."""
    entity_details = []
    
    # Check if label_list is actually a list, handle empty cases
    if not isinstance(label_list, list):
        return entity_details

    for item in label_list:
        start = item.get("start")
        end = item.get("end")
        text = item.get("text", "")
        labels = item.get("labels", [])
        label_category = labels[0] if labels else "UNKNOWN"
        
        if start is not None and end is not None:
            entity_details.append((start, end, label_category, text))
            
    return entity_details

def main():
    print(f"Loading predictions from {PREDICTIONS_FILE}...")
    with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
        predictions_data = json.load(f)
        
    print(f"Loading golden standard from {GOLDEN_FILE}...")
    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    # 1. Map Golden Standard by relative path
    golden_details_map = {}
    for task in golden_data:
        # FIX: Check both the root and the "data" dictionary for rel_path
        rel_path = task.get("rel_path") or task.get("data", {}).get("rel_path")
        
        if rel_path:
            # FIX: Normalize Windows backslashes to Mac/Linux forward slashes
            rel_path = rel_path.replace("\\", "/")
            
            # Note: Checking "label" (Label Studio MIN-JSON) or "annotations" (Standard JSON)
            labels = task.get("label", [])
            entity_details = get_entity_set_golden(labels)
            golden_details_map[rel_path] = entity_details

    print("Merging annotations into predictions...")
    
    # 2. Iterate through predictions and merge
    for task in predictions_data:
        rel_path = task.get("data", {}).get("rel_path")
        
        if not rel_path:
            continue
            
        # FIX: Normalize the prediction paths just in case
        rel_path = rel_path.replace("\\", "/")
        
        if rel_path not in golden_details_map:
            print(f"Warning: No golden standard found for {rel_path}. Skipping.")
            continue
            
        true_details = golden_details_map[rel_path]
        
        # Format annotations correctly for the Label Studio UI
        annotations_result = []
        for s, e, l, text_snippet in true_details:
            annotations_result.append({
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": s,
                    "end": e,
                    "text": text_snippet,
                    "labels": [l]
                }
            })
            
        # Insert the golden standard as "annotations"
        task["annotations"] = [{"result": annotations_result}]

        # make sure annotations precedes predictions
        if "predictions" in task:
            task["predictions"] = task.pop("predictions")

    # 3. Save the new combined file
    print(f"Saving merged file to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions_data, f, ensure_ascii=False, indent=2)
    print("Save complete!\n")

if __name__ == "__main__":
    main()
