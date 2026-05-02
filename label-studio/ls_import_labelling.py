import os
import re
import json

# --- CONFIGURATION ---
ORIGINAL_INPUT_FOLDER = "input"
MODEL_NAME = "Qwen3.6-27B-NonThinking"
MODEL_OUTPUT_FOLDER = "output-" + MODEL_NAME
OUTPUT_JSON = "label_studio_"+ MODEL_NAME +"_predictions.json"


# 18 HIPAA categories
VALID_TAGS = [
    "NAME", "LOCATION", "DATE", "PHONE", "FAX", "EMAIL", "SSN", 
    "MEDICALID", "HEALTHPLANID", "ACCOUNT", "LICENSE", "VEHICLE", 
    "DEVICE", "URL", "IP", "BIOMETRIC", "PHOTO", "OTHERID"
]

def clean_tag(tag_name):
    """Normalizes the tag name and ensures it falls into a valid category."""
    tag_name = tag_name.strip().upper()
    tag_name = re.sub(r'^\d+\.', '', tag_name) 
    
    if tag_name in VALID_TAGS:
        return tag_name
    return "OTHERID"

def extract_entities_from_model(model_text):
    """
    Extracts all tags and their content from the model's markdown output.
    Returns a list of tuples: [(tag, content), (tag, content)]
    """
    entities = []
    # Using re.DOTALL in case an entity spans across multiple lines
    pattern = r"<(.*?)>(.*?)</\1>"
    
    for match in re.finditer(pattern, model_text, flags=re.DOTALL):
        raw_tag = match.group(1)
        content = match.group(2)
        label = clean_tag(raw_tag)
        entities.append((label, content))
        
    return entities

def align_entities_to_original(original_text, entities):
    """
    Maps the extracted entities to their exact character coordinates 
    within the unmodified original text.
    """
    predictions = []
    search_start = 0
    
    for label, content in entities:
        # Strip trailing/leading spaces from the entity text to improve match success
        content_stripped = content.strip()
        if not content_stripped:
            continue
            
        # 1. First, search for the exact text AFTER our last found position.
        # This prevents picking the first "John" in the document if we are looking for the 3rd "John".
        pos = original_text.find(content_stripped, search_start)
        
        # 2. Fallback: If not found, the LLM might have swapped the order of some tags. 
        # Let's search from the beginning of the document just in case.
        if pos == -1:
            pos = original_text.find(content_stripped, 0)
            
        # 3. If we found a match in the original text, record its exact coordinates.
        if pos != -1:
            predictions.append({
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": pos,
                    "end": pos + len(content_stripped),
                    "text": content_stripped,
                    "labels": [label]
                }
            })
            # Move our search pointer forward so the next entity is searched after this one
            search_start = pos + len(content_stripped)
        else:
            # The LLM hallucinated text, fixed a typo, or changed a character.
            # We skip this entity because it doesn't exist in the original text.
            pass 
            
    return predictions

def main():
    if not os.path.exists(ORIGINAL_INPUT_FOLDER):
        print(f"❌ Error: Input folder '{ORIGINAL_INPUT_FOLDER}' not found.")
        return
        
    all_tasks = []
    print(f"📂 Starting two-folder alignment process...")
    
    # Scan the ORIGINAL input folder to get the Ground Truth texts
    for root, dirs, files in os.walk(ORIGINAL_INPUT_FOLDER):
        for file in sorted(files):
            if file.endswith(".txt") or file.endswith(".md"):
                original_full_path = os.path.join(root, file)
                rel_path = os.path.relpath(original_full_path, ORIGINAL_INPUT_FOLDER)
                
                # Determine the expected path of the model's output file
                # The generation script saves them as .md files
                filename_no_ext = os.path.splitext(rel_path)[0]
                model_output_path = os.path.join(MODEL_OUTPUT_FOLDER, f"{filename_no_ext}.md")
                
                # 1. Read the UNMODIFIED original text
                with open(original_full_path, "r", encoding="utf-8") as f:
                    original_text = f.read()
                
                predictions = []
                
                # 2. Check if the model actually processed this file
                if os.path.exists(model_output_path):
                    with open(model_output_path, "r", encoding="utf-8") as f:
                        model_text = f.read()
                        
                    # Extract tags from the model's output
                    entities = extract_entities_from_model(model_text)
                    
                    # Align those tags to the original text's coordinates
                    predictions = align_entities_to_original(original_text, entities)
                else:
                    print(f"⚠️ Warning: Model output missing for '{rel_path}'")
                
                # 3. Build the Label Studio Task with Predictions
                task = {
                    "data": {
                        "text": original_text, # Base text is ALWAYS the original
                        "file_name": file,
                        "rel_path": rel_path
                    }
                }
                
                # Only add predictions if we found some
                if predictions:
                    task["predictions"] = [{
                        "model_version": MODEL_NAME,
                        "result": predictions
                    }]
                    
                all_tasks.append(task)

    # Save the consolidated JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_tasks, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Success! {len(all_tasks)} tasks aligned and saved to '{OUTPUT_JSON}'.")
    print(f"💡 You can now safely import this into Label Studio to evaluate {MODEL_NAME}.")

if __name__ == "__main__":
    main()
