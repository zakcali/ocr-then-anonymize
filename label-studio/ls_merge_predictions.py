import json
import argparse
import sys
import os

def merge_label_studio_predictions(output_file, input_files):
    if len(input_files) < 2:
        print("Need at least 2 files to merge.")
        return

    merged_tasks = {}

    # We process input files in reverse order. 
    # Label Studio defaults to the last prediction in the list.
    # By reversing, the FIRST file passed in the command line 
    # gets appended LAST, making it the default prediction.
    for file_path in reversed(input_files):
        print(f"Loading {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for task in data:
                # Use a reliable unique key, like rel_path or file_name in data
                task_data = task.get("data", {})
                
                # Try to find a unique identifier for the task
                task_key = task_data.get("rel_path") or task_data.get("file_name") or task_data.get("text")
                
                # Fallback to stringifying the entire data dictionary if no obvious unique string is found
                if not task_key:
                    task_key = json.dumps(task_data, sort_keys=True)

                if task_key not in merged_tasks:
                    # Create a new entry
                    merged_tasks[task_key] = {
                        "data": task_data,
                        "predictions": []
                    }
                    
                # Append new predictions to the existing task
                if "predictions" in task:
                    merged_tasks[task_key]["predictions"].extend(task["predictions"])
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    final_output = list(merged_tasks.values())
    
    print(f"Writing {len(final_output)} merged tasks to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple Label Studio JSON prediction files into one.")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file path (e.g., merged.json)")
    parser.add_argument("inputs", nargs="+", help="Input JSON file paths to merge")
    
    args = parser.parse_args()
    merge_label_studio_predictions(args.output, args.inputs)
