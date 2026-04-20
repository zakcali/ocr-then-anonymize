import os
import asyncio
from openai import AsyncOpenAI
import httpx
import re
from tqdm import tqdm
import random

# --- Configuration ---
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output-gpt-oss-120b"

# API Configuration
VLLM_URL = os.environ.get("VLLM_URL")
if not VLLM_URL:
    raise RuntimeError("VLLM_URL environment variable is not set.")
VLLM_KEY = "EMPTY"
MODEL_NAME = "openai/gpt-oss-120b"

# Concurrency
CONCURRENCY_LIMIT = 50

# Generation Parameters
TEMPERATURE = 0.3
MAX_TOKENS = 31000
TOP_P = 0.95
EFFORT = "medium" # ["low", "medium", "high"]

def load_prompt(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().strip()

# Prompts
SYSTEM_PROMPT = load_prompt("anonym_system.txt")
PRE_PROMPT = load_prompt("anonym_pre.txt")

async def wait_for_gpu_buffer(vllm_url):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{vllm_url}/metrics")
            match = re.search(r'vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.]+)', response.text)
            if match:
                usage = float(match.group(1)) * 100
                if usage > 84.0:
                    return False  # no print here
                return True
            return True
        except:
            return True

def split_into_pages(text):
    """
    Splits OCR output into individual pages using <!-- Page N --> markers.
    - Strips the '# Document: ...' header line (may contain real name/filename).
    - Returns a list of (page_num, page_content) tuples.
    - If no page markers exist, treats the whole file as a single page.
    """
    # Remove the '# Document: ...' line at the top if present
    text = re.sub(r'^#\s*Document:.*\n?', '', text, flags=re.IGNORECASE).strip()

    # Split on <!-- Page N --> markers
    parts = re.split(r'(<!--\s*Page\s+(\d+)\s*-->)', text)

    pages = []

    if len(parts) == 1:
        # No markers found — treat entire file as page 1
        content = parts[0].strip()
        if content:
            pages.append((1, content))
        return pages

    # parts[0] is content before the first marker — discard (empty after header strip)
    i = 1
    while i < len(parts):
        page_num = int(parts[i + 1])
        content = parts[i + 2].strip() if i + 2 < len(parts) else ""
        if content:
            pages.append((page_num, content))
        i += 3

    return pages


async def anonymize_page(client, page_content, semaphore, pbar_total, relative_path, page_num, total_pages):
    """Sends a single page to the model for anonymization. Updates progress bar on completion."""
    
    # 1. Stagger the initial start to prevent a massive CPU spike
    await asyncio.sleep(random.uniform(0, 3)) 

    # 2. Enter the semaphore FIRST. This limits active metric-checking to 45 tasks.
    async with semaphore:
        
        # 3. Check GPU Buffer before sending the request
        was_waiting = False
        while not await wait_for_gpu_buffer(VLLM_URL.replace("/v1", "")):
            if not was_waiting:
                tqdm.write(f"⏳ KV Cache high - Waiting for buffer...")
            was_waiting = True
            await asyncio.sleep(5 + random.uniform(0, 2))
        if was_waiting:
            tqdm.write(f"✅ KV Cache recovered - Resuming")
            
        # Final tiny stagger to keep the vLLM scheduler happy
        await asyncio.sleep(random.uniform(0, 0.5)) 

        if page_num == 1:
            tqdm.write(f"▶️  Starting: {relative_path} ({total_pages} pages)")
            
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PRE_PROMPT + page_content}
        ]
        
        try:
            completion = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                reasoning_effort=EFFORT 
            )
            
            choice = completion.choices[0]
            if choice.finish_reason == "length":
                tqdm.write(f"⚠️ WARNING: {relative_path} (P{page_num}) CUT SHORT - will retry next run")
                return None  # ← triggers abort + no file written
            
            # Clean up potential markdown code blocks wrapped by the model
            answer = choice.message.content or ""
            answer = re.sub(r'^```markdown\n|```$', '', answer, flags=re.MULTILINE).strip()
            
            pbar_total.update(1)
            return answer

        except Exception as e:
            tqdm.write(f"❌ API Error on {relative_path} (P{page_num}): {e}")
            await asyncio.sleep(10) # Cooling off period
            return None # Fixed capitalization


async def process_single_file(client, file_path, relative_path, semaphore, pbar_total):
    filename_no_ext = os.path.splitext(relative_path)[0]
    output_path = os.path.join(OUTPUT_FOLDER, f"{filename_no_ext}.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Skip if already fully processed
    if os.path.exists(output_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        pages = split_into_pages(text)
        pbar_total.update(len(pages))
        return f"⏭️  Skipped (exists): {relative_path}"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        pages = split_into_pages(text)

        if not pages:
            return f"⚠️  Skipped (empty): {relative_path}"

        page_tasks = [
            anonymize_page(client, page_content, semaphore, pbar_total, relative_path, page_num, len(pages))
            for page_num, (_, page_content) in enumerate(pages, start=1)
        ]
        anonymized_pages = await asyncio.gather(*page_tasks)

        # Don't write if any page failed
        if any(page is None or page == "" for page in anonymized_pages):
            tqdm.write(f"❌ Skipping write for {relative_path} — one or more pages failed")
            return f"❌ Failed: {relative_path}"

        output_parts = []
        for (page_num, _), anonymized_content in zip(pages, anonymized_pages):
            output_parts.append(f"<!-- Page {page_num} -->\n{anonymized_content}")

        final_output = "\n\n".join(output_parts) + "\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_output)

        tqdm.write(f"✅ Done:     {relative_path} ({len(pages)} pages)")
        return f"✅ Done: {relative_path} ({len(pages)} pages)"

    except Exception as e:
        tqdm.write(f"❌ Error:    {relative_path}: {e}")
        return f"❌ Error: {relative_path}: {e}"


async def main():
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print("Created input folder.")
        return
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    all_files = []
    print(f"📂 Scanning '{INPUT_FOLDER}' recursively...")

    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.endswith(".txt") or file.endswith(".md"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, INPUT_FOLDER)
                all_files.append((full_path, rel_path))

    all_files.sort()

    if not all_files:
        print("No files found.")
        return

    # Pre-scan to count total pages for progress bar
    total_pages = 0
    file_info = []
    for full_path, rel_path in all_files:
        with open(full_path, "r", encoding="utf-8") as f:
            text = f.read()
        pages = split_into_pages(text)
        total_pages += len(pages)
        file_info.append((full_path, rel_path, len(pages)))

    print(f"🚀 Found {len(all_files)} files — {total_pages} pages total.")
    print(f"🔥 Starting Async Batch Processing ({CONCURRENCY_LIMIT} simultaneous requests)...")
    print("-" * 60)

    client = AsyncOpenAI(
        base_url=VLLM_URL,
        api_key=VLLM_KEY,
        timeout=httpx.Timeout(
            connect=30.0,
            read=7200.0,
            write=60.0,
            pool=30.0
        )
    )

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    with tqdm(total=total_pages, unit="page", desc="📊 Total") as pbar_total:
        tasks = [
            process_single_file(client, f_path, r_path, semaphore, pbar_total)
            for f_path, r_path, _ in file_info
        ]

        completed_count = 0
        for task in asyncio.as_completed(tasks):
            result = await task
            completed_count += 1

    print("\n🎉 All Done.")

if __name__ == "__main__":
    asyncio.run(main())
