import os
import base64
import io
import time
from multiprocessing import Pool, current_process
from pdf2image import convert_from_path
from PIL import Image
from openai import OpenAI
from tqdm import tqdm

# ================= CONFIGURATION =================
INPUT_FOLDER = "pdf-in"
OUTPUT_FOLDER = "md-out"

API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "QuantTrio/Qwen3-VL-32B-Instruct-AWQ"

# This will process 6 FILES simultaneously
CONCURRENCY =8

# DPI for PDF rendering
DPI = 200 

# Max pixel dimension for raw images (approx A4 height at 200 DPI)
# This prevents 4K/8K images from crashing the VLM.
MAX_IMAGE_DIMENSION = 2240 
# =================================================

def encode_image(image):
    """
    Converts a PIL Image to base64 string.
    Force converts to JPEG with specific quality to standardize input.
    """
    buffered = io.BytesIO()
    # Ensure image is in RGB mode (handle PNG/WebP transparency)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
        
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_client():
    """Create a unique client for each worker process"""
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL, timeout=600)

def ocr_single_image(client, image, page_num, doc_name):
    """Sends a single image to vLLM"""
    base64_image = encode_image(image)
        
    prompt_text_tr = (
        "Bu resimdeki Türkçe tıbbi belgeyi Markdown formatına çevir. "
        "Kurallar:\n"
        "1. Metni olduğu gibi, kelimesi kelimesine Türkçe olarak yaz.\n"
        "2. Tabloları mutlaka Markdown tablosu (| | |) olarak oluştur.\n"
        "3. Tıbbi terimleri ve sayıları hatasız aktar.\n"
        "4. Başlıkları kalın yap.\n"
        "5. Yorum yapma, sadece metni ver."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": prompt_text_tr},
                    ],
                }
            ],
            max_tokens=8192,
            temperature=0.01,
            top_p=0.1,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            stream=False,
            extra_body={"repetition_penalty": 1.05}
        )
        content = response.choices[0].message.content
        if content:
            content = content.replace("```markdown", "").replace("```", "").strip()
        return content

    except Exception as e:
        print(f"[{doc_name}] Page {page_num} Error: {e}")
        return f"\n<!-- Page {page_num} Error -->\n"

def process_file(file_path):
    """
    Worker function: Handles ONE file (PDF or Image) from start to finish.
    """
    try:
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()
        
        # 1. Setup Client for this worker
        client = get_client()

        # 2. Output Path
        rel_path = os.path.relpath(file_path, INPUT_FOLDER)
        output_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(rel_path)[0] + ".md")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        images = []

        # 3. Load File based on type
        if ext == '.pdf':
            # Convert PDF to Images
            images = convert_from_path(file_path, dpi=DPI)
        else:
            # Handle standard images (JPG, PNG, WEBP)
            with Image.open(file_path) as img:
                # Copy image to break file handle dependency
                img = img.copy()
                
                # Convert to RGB to remove alpha channels (transparency)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')

                # Resize if image is massive (prevents VLM crash)
                if max(img.size) > MAX_IMAGE_DIMENSION:
                    img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
                
                images = [img]
        
        full_markdown = ""

        # 4. Process Images
        for i, image in enumerate(images):

            # ================= TEMPORARY PATCH START =================
            # Saves the exact image being sent to AI as a JPEG in the output folder
            # print(f"--> STARTING: {file_name} | Page {i+1}", flush=True)
            # debug_path = os.path.splitext(output_path)[0] + f"_page_{i+1}.jpg"
            # image.save(debug_path, "JPEG", quality=85)
            # ================= TEMPORARY PATCH END ===================

            # Pass image to OCR (encode_image handles JPEG conversion internally)
            page_content = ocr_single_image(client, image, i + 1, file_name)
            full_markdown += f"<!-- Page {i+1} -->\n{page_content}\n\n"

        # 5. Save
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_markdown)
        
        return f"Done: {file_name} ({len(images)} pages/images)"

    except Exception as e:
        return f"Failed: {file_path} | {e}"

def main():
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)

    valid_extensions = ('.pdf', '.jpg', '.jpeg', '.png', '.webp')
    input_files = []
    
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.lower().endswith(valid_extensions):
                input_files.append(os.path.join(root, file))
    
    input_files.sort()
    print(f"Found {len(input_files)} documents (PDFs & Images).")
    print(f"Processing {CONCURRENCY} FILES simultaneously...")

    # Use imap to get a progress bar
    with Pool(processes=CONCURRENCY) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_file, input_files),
            total=len(input_files),
            desc="Total Progress"
        ))

    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()
