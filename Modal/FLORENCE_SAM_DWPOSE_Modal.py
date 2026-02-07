import os
import io
import modal

# --- Constants ---
CHECKPOINT_DIR = "/root/checkpoints"
DWPOSE_DIR = f"{CHECKPOINT_DIR}/dwpose"

# --- 1. Define the Download Function ---
def download_models():
    from huggingface_hub import snapshot_download
    import os
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DWPOSE_DIR, exist_ok=True)
    
    print("⬇️ Downloading Florence-2...")
    snapshot_download(repo_id="microsoft/Florence-2-large", local_dir=f"{CHECKPOINT_DIR}/florence2")

    print("⬇️ Downloading SAM 2 Checkpoint...")
    # Using curl/wget to fetch the checkpoint
    os.system(f"wget -P {CHECKPOINT_DIR} https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt")

    print("⬇️ Downloading DWPose Models (ONNX)...")
    # Manually download ONNX files to avoid runtime HF connection issues
    os.system(f"wget -q -O {DWPOSE_DIR}/yolox_l.onnx https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx")
    os.system(f"wget -q -O {DWPOSE_DIR}/dw-ll_ucoco_384.onnx https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx")

# --- 2. Define the Container Image ---
image = (
    modal.Image.debian_slim()
    .apt_install("git", "wget", "libgl1", "libglib2.0-0") # libgl1 is crucial for opencv
    .pip_install(
        "torch", "torchvision",
        "transformers==4.46.3", 
        "accelerate",
        "opencv-python", 
        "pillow", 
        "huggingface_hub",
        "timm", 
        "einops", 
        "numpy",
        # DWPose specific stack
        "dwpose",            # The standalone package (not controlnet_aux)
        "onnxruntime-gpu",   # Required for DWPose inference
        "mediapipe",         # Required by dwpose package internally
        "protobuf",          # Often needed to prevent mediapipe import errors
        "scipy",
        "scikit-image",
        "matplotlib",
    )
    .pip_install("git+https://github.com/facebookresearch/sam2.git")
    .run_function(
        download_models,
        secrets=[modal.Secret.from_name("huggingface-secret")]
    )
)

app = modal.App("maskedbar-pipeline-full", image=image)

# --- 3. The Main Inference Class ---
@app.cls(
    gpu="A10G", 
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
    concurrency_limit=1
)
class ImagePipeline:
    @modal.enter()
    def load_models(self):
        """Loads models into GPU memory when the container starts."""
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # Specific Imports for DWPose
        import huggingface_hub
        from dwpose import DwposeDetector

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading models on {self.device}...")

        # --- A. PATCH HF DOWNLOAD FOR DWPOSE ---
        # We patch the hub download function to intercept requests for the ONNX files
        # and serve them from our local pre-downloaded path.
        global original_hf_download
        original_hf_download = huggingface_hub.hf_hub_download

        def patched_hf_download(repo_id, filename, **kwargs):
            if "yolox_l" in filename:
                print(f"🛡️ Patch: Loading Local YOLOX -> {DWPOSE_DIR}/yolox_l.onnx")
                return f"{DWPOSE_DIR}/yolox_l.onnx"
            if "dw-ll_ucoco" in filename:
                print(f"🛡️ Patch: Loading Local DWPose -> {DWPOSE_DIR}/dw-ll_ucoco_384.onnx")
                return f"{DWPOSE_DIR}/dw-ll_ucoco_384.onnx"
            return original_hf_download(repo_id, filename, **kwargs)

        huggingface_hub.hf_hub_download = patched_hf_download
        print("🛡️ DWPose Monkey-Patch applied.")

        # --- B. Load Florence-2 ---
        florence_path = f"{CHECKPOINT_DIR}/florence2"
        self.florence_processor = AutoProcessor.from_pretrained(florence_path, trust_remote_code=True)
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            florence_path, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device).eval()

        # --- C. Load SAM 2 ---
        sam2_ckpt = f"{CHECKPOINT_DIR}/sam2.1_hiera_large.pt"
        sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" 
        self.sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # --- D. Load DWPose ---
        # The patch ensures this loads from local files without internet check issues
        self.dwpose_model = DwposeDetector.from_pretrained_default()
        # dwpose usually handles .to(device) internally via onnxruntime providers, 
        # but if the class supports it:
        if hasattr(self.dwpose_model, 'to'):
             self.dwpose_model.to(self.device)

        print("✅ All Models Loaded Successfully.")

    @modal.method()
    def process_image(self, image_bytes, text_prompt):
        import torch
        import numpy as np
        import cv2
        from PIL import Image

        # Load Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size
        
        # --- STEP 1: Florence (Detection) ---
        print(f"   Step 1: Detection for '{text_prompt}'...")
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        prompt = task_prompt + text_prompt
        
        inputs = self.florence_processor(text=prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
        
        with torch.inference_mode():
            generated_ids = self.florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
        
        generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.florence_processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(w, h)
        )
        
        bboxes = parsed_answer.get(task_prompt, {}).get('bboxes', [])
        
        if not bboxes:
            print("   ⚠️ No object detected. Returning original.")
            byte_arr = io.BytesIO()
            image.save(byte_arr, format='PNG')
            return byte_arr.getvalue()
            
        box_prompt = np.array(bboxes[0])
        print(f"   🎯 Detected Box: {box_prompt}")

        # --- STEP 2: SAM 2 (Segmentation) ---
        print(f"   Step 2: Segmentation...")
        self.sam2_predictor.set_image(np.array(image))
        masks, _, _ = self.sam2_predictor.predict(
            box=box_prompt[None, :],
            multimask_output=False
        )
        final_mask = masks[0]
        
        # Mask Processing
        binary_mask_img = (final_mask > 0).astype(np.uint8) * 255
        edges = cv2.Canny(binary_mask_img, 100, 200)
        kernel = np.ones((2, 2), np.uint8)
        thick_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # --- STEP 3: DWPose (Skeleton) ---
        print(f"   Step 3: Generating DWPose Skeleton...")
        
        # FIX: We must match the Colab arguments to get the image output.
        # image_and_json=True returns a tuple: (out_img, keypoints, source)
        out_img, keypoints, source = self.dwpose_model(
            image,
            include_hand=True,
            include_face=False,
            include_body=True,
            image_and_json=True,  # Crucial for returning PIL image
            detect_resolution=512
        )
        
        # Ensure we have a PIL image
        if not isinstance(out_img, Image.Image):
             # Fallback if library changes behavior, though unlikely with fixed install
             out_img = Image.fromarray(out_img)

        # Resize skeleton to match original image exactly
        skeleton_image = out_img.resize((w, h), Image.LANCZOS)
        
        # --- STEP 4: Final Overlay ---
        print(f"   Step 4: Composing Result...")
        
        base_np = np.array(skeleton_image.convert("RGB"))
        
        # Define where edges are (from SAM)
        edge_mask = thick_edges > 100
        
        # Paint edges White onto the Skeleton background
        base_np[edge_mask] = [255, 255, 255]
        
        final_result = Image.fromarray(base_np)

        # Convert back to bytes
        byte_arr = io.BytesIO()
        final_result.save(byte_arr, format='PNG')
        return byte_arr.getvalue()

# --- 4. Local Entrypoint ---
@app.local_entrypoint()
def main(filepath: str = "bench.jpeg", prompt: str = "bench"):
    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' not found.")
        return

    print(f"🚀 Reading '{filepath}' and sending to Modal...")
    with open(filepath, "rb") as f:
        image_bytes = f.read()

    pipeline = ImagePipeline()
    try:
        result_bytes = pipeline.process_image.remote(image_bytes, prompt)
    except Exception as e:
        print(f"❌ Remote execution failed: {e}")
        return

    output_path = "modal_output_dwpose.png"
    with open(output_path, "wb") as f:
        f.write(result_bytes)
    
    print(f"✅ Success! Result saved to: {output_path}")