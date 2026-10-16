import modal

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
from io import BytesIO
import base64
import time
import threading

import torch
from diffusers import QwenImage21Pipeline
from PIL import Image

MIN_DIMENSION = 32
MAX_DIMENSION = 2752
MAX_TOTAL_PIXELS = 2752 * 1536  # 4,227,072 pixels (accommodates 2048x2048 and 16:9 2752x1536)
MAX_NUM_IMAGES = 4

class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: Optional[int] = 1
    size: Optional[str] = "512x512"
    response_format: Optional[str] = "b64_json"
    user: Optional[str] = None

class ImageResponseData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None

class OpenAIImageResponse(BaseModel):
    created: int
    data: list[ImageResponseData]

def parse_and_validate_size(size_str: Optional[str]) -> tuple[int, int]:
    if not size_str:
        return 512, 512
    try:
        parts = size_str.lower().split("x")
        if len(parts) != 2:
            raise ValueError()
        width, height = int(parts[0]), int(parts[1])
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size format '{size_str}'. Expected 'WIDTHxHEIGHT' (e.g. '1024x1024')."
        )

    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions ({width}x{height}) are too small. Minimum is {MIN_DIMENSION}x{MIN_DIMENSION}."
        )
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image side length exceeds maximum allowed limit of {MAX_DIMENSION}px."
        )
    if (width * height) > MAX_TOTAL_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"Image resolution ({width}x{height} = {width * height:,} pixels) exceeds the maximum pixel count of Qwen-Image-2.1 ({MAX_TOTAL_PIXELS:,} pixels)."
        )

    # Snap to nearest multiple of 32 (required by 64-ch VAE)
    width = max(MIN_DIMENSION, round(width / 32) * 32)
    height = max(MIN_DIMENSION, round(height / 32) * 32)
    if (width * height) > MAX_TOTAL_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"Adjusted image resolution ({width}x{height} = {width * height:,} pixels) exceeds the maximum pixel count of Qwen-Image-2.1 ({MAX_TOTAL_PIXELS:,} pixels)."
        )
    return width, height

def validate_n(n: Optional[int]) -> int:
    if n is None:
        return 1
    if n < 1 or n > MAX_NUM_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Parameter 'n' must be between 1 and {MAX_NUM_IMAGES}."
        )
    return n

def validate_response_format(fmt: Optional[str]) -> str:
    fmt_normalized = (fmt or "b64_json").lower()
    if fmt_normalized not in ("b64_json", "url"):
        raise HTTPException(
            status_code=400,
            detail="Parameter 'response_format' must be either 'b64_json' or 'url'."
        )
    return fmt_normalized

def container_creation_finalizer():
    print("Container is being created. This runs once per container creation.")

app = modal.App("Qwen-Image-2.1-API")

MODEL_NAME = "Qwen/Qwen-Image-2.1"

HF_CACHE_VOL = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim()
    .uv_pip_install(
        "torch",
        "torchvision",
        "diffusers[torch]",
        "transformers",
        "ftfy",
        "fastapi",
        "pydantic",
        "huggingface_hub",
        "python-multipart",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": HF_CACHE_PATH})
    .run_function(container_creation_finalizer)
)


@app.cls(
    routing_region="eu-west",
    image=image,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    gpu="L40S",
    cpu=(0.125, 8.0),
    memory=(512, 48 * 1024),
    secrets=[modal.Secret.from_name("hf-token")],
    scaledown_window=60,
    max_containers=1,
    timeout=600
)
@modal.concurrent(max_inputs=20)
class ModelAPI:
    
    model_id: str = modal.parameter(default=MODEL_NAME)
    inference_lock: threading.Lock
    
    @modal.enter()
    def load_model(self):
        self.inference_lock = threading.Lock()
        
        print(f"[*] Loading model '{self.model_id}'...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe = QwenImage21Pipeline.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)

        HF_CACHE_VOL.commit()

        # Eliminate the +28.2 GB end-of-generation VAE spike at 2048x2048
        self.pipe.vae.enable_tiling(
            tile_sample_min_height=1024,
            tile_sample_min_width=1024,
            tile_sample_stride_height=768,
            tile_sample_stride_width=768,
        )
        # Decode batch items one-by-one if n > 1
        self.pipe.vae.enable_slicing()

        print(f"[+] Model successfully loaded on engine: {device} (VAE Tiling & Slicing active)")

    @modal.asgi_app(requires_proxy_auth=True)
    def api(self):
        api = FastAPI()
        
        @api.get("/")
        def health_check():
            return {"status": "online", "model": self.model_id}

        @api.post("/v1/images/generations", response_model=OpenAIImageResponse)
        def generate_image(request: ImageGenerationRequest):
            if getattr(self, "pipe", None) is None:
                raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

            try:
                clean_prompt = request.prompt.strip() if request.prompt else ""
                if not clean_prompt:
                    raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

                width, height = parse_and_validate_size(request.size)
                num_images = validate_n(request.n)
                resp_format = validate_response_format(request.response_format)

                print(f"[Inference] Generating {num_images}x ({width}x{height}) image(s) for: '{clean_prompt}'")

                is_turbo = "turbo" in self.model_id.lower()
                pipe_kwargs = {
                    "prompt": clean_prompt,
                    "num_images_per_prompt": num_images,
                    "width": width,
                    "height": height,
                    "use_kv_cache": True,
                }
                if not is_turbo:
                    pipe_kwargs["num_inference_steps"] = 40

                # Serialize GPU access across worker threads and disable gradient tracking
                with self.inference_lock:
                    with torch.inference_mode():
                        output = self.pipe(**pipe_kwargs)
                    generated_images = output.images
                    del output
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                response_data = []
                for generated_img in generated_images:
                    buffered = BytesIO()
                    generated_img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    if resp_format == "url":
                        response_data.append({"url": f"data:image/png;base64,{img_str}"})
                    else:
                        response_data.append({"b64_json": img_str})

                return {
                    "created": int(time.time()),
                    "data": response_data,
                }
            except HTTPException:
                raise
            except Exception as e:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise HTTPException(status_code=500, detail=str(e))

        @api.post("/v1/images/edits", response_model=OpenAIImageResponse)
        def edit_image(
            prompt: str = Form(...),
            image: Optional[List[UploadFile]] = File(None),
            image_brackets: Optional[List[UploadFile]] = File(None, alias="image[]"),
            mask: Optional[UploadFile] = File(None),
            model: Optional[str] = Form(None),
            n: Optional[int] = Form(1),
            size: Optional[str] = Form("512x512"),
            response_format: Optional[str] = Form("b64_json"),
            user: Optional[str] = Form(None),
        ):
            if getattr(self, "pipe", None) is None:
                raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

            try:
                clean_prompt = prompt.strip() if prompt else ""
                if not clean_prompt:
                    raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

                uploaded_images = image or image_brackets
                if not uploaded_images:
                    raise HTTPException(status_code=422, detail="No image provided in 'image' or 'image[]'")

                width, height = parse_and_validate_size(size)
                num_images = validate_n(n)
                resp_format = validate_response_format(response_format)

                # Validate and load uploaded image(s) safely
                input_images = []
                for img_upload in uploaded_images:
                    try:
                        img_upload.file.seek(0)
                        pil_img = Image.open(img_upload.file)
                        pil_img.load()
                        input_images.append(pil_img.convert("RGB"))
                    except Exception as e:
                        filename = getattr(img_upload, "filename", "uploaded file")
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid or corrupted image file '{filename}': {str(e)}"
                        )
                    finally:
                        try:
                            img_upload.file.close()
                        except Exception:
                            pass

                print(f"[Inference] Generating {num_images}x ({width}x{height}) image(s) for: '{clean_prompt}'")

                is_turbo = "turbo" in self.model_id.lower()
                pipe_kwargs = {
                    "prompt": clean_prompt,
                    "image": input_images,
                    "num_images_per_prompt": num_images,
                    "width": width,
                    "height": height,
                    "use_kv_cache": True,
                }
                if not is_turbo:
                    pipe_kwargs["num_inference_steps"] = 40

                # Serialize GPU access across worker threads and disable gradient tracking
                with self.inference_lock:
                    with torch.inference_mode():
                        output = self.pipe(**pipe_kwargs)
                    generated_images = output.images
                    del output
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                response_data = []
                for generated_img in generated_images:
                    buffered = BytesIO()
                    generated_img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    if resp_format == "url":
                        response_data.append({"url": f"data:image/png;base64,{img_str}"})
                    else:
                        response_data.append({"b64_json": img_str})

                return {
                    "created": int(time.time()),
                    "data": response_data,
                }
            except HTTPException:
                raise
            except Exception as e:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise HTTPException(status_code=500, detail=str(e))

        return api
