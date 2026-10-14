import modal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
import base64
import time

import torch
from diffusers import QwenImage21Pipeline

class ImageGenerationRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "512x512"
    response_format: Optional[str] = "b64_json"

class ImageResponseData(BaseModel):
    b64_json: str

class OpenAIImageResponse(BaseModel):
    created: int
    data: list[ImageResponseData]

def container_creation_finalizer():
    print("Container is being created. This runs once per container creation.")

app = modal.App("Image-API")

MODEL_NAME = "Qwen/Qwen-Image-2.1"

HF_CACHE_VOL = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

MODEL_PATH = f"{HF_CACHE_PATH}/{MODEL_NAME}"

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
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    timeout=600
)
@modal.concurrent(max_inputs=20)
class ModelAPI:
    
    model_id: str = modal.parameter(default=MODEL_NAME)
    
    @modal.enter(snap=True)
    def load_model_and_create_snapshot(self):
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

        # Warmup
        self.pipe("warmup", num_inference_steps=1, width=512, height=512)

        print(f"[+] Model successfully loaded on engine: {device} (VAE Tiling & Slicing active)")

    @modal.enter(snap=False)
    def restore_snapshot(self):
        print("Container restored directly from GPU snapshot!")

    @modal.asgi_app(requires_proxy_auth=True)
    def api(self):
        api = FastAPI()
        
        @api.get("/")
        def health_check():
            return {"status": "online", "model": self.model_id}

        @api.post("/v1/images/generations", response_model=OpenAIImageResponse)
        def generate_image(request: ImageGenerationRequest):
            if self.pipe is None:
                raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

            try:
                # Extract image dimensions & snap to multiples of 32 (required by 64-ch VAE)
                try:
                    width, height = map(int, request.size.split("x"))
                    width = max(32, (width // 32) * 32)
                    height = max(32, (height // 32) * 32)
                except ValueError:
                    width, height = 512, 512

                num_images = request.n if request.n and request.n > 0 else 1

                print(f"[Inference] Generating {num_images}x ({width}x{height}) image(s) for: '{request.prompt}'")

                # Serialize GPU access across FastAPI worker threads & disable grad tracking
                with torch.inference_mode():
                    # If Turbo model is used, let it use its native 8-step sigma schedule
                    is_turbo = "turbo" in self.model_id.lower()
                    pipe_kwargs = {
                        "prompt": [request.prompt] * num_images,
                        "width": width,
                        "height": height,
                        "use_kv_cache": True,
                    }
                    if not is_turbo:
                        pipe_kwargs["num_inference_steps"] = 20

                    output = self.pipe(**pipe_kwargs)

                response_data = []
                for generated_img in output.images:
                    buffered = BytesIO()
                    generated_img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    response_data.append({"b64_json": img_str})

                # Free temporary activation/KV-cache blocks before the next API request
                del output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return {
                    "created": int(time.time()),
                    "data": response_data,
                }
            except Exception as e:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise HTTPException(status_code=500, detail=str(e))

        return api
