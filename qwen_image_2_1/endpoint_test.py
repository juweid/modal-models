import os
import base64

from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO

load_dotenv()
    
KEY = os.environ.get("MODAL_API_TOKEN")
BASE_URL = os.environ.get("MODAL_API_URL") + "/v1"

openai_client = OpenAI(api_key=KEY, base_url=BASE_URL)

test_prompt = "A tabby kitten on a couch"

print("Generating image for prompt:", test_prompt)

openai_response = openai_client.images.generate(
    model="Qwen/Qwen-Image-2.1",
    prompt=test_prompt,
    n=1,
    size="512x512",
    response_format="b64_json"
)

print("Generation finished. Displaying and saving the generated image(s)...")

for idx, img_data in enumerate(openai_response.data):
    img_bytes = base64.b64decode(img_data.b64_json)
    imag = Image.open(BytesIO(img_bytes))
    imag.show()
    imag.save(f"generated_image_{idx + 1}.png")
 