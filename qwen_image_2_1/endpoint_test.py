import os
import base64

from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO

load_dotenv()
    
KEY = os.environ.get("MODAL_API_TOKEN")
BASE_URL = os.environ.get("MODAL_QWEN21_API_URL") + "/v1"

openai_client = OpenAI(api_key=KEY, base_url=BASE_URL)

test_prompt = "A wide-angle photograph of a lone American bison in a misty geothermal landscape, half hidden by white steam, a slate-blue stream and evergreens fading into the mist behind. Muted, cool palette, soft overcast light."

print("Generating image for prompt:", test_prompt)

openai_response = openai_client.images.generate(
    model="Qwen/Qwen-Image-2.1",
    prompt=test_prompt,
    n=1,
    size="1024x1024",
    response_format="b64_json"
)

print("Generation finished. Displaying and saving the generated image(s)...")

for idx, img_data in enumerate(openai_response.data):
    img_bytes = base64.b64decode(img_data.b64_json)
    imag = Image.open(BytesIO(img_bytes))
    imag.save(f"generated_image_{idx + 1}.png")
 
openai_response = openai_client.images.edit(
    model="Qwen/Qwen-Image-2.1",
    prompt="The same lone American bison in the same misty geothermal landscape, now at night under a full moon: the steam lit from behind, cool blue light, deep shadows, stars above the evergreens.",
    image=open("generated_image_1.png", "rb"),
    n=1,
    size="1024x1024",
    response_format="b64_json"
)

for idx, img_data in enumerate(openai_response.data):
    img_bytes = base64.b64decode(img_data.b64_json)
    imag = Image.open(BytesIO(img_bytes))
    imag.show()
    imag.save(f"edited_image_{idx + 1}.png")