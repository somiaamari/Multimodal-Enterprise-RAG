import base64
import os
from io import BytesIO
from typing import Optional

import requests
from PIL import Image

from app.utils.config import Settings


class ImageCaptioner:
    """
    Uses Hugging Face Inference API to generate captions for images.
    Free tier: 1,000 requests per day.
    """

    def __init__(self):
        # Use BLIP model for image captioning (free, no GPU needed)
        self.api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
        self.headers = {}

        # Try to use HF token if available
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            self.headers["Authorization"] = f"Bearer {hf_token}"
        else:
            print("⚠️  No HF_TOKEN found. Rate limits will be strict.")

    def caption_image(self, image_path: str) -> str:
        """Generate a caption for an image file."""
        try:
            # Open and prepare image
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Call Hugging Face API
            response = requests.post(
                self.api_url, headers=self.headers, data=image_data
            )

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    caption = result[0].get(
                        "generated_text", "A chart from the document."
                    )
                    return caption
                else:
                    return "A chart from the document."
            else:
                print(
                    f"⚠️  API Error (status {response.status_code}): {response.text[:200]}"
                )
                return "A chart from the document."

        except Exception as e:
            print(f"❌ Failed to caption image {image_path}: {e}")
            return "A chart from the document."

    def caption_images_batch(self, image_paths: list) -> dict:
        """Caption multiple images and return a dict mapping path -> caption."""
        results = {}
        for path in image_paths:
            print(f"   🖼️  Captioning: {os.path.basename(path)}")
            caption = self.caption_image(path)
            results[path] = caption
        return results
