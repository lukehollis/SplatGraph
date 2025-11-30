import os
import json
import requests
import base64

class PhysicsPredictor:
    def __init__(self, api_key, model="x-ai/grok-4.1-fast:free"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def predict(self, object_data):
        image_path = object_data['best_crop_path']
        base64_image = self.encode_image(image_path)
        
        # Determine mime type
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
        elif ext == '.png':
            mime_type = "image/png"
        else:
            mime_type = "image/jpeg" # Default

        prompt = """
        Analyze the object in this image and predict its physical properties.
        Return a JSON object with the following keys:
        - name: A short name for the object.
        - material: The primary material (e.g., wood, metal, plastic).
        - mass_kg: Estimated mass in kilograms.
        - friction_coefficient: Estimated static friction coefficient (0.0 to 1.0).
        - elasticity: Estimated elasticity/restitution (0.0 to 1.0).
        - description: A brief description of the object.
        
        Output ONLY the JSON.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/SplatGraph/SplatGraph", # Required by OpenRouter
            "X-Title": "SplatGraph" # Optional
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Clean up markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
                
            return json.loads(content)
        except Exception as e:
            print(f"Error predicting physics for object {object_data['id']}: {e}")
            if 'response' in locals():
                print(f"Response content: {response.text}")
            return None
