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
        
        # Determine mime type for crop
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
        elif ext == '.png':
            mime_type = "image/png"
        else:
            mime_type = "image/jpeg" # Default

        # Handle context image
        context_path = object_data.get('context_path')
        base64_context = None
        context_mime_type = "image/jpeg"
        
        if context_path and os.path.exists(context_path):
            base64_context = self.encode_image(context_path)
            ext_ctx = os.path.splitext(context_path)[1].lower()
            if ext_ctx in ['.jpg', '.jpeg']:
                context_mime_type = "image/jpeg"
            elif ext_ctx == '.png':
                context_mime_type = "image/png"

        prompt = """
        Analyze the object in the first image (the cropped view). 
        The second image provides the full scene context to help you identify the object and its scale.
        
        Predict its physical properties for use in a game engine and return a JSON object with the following keys:
        - name: A short name for the object.
        - material: The primary material (e.g., wood, metal, plastic).
        - mass_kg: Estimated mass in kilograms.
        - friction_coefficient: Estimated static friction coefficient (0.0 to 1.0).
        - elasticity: Estimated elasticity/restitution (0.0 to 1.0).
        - description: A brief description of the object.
        
        # Core Physics
        - motion_type: 'static' (e.g., walls, heavy furniture), 'dynamic' (movable props), or 'kinematic' (machinery/doors).
        - collision_primitive: Best fitting simple collider: 'box', 'sphere', 'capsule', 'cylinder', or 'convex_hull'.
        - center_of_mass: Approximate location: 'center', 'bottom' (stable), 'top' (unstable).

        # Gameplay / Interaction
        - destructibility: 'indestructible', 'breakable', or 'explosive'.
        - health: Estimated health points (1-100) if destructible, else null.
        - flammability: 0.0 (non-flammable) to 1.0 (highly flammable).
        - surface_sound: Audio material type: 'wood', 'metal', 'concrete', 'dirt', 'glass', 'plastic', 'fabric'.

        # Visual / Material (PBR)
        - roughness: 0.0 (smooth/mirror) to 1.0 (matte).
        - metallic: 0.0 (dielectric) to 1.0 (metal).

        # Dimensions
        - dimensions: {
            "length": Estimated length in meters,
            "width": Estimated width in meters,
            "height": Estimated height in meters
          }
        
        Output ONLY the JSON.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/SplatGraph/SplatGraph", # Required by OpenRouter
            "X-Title": "SplatGraph" # Optional
        }

        content_list = [
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
        
        if base64_context:
            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{context_mime_type};base64,{base64_context}"
                }
            })

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content_list
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
