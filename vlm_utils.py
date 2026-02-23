
import os
import base64
import json
import requests
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_building_crop(image_path, api_key=None):
    """
    Analyzes a building crop using OpenAI's GPT-4o to determine usage and story count.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    base_url = None
    
    if api_key and api_key.startswith("sk-or-"):
         base_url = "https://openrouter.ai/api/v1"
         print("Using OpenRouter API key.")

    if not api_key:
        print("Warning: No OPENROUTER_API_KEY found. Skipping VLM analysis.")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    base64_image = encode_image(image_path)
    
    prompt = """
    Analyze this image of a building (or object) from an aerial/drone view.
    Identify:
    1. Object Type: Building, Tree, Car, Usage, or Other.
    2. Approximate number of stories (visual count) - 0 if not a building.
    3. Likely usage (Residential, Commercial, Office, Industrial, Mixed-use, or Other) - Only for buildings.
    4. Estimated number of occupants (based on size/type) - 0 if not a building.
    5. A brief 1-sentence description.

    Respond in JSON format:
    {
        "object_type": "Building" | "Tree" | "Car" | "Other",
        "is_building": boolean,
        "stories_visual": int,
        "usage": "string",
        "estimated_occupants": int,
        "description": "string"
    }
    """


    model_name = "google/gemini-3-flash-preview" 

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
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
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=300
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
        
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return None

def predict_physics_properties(image_path, context_path=None, model="google/gemini-2.0-flash-001", api_key=None):
    """
    Predicts physics properties for an object crop using an LLM.
    Returns a dictionary with 'metadata' and 'usd_physics'.
    """
    if not api_key:
         api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
         
    base_url = None
    if os.environ.get("OPENROUTER_API_KEY"):
         base_url = "https://openrouter.ai/api/v1"
         
    if not api_key:
        print("Warning: No API key found. Skipping physics prediction.")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    base64_image = encode_image(image_path)
    
    # Determine mime type
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png" if ext == '.png' else "image/jpeg"

    messages = [
        {
            "role": "system",
            "content": "You are an expert 3D asset analyzer for physics simulation."
        }
    ]
    
    user_content = []
    
    prompt = """
    Analyze this object CROP.
    Identify the object and predict its physical properties for a physics engine (USD Physics).
    
    The CONTEXT image (if provided) is only for scale/orientation reference. Focus on the CROP.

    Return a JSON object with:
    1. "metadata": {
       "name": "string (snake_case)",
       "material": "string",
       "description": "string",
       "roughness": 0.0-1.0
    }
    2. "usd_physics": {
       "mass": { "mass": float (kg) },
       "rigid_body": { "rigidBodyEnabled": boolean },
       "collision": { "collisionEnabled": true, "collisionApproximation": "convexHull" },
       "deformable": { 
           "enabled": boolean, 
           "stiffness": float (100.0=soft, 100000.0=hard)
       }
    }
    
    CRITICAL: If the object is fabric, plush, soft, sponge, or rubber, set "deformable": { "enabled": true } and "rigid_body": { "rigidBodyEnabled": false }.
    """
    
    user_content.append({"type": "text", "text": prompt})
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
    })
    
    if context_path and os.path.exists(context_path):
        base64_context = encode_image(context_path)
        ext_ctx = os.path.splitext(context_path)[1].lower()
        ctx_mime = "image/png" if ext_ctx == '.png' else "image/jpeg"
        user_content.append({"type": "text", "text": "Context Image:"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{ctx_mime};base64,{base64_context}"}
        })

    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=500
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error in predict_physics_properties: {e}")
        return None
