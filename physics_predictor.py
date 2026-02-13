import os
import json
from vlm_utils import predict_physics_properties

class PhysicsPredictor:
    def __init__(self, api_key, model="google/gemini-2.0-flash-001"):
        self.api_key = api_key
        self.model = model

    def predict(self, object_data):
        image_path = object_data.get('best_crop_path')
        if not image_path:
            print(f"No crop path for object {object_data.get('id')}")
            return None
            
        context_path = object_data.get('context_path')
        
        return predict_physics_properties(
            image_path=image_path,
            context_path=context_path,
            model=self.model,
            api_key=self.api_key
        )

