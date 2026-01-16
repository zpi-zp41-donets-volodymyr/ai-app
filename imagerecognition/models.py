import os
import ssl
import numpy as np
from PIL import Image
from django.conf import settings
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions


# Set up SSL context for downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Set Keras cache directory
os.environ['KERAS_HOME'] = os.path.join(settings.BASE_DIR, 'keras_cache')


class ImageRecognizer:
    """Handles image recognition using MobileNetV2 pre-trained model"""
    
    _model = None
    
    @classmethod
    def get_model(cls):
        """Load or return cached model"""
        if cls._model is None:
            try:
                cls._model = MobileNetV2(weights='imagenet')
            except Exception as e:
                # Fallback: try loading without pre-trained weights first
                cls._model = MobileNetV2(weights=None)
                raise Exception(f"Warning: Model loaded without pre-trained weights. Error: {str(e)}")
        return cls._model
    
    @classmethod
    def recognize_image(cls, image_path, top_k=5):
        """
        Recognize objects in an image
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples (label, confidence)
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Get predictions
            model = cls.get_model()
            predictions = model.predict(img_array, verbose=0)
            decoded = decode_predictions(predictions, top=top_k)
            
            # Format results
            results = []
            for _, label, confidence in decoded[0]:
                results.append({
                    'label': label,
                    'confidence': float(confidence) * 100
                })
            
            return results
        
        except Exception as e:
            raise Exception(f"Error recognizing image: {str(e)}")
