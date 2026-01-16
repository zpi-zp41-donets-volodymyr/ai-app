import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
import json
from .models import ImageRecognizer


def index(request):
    """Display image upload form and results"""
    return render(request, 'imagerecognition/index.html')


@csrf_exempt
@require_http_methods(["POST"])
def recognize(request):
    """Handle image upload and recognition"""
    try:
        if 'image' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No image file provided'
            }, status=400)
        
        image_file = request.FILES['image']
        
        # Validate file type
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
        file_extension = image_file.name.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            return JsonResponse({
                'success': False,
                'error': f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'
            }, status=400)
        
        # Save temporary file
        temp_path = default_storage.save(f'temp/{image_file.name}', image_file)
        full_path = os.path.join(settings.MEDIA_ROOT, temp_path)
        
        try:
            # Recognize image
            results = ImageRecognizer.recognize_image(full_path, top_k=5)
            
            return JsonResponse({
                'success': True,
                'results': results,
                'filename': image_file.name
            })
        
        except Exception as recognition_error:
            return JsonResponse({
                'success': False,
                'error': f'Image recognition failed: {str(recognition_error)}. Please ensure you have internet connection for the first download.'
            }, status=500)
        
        finally:
            # Clean up temporary file
            try:
                if default_storage.exists(temp_path):
                    default_storage.delete(temp_path)
            except:
                pass
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
