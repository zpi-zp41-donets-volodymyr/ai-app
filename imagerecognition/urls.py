from django.urls import path
from . import views

app_name = 'imagerecognition'

urlpatterns = [
    path('', views.index, name='index'),
    path('recognize/', views.recognize, name='recognize'),
]
