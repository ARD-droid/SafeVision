from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_video, name='upload_video'),
    path('videos/', views.video_list, name='video_list'),
    path('frames/<int:video_id>/<str:folder>/', views.moderated_frames_view, name='extracted_frames'),
    path('transcribe/<int:video_id>/', views.transcribe_audio, name='transcribe_audio'),
    path('live/', views.live_options_view, name='live_options'),
]
