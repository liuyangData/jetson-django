from django.urls import path
from . import views 
urlpatterns = [
    path("stream/<rtsp>", views.stream, name="stream with rtsp"),
    path("stream", views.stream_webcam, name="stream with the webcam"),
    path("helmet_detect/<cam>", views.stream_cam, name="run a sample video with helment detect, cam: [1-12]"),
    path("helmet_detect", views.stream_helmet, name="run the default sample video with helment detect"),
]