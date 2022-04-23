
from django.urls import path
from .views import (
    UploadVideosView,
    UserVideosView,
    )
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index),
    path("upload/", UploadVideosView.as_view(), name="upload_videos"),
    path(
        "user/<username>/",
        UserVideosView.as_view(),
        name="user_videoshow",
    ),
    path("process/detail/<pk>", views.VideosInformationView, name="details"),
    path("process/detect/<pk>", views.text, name="detect_models"),
    # path("process/detects", views.funs, name="detects"),
    path("process/<pk>/<m>", views.funs, name="ds"),
    path("process/models_information/", views.ModelsDetailView, name="modelsinfo"),
    path("reminder2/<num>", views.reminder2, name="reminder2"),
    path("reminder1/<num>", views.reminder1, name="reminder1"),

    path("result", views.threshold, name='result')
    
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)