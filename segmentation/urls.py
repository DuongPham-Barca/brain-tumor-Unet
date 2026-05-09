from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('generate_report/', views.generate_report, name='generate_report'),
    path('download_report/<str:image_id>/', views.download_report, name='download_report'),
]