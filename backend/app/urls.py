from django.urls import path
from .views import DataAnalysisAPIView, SaveResultsAPIView

urlpatterns = [
    path('api/analysis/', DataAnalysisAPIView.as_view(), name='data_analysis'),
    path('api/save-results/', SaveResultsAPIView.as_view(), name='save_results'),
]