# from django.db import models

# class UploadedData(models.Model):
#     file_name = models.CharField(max_length=255)
#     table_name = models.CharField(max_length=255)
#     upload_date = models.DateTimeField(auto_now_add=True)
    
#     class Meta:
#         db_table = 'uploaded_data'

# class QueryHistory(models.Model):
#     question = models.TextField()
#     sql_query = models.TextField()
#     results = models.JSONField()
#     execution_time = models.FloatField()
#     created_at = models.DateTimeField(auto_now_add=True)
    
#     class Meta:
#         db_table = 'query_history'