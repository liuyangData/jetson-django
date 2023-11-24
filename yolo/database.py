from django.db import models

# Create your models here.
class Alerts(models.Model):
	datetime = models.DateTimeField()
	camera = models.CharField(max_length=200)
	violation = models.CharField(max_length=200)
	

