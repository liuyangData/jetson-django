from django.db import models

class Logs(models.Model):
	datetime = models.DateTimeField("date published")
	camera = models.CharField(max_length=200)
	content = models.CharField(max_length=200)
