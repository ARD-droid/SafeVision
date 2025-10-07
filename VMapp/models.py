from django.db import models

class Video(models.Model):
    title = models.CharField(max_length=255)
    video_file = models.FileField(upload_to='videos', null=True, blank=True)  # We will handle the naming manually

    def __str__(self):
        return self.title
