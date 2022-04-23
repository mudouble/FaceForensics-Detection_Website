from django.db import models
# import os
# from django.contrib.auth.models import User


class Videos_Post(models.Model):
    videos = models.FileField(upload_to='in_out_videos/manipulate',
                              default="",
                              blank=True)  # 伪造视频存放地址
    detect_videos = models.FileField(upload_to='in_out_videos/result',
                                     default="",
                                     blank=True)
    title = models.CharField(max_length=200)
    forging_method = models.CharField(max_length=100, default='')
    compressed_format = models.CharField(max_length=100, default='')

    def __str__(self):
        return self.title

  
