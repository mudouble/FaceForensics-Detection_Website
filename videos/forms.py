from django import forms
from .models import Videos_Post


# Compressed_format.objects.filter().delete()
# Forging_method.objects.filter().delete()
class UploadForm(forms.ModelForm):
    class Meta:
        model = Videos_Post
        fields = ["title", "videos"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def clean(self):
        super().clean()