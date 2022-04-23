# from django import urls
from django.conf import settings
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from django.conf.urls.static import static
from django.conf.urls import url

urlpatterns = [
	path('', RedirectView.as_view(url='/videos'), name='home'),
	path('accounts/', include('signup.urls')),
	path('accounts/', include('django.contrib.auth.urls')),
	path('admin/', admin.site.urls),
	path('videos/', include('videos.urls')),
]
urlpatterns += [url(r'^silk/', include('silk.urls', namespace='silk'))]

if settings.DEBUG:
	urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
