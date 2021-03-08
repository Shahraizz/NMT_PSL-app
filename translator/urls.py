
from django.urls import path, include

from .views import translator_page_view
from .views import translate_query

urlpatterns = [
    path('', translator_page_view, name='home'),
    path('translate', translate_query, name='translator'),
]
