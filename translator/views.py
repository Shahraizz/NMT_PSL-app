from django.shortcuts import render
from django.http import HttpResponse

from django.http import JsonResponse

from .nmt_model.Transformer import Model


MODEL = Model()

# Create your views here.
def translator_page_view(request, *args, **kwargs):
    return render(request, "translator.html", {})

def translate_query(request, *args, **kwargs):
    translation = {
        "result": MODEL.translate(request.GET.get("query"))
    }
    return JsonResponse(translation, status=200)