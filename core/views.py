
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .GeneticAlg import GeneticAlg

# Create your views here.
def index(request):
 template = loader.get_template('index.html')
 return HttpResponse(template.render())

def add(request):
    print(request.POST)
    f=request.POST['function']
    cons=request.POST.getlist('cons')
    aab=[v.encode("ascii", "ignore").decode for v in cons]
    newkeys=[key.strip('\u202b\u202a') for key in cons]
    newkeys1=[key.strip('\u202c') for key in newkeys]
    print(newkeys1)
    limits=request.POST.getlist('Limits')
    aa=GeneticAlg(generationSize = 20, numGenerations = 2000, timeoutMin = 0.1,problem=f,Constarin=newkeys1,Limits=limits)

    
    print(aa.BestVars)
    return render(request, "result.html", {"result":aa.BestVars[1:]})
