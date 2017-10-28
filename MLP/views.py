from django.shortcuts import render
from django.http import HttpResponse
from MLP import mylib as ml
from MLP import comparaison as cm

def Accueil(request):
    return (HttpResponse(render(request,'accueil.html')))

def App(request):
    if(request.method=='POST'):
        t=str(request.POST['tweet'])
        if(len(t)<3):
            pol = 'veuillez entrer une phrase valide'
            return (HttpResponse(render(request, 'App.html', {'pol': pol})))

        alg=int(request.POST['algorithme'])

        if(str(request.POST.get("cb"))=='on'):
            if(str(request.POST.get("st"))=='on'):
                s=1
            else:
                s=0
            if(alg == 0):
                if(str(request.POST.get("k")).isnumeric()):
                    k = int(request.POST.get("k"))
                else:
                    k=35
                pol=ml.predictknnp(t,k,s)
                return (HttpResponse (render(request, 'App.html',{'pol':pol} )))
            elif (alg == 1):
                if (str(request.POST.get("nb")).isnumeric()):
                    nb = int(request.POST.get("nb"))
                else:
                    nb = 0
                pol=ml.predictnbp(t, nb,s)
                return (HttpResponse (render(request, 'App.html', {'pol': pol})))
            else:
                if (str(request.POST.get("ker")).isnumeric()):
                    ker = int(request.POST.get("ker"))
                else:
                    ker = 0
                pol=ml.predictsvmp(t,ker,s)
                return (HttpResponse(render(request, 'App.html', {'pol': pol})))
        else:
            if (alg == 0):
                pol = ml.predictknnp(t,35,0)
                return (HttpResponse(render(request, 'App.html', {'pol': pol})))

            elif (alg == 1):
                pol = ml.predictnbp(t,0,0)
                return (HttpResponse(render(request, 'App.html', {'pol': pol})))
            else:
                pol = ml.predictsvmp(t,0,0)
                return (HttpResponse(render(request, 'App.html', {'pol': pol})))

    else:
        return (HttpResponse(render(request,'App.html')))

def Doc(request):
    return (HttpResponse(render(request,'documentation.html')))

def ABOUT(request):
    return (HttpResponse(render(request,'apropos.html')))

def test(request):
    return cm.Impalg()