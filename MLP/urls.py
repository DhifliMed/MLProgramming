
from django.conf.urls import url,include
from . import  views
urlpatterns = [
    url(r'^$', views.Accueil,name='Accueil'),
    url(r'^App/', views.App,name='App'),
    url(r'^Doc/', views.Doc,name='Doc'),
    url(r'^apropos/', views.ABOUT,name='About'),
    url(r'^test/', views.test,name='TEST'),
]