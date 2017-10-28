from django import forms

class tform(forms.Form):
    tweet = forms.CharField(label='tweet', max_length=500)