from django import forms

class AddForm(forms.Form):
    city=forms.CharField()

class party(forms.Form):
    keyword=forms.CharField()



