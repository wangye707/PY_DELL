"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
admin.autodiscover()
from django.urls import path
from django.conf.urls import url,include

from learn import views as learn_views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',learn_views.index,name='index'),
    path('home/',learn_views.home,name='home'),
    path('add/',learn_views.add,name='add'),
    path('add2/<int:a>/<int:b>/', learn_views.add2),
    path('grades/',learn_views.grades,name='grades'),
    path('students/', learn_views.students, name='students'),
    path('grades/[\d]', learn_views.grade_students, name='grade_students'),

]
