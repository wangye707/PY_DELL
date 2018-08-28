#coding:utf-8
from django.shortcuts import render
from django.http import HttpResponse

from .forms import AddForm
from  .forms import party
# Create your views here.
from requests import get
from scrapy import Selector
from work.NLP.TF_IDF import *



def get_se(url,xpath,kv,flag):
    headers = {
        "User-Agent": "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36"
    }

    print("开始获取")
    if flag==0:
        r=get(url,params=kv)
    else:
        r=get(url)
    r.encoding = r.apparent_encoding
    html=r.text
    se=Selector(text=html).xpath(xpath).extract()
    return se

def weather(a):
    bsae_url = "http://www.baidu.com/s"
    kv = {'wd': a+"天气"}
    xpath='//div[@id="1"]/h3/a/@href'
    links=get_se(bsae_url,xpath,kv,0)
    content=get_se(links[0],'//input[ @ id = "hidden_title"]/@value',kv,1)
    print(content)
    return content[0]



def index(request):
    if request.method=='POST':
        form=AddForm(request.POST)
        if form.is_valid():
            a=form.cleaned_data['city']
            #b=form.cleaned_data['b']
            if len(a)!=0:
                res=weather(a)
            print("获取成功")
            return  HttpResponse("今天是"+res[:6]+"，"+res[10:12]+"，"+a+res[14:])
    else:
        form=AddForm()
    return render(request,'index.html',{'form':form})

# def weather(request):
#     if request.method == 'POST':

def index2(request):
    if request.method=='POST':
        keyword=party(request.POST)
        if keyword.is_valid():
            a=keyword.cleaned_data['keyword']
            #b=form.cleaned_data['b']
            if len(a)!=0:
                #调用if函数
                out_html=TF_IDF(a)
            print("获取成功")
            return  HttpResponse(out_html)
    else:
        keyword=party()
    return render(request,'index.html',{'keyword':keyword})

def add(request):
    a=request.GET['a']
    b=request.GET['b']
    c=int(a)+int(b)
    return HttpResponse(str(c))

def add2(request,a,b):
    c=int(a)+int(b)
    return HttpResponse(str(c))

def home(request):
    List=map(str,range(100))
    return render(request,'home.html',{'List':List})

from .models import Grades,Students
def grades(request):
    gradesList=Grades.objects.all()
    #讲数据传递给模板,模板再渲染页面
    return render(request,"grades.html",{"grades":gradesList})

def students(request):
    studentsList = Students.objects.all()
    # 讲数据传递给模板,模板再渲染页面
    return render(request, "students.html", {"students": studentsList})

def grade_students(request,num):
    grade=Grades.objects.get(pk=num)
    studentsList=grade.students_set.all()
    return render(request,'students.html',{"students":studentsList})