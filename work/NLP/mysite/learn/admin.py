from django.contrib import admin
from .models import Grades,Students
# Register your models here.
class GradesAdmin(admin.ModelAdmin):
    #列表页属性
    list_display = ['pk','gname','gdate','ggirlnum','gboynum','isDelete']
    list_filter = ['gname']
    search_fields = ['gname']
    list_per_page = 5

    #添加，修改页属性
    fields = ['ggirlnum','gboynum','gname','gdate','isDelete']


admin.site.register(Grades,GradesAdmin)

class StudentsAdmin(admin.ModelAdmin):
    #列表页属性
    list_display = ['pk','sname','sage','sgender','scontend','sgrade','isDelete']
    list_per_page = 2
admin.site.register(Students,StudentsAdmin)
