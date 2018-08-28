# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 13:23:41 2018

@author: P
"""


import urllib.request as urllib2
import threading
import requests
import time
import os
import datetime
import re

pa=os.getcwd()
big_path=pa+'\\国产小视频\\'

class Mythread(threading.Thread):
    def __init__(self,url,startpos,endpos,f):
        super(Mythread,self).__init__()
        self.url=url
        self.startpos=startpos
        self.endpos=endpos
        self.fd=f
    
    def download(self):
        print('start thread:%s at %s'%(self.getName(),time.time()))
        headers={}
        #headers={'Range':'bytes=%s-%s'%(self.startpos,self.endpos)}
        headers['User-Agent']='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
        headers['Range']='bytes=%s-%s'%(self.startpos,self.endpos)
        res=requests.get(self.url,headers=headers)
        self.fd.seek(self.startpos)
        self.fd.write(res.content)
        print('Stop thread:%s at%s'%(self.getName(),time.time()))
        self.fd.close()
    
    def run(self):
        self.download()

def thread_down(down_url,big_path,filename):
    url=down_url
    filesize=int(requests.head(url).headers['Content-Length'])
    print('%s filesize:%s'%(filename,filesize))

    time1=datetime.datetime.now()
    threadnum=3
    threading.BoundedSemaphore(threadnum)#允许线程个数
    step=filesize//threadnum
    mtd_list=[]
    start=0
    end=-1
    
    tempf = open(big_path+filename,'w')
    tempf.close()
    with open(big_path+filename,'rb+')as f:
        #获得文件句柄
        fileno=f.fileno()#返回一个整型的文件描述符，可用于底层操作系统的 I/O 操作
        print(fileno)
        while end<filesize-1:
            start=end+1
            end=start+step-1
            if end>filesize:
                end=filesize
            print ('Start:%s,end:%s'%(start,end))
            dup=os.dup(fileno)#复制文件句柄
            fd=os.fdopen(dup,'rb+',-1)
            t=Mythread(url,start,end,fd)
            t.start()
            mtd_list.append(t)
        for t in mtd_list:
            t.join()
    f.close()
    time2=datetime.datetime.now()
    use_time=time2-time1
    print ("Time used: "+str(use_time)[:-7]+", ")
    


def download_the_av(url):
    head={}
    head['User-Agent']='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    req = urllib2.Request(url,headers=head)
    response = urllib2.urlopen(req)
    content = response.read().decode("utf-8")
    a=[]
    name=[]
    m=[]
    while len(content)<100:
        print("try again...")
        content = urllib2.urlopen(req).read().decode("utf-8")
    print ("All length:" +str(len(content)))
    
    p=r'<title>([^-]+)-'
    name = re.findall(p,content)
    for each1 in name:
        title1=each1
    
    ky=r'>generate_down\(([^"]+) \+'
    x=re.findall(ky,content)
    for each in x:
        m=each
    kay=r' \+ "([^"]+)"\);</script>'
    a=re.findall(m+kay,content)
    for each2 in a:
        the_url='http://555.maomixia555.com:888'+each2
        content=each2
    
    m=re.findall(r'\.(.+)',content)
    for each3 in m:
        title2=each3
    title = title1 +'.'+title2
    
    thread_down(the_url,big_path,title)
    

def open_url(url):
    req=urllib2.Request(url)
    req.add_header('User-Agent','Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36')
    page = urllib2.urlopen(req)
    html = page.read().decode('utf-8')
    
    return html

urls=[]

def get_num(html):
    p=r'<li><a href="([^"]+)" target="_blank">'
    num_list = re.findall(p,html)
    for each in num_list:
        urls.append("https://www.899ii.com"+each)
    
if __name__ == '__main__':
    urlk = 'https://www.899ii.com/htm/downlist6/'
    #key = input("请输入你想看到的关键词：")
    #big_path=big_path+key+'\\'
    print("你想一口气来几部？")
    want=int(input("请输入数字:"))
    print('正在连接...')
    get_num(open_url(urlk))
    if not os.path.exists(big_path):
        os.makedirs(big_path)


print (len(urls))
print (" videos to download...")
count=0
for url in urls:
    print (count)
    count+=1
    if count<=want:
        download_the_av(url)
        continue
    else:
        break
print ("All done")