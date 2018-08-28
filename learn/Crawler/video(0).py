#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:10:05 2018

@author: p
"""


import urllib.request as urllib2
import datetime
import re
import os.path


#to_find_string="https://bd.phncdn.com/videos/"
pa=os.getcwd()
big_path=pa+'\\'

def save_file(this_download_url,path):
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    time1=datetime.datetime.now()
    print (str(time1)[:-7])
    if (os.path.isfile(path)):
        file_size=os.path.getsize(path)/1024/1024
        print ("File "+path+" ("+ str(file_size)+"Mb) already exists.")
        return
    else:   
        print ("Downloading "+path+"...",)
        head={}
        head['User-Agent']='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
        f = urllib2.urlopen(this_download_url,headers=head) 
        data = f.read() 
        with open(path, "wb") as code:     
            code.write(data)  
        time2=datetime.datetime.now()
        print (str(time2)[:-7],)
        print (path+" Done.")
        use_time=time2-time1
        print ("Time used: "+str(use_time)[:-7]+", ")
        file_size=os.path.getsize(path)/1024/1024
        print ("File size: "+str(file_size)+" MB, Speed: "+str(file_size/(use_time.total_seconds()))[:4]+"MB/s")


def download_the_av(url):
    head={}
    head['User-Agent']='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    req = urllib2.Request(url,headers=head)
    response = urllib2.urlopen(req)
    content = response.read().decode("utf-8")
    while len(content)<100:
        print("try again...")
        content = urllib2.urlopen(req).read().decode("utf-8")
    print ("All length:" +str(len(content)))
    
    p=r'<meta name="twitter:title" content="([^"]+)">'
    name = re.findall(p,content)
    for each in name:
        title=each

    quality=['720','480','240']
    for i in quality:
        find_position=content.find("\"quality\":\""+i+"\"")
        if find_position>0:
            print ("Quality: "+i+"P")
            break
    to_find=content[find_position:find_position+4000]

    pattern=re.compile(r"\"videoUrl\":\"[^\"]*\"")
    match = pattern.search(to_find) 
    if match:
        the_url=match.group() 
    the_url=the_url[12:-1]#the real url
    the_url=the_url.replace("\\/","/")
    save_file(the_url,big_path+title+".mp4")


def open_url(url):
    req=urllib2.Request(url)
    req.add_header('User-Agent','Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36')
    page = urllib2.urlopen(req)
    html = page.read().decode('utf-8')
    
    return html

urls=[]

def get_num(html):
    flag=True
    p=r'<a href="/view_video\.php\?viewkey=ph5([^"]+)" title='
    num_list = re.findall(p,html)
    count=0
    for each in num_list:
        if count<=4:
            count+=1
            continue
        else:    
            if flag==True:
                urls.append("https://www.pornhub.com/view_video.php?viewkey=ph5"+each)
                flag=False
            else:
                flag=True
                continue
    
if __name__ == '__main__':
    urlk = 'https://www.pornhub.com/video/search?search='
    key = input("请输入你想看到的关键词：")
    big_path=big_path+key+'\\'
    get_num(open_url(urlk+key))
    if not os.path.exists(big_path):
        os.makedirs(big_path)


print (len(urls))
print (" videos to download...")
count=0

for url in urls:
    print (count)
    count+=1
    if count<=2:
        download_the_av(url)
        continue
    else:
        break
print ("All done")