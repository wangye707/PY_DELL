#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: WangYe
# @Date  : 2018/9/4
# @Software: PyCharm
# !/usr/bin/python
# -*- coding: UTF-8 -*-
import urllib.parse, urllib.request
# 这是python3，原先py2里的urllib2或者其他都包含在了py3的urllib里了，
# py3里的urllib里的parse和request一定要这么导入，直接import urllib
# 是不行的

import time
import json
import hashlib
import base64
import numpy as np

def main():
    f = open(r"C:\Users\wy\Desktop\data\科大讯飞输出\test_1.pcm", 'rb')
    # rb表示以二进制格式只读打开文件
    #f = open("sw02725-A_018202-018583#sw02141-B_013798-014064.wav")
    # f.seek(0)
    # f.read(44)
    # data = np.fromfile(f, dtype=np.int16)
    # data.tofile("test.pcm")


    #file_content = f.read()
    file_content=f.read()
    # file_content 是二进制内容，bytes类型
    # 由于Python的字符串类型是str，在内存中以Unicode表示，一个字符对应若干个字节。
    # 如果要在网络上传输，或者保存到磁盘上，就需要把str变为以字节为单位的bytes
    # 以Unicode表示的str通过encode()方法可以编码为指定的bytes，例如：
    # >> > 'ABC'.encode('ascii')
    # b'ABC'
    # >> > '中文'.encode('utf-8')
    # b'\xe4\xb8\xad\xe6\x96\x87'
    # >> > '中文'.encode('ascii')
    # Traceback(most
    # recent
    # call
    # last):
    # File
    # "<stdin>", line 1, in < module >
    # UnicodeEncodeError: 'ascii' codec can't encode characters in position 0-1: ordinal not in range(128)

    base64_audio = base64.b64encode(file_content)
    # base64.b64encode()参数是bytes类型，返回也是bytes类型

    body = urllib.parse.urlencode({'audio': base64_audio})
    url = 'http://api.xfyun.cn/v1/service/v1/iat'
    api_key = '25ea09c47d03108ac043391960a01dc2'  # api key在这里
    x_appid = '5b8e2771'  # appid在这里
    param = {"engine_type": "sms16k", "aue": "raw"}

    x_time = int(int(round(time.time() * 1000)) / 1000)

    x_param = base64.b64encode(json.dumps(param).replace(' ', '').encode('utf-8'))
    # 这是3.x的用法，因为3.x中字符都为unicode编码，而b64encode函数的参数为byte类型，
    # 所以必须先转码为utf-8的bytes

    # >> print(x_param)
    # >> b'YWJjcjM0cjM0NHI ='
    # 结果和我们预想的有点区别，我们只想要获得YWJjcjM0cjM0NHI =，而字符串被b
    # ''包围了。这时肯定有人说了，用正则取出来就好了。。。别急。b表示
    # byte的意思，我们只要再将byte转换回去就好了:
    # >> x_param = str(x_param, 'utf-8')

    # Python3 字符编码 https://www.cnblogs.com/284628487a/p/5584714.html

    x_checksum_content = api_key + str(x_time) + str(x_param, 'utf-8')
    x_checksum = hashlib.md5(x_checksum_content.encode('utf-8')).hexdigest()
    # python3里的hashlib.md5()参数也是要求bytes类型的，x_checksum_content是以Unicode
    # 编码的，所以需要转成bytes。
    # 讯飞api说明：
    # 授权认证，调用接口需要将Appid，CurTime, Param和CheckSum信息放在HTTP请求头中；
    # 接口统一为UTF-8编码；
    # 接口支持http和https；
    # 请求方式为POST。

    x_header = {'X-Appid': x_appid,
                'X-CurTime': x_time,
                'X-Param': x_param,
                'X-CheckSum': x_checksum,
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8"
                }

    req = urllib.request.Request(url=url, data=body.encode('utf-8'), headers=x_header, method='POST')
    # 不要忘记url = ??, data = ??, headers = ??, method = ?? 中的“ = ”，这是python3！！

    result = urllib.request.urlopen(req)
    result = result.read().decode('utf-8')
    # 返回的数据需要再以utf-8解码

    print('结果',result)
    return


if __name__ == '__main__':
    main()

#"code":"0",
# "data":"",
# "desc":"success",
# "sid":"zat004be2b6@ch101c0ee97993515100"}
