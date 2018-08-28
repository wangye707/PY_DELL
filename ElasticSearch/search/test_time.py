import datetime
import time


def doSth():

    from .read import Read
    upfilepath = r'F:\data\ceshi'
    red = Read()
    red.eachFile_0(filepath=upfilepath)

    time.sleep(3600)



def main(h=12, m=1):
    '''h表示设定的小时，m为设定的分钟'''
    while True:
        # 判断是否达到设定时间，例如0:00
        while True:
            now = datetime.datetime.now()
            # 到达设定时间，结束内循环
            if now.hour == h and now.minute == m:
                break
            # 不到时间就等20秒之后再次检测
            time.sleep(20)
        # 做正事，一天做一次
        doSth()
main()