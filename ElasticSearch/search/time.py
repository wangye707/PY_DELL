import time
import sched

schedule = sched.scheduler(time.time, time.sleep)


def func():
    from .read import Read
    upfilepath = r'C:\Users\wy\Desktop\data\elasticsearch'
    red = Read()
    red.eachFile_0(filepath=upfilepath)


print(time.time())
schedule.enter(1, 0, func)
schedule.run()
print(time.time())