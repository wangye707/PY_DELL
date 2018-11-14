#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : transform_test.py
# @Author: WangYe
# @Date  : 2018/9/11
# @Software: PyCharm
from pydub import AudioSegment


# def trans_mp3_to_wav(filepath):
#     song = AudioSegment.from_wav(filepath)
#     song.export("now.avi", format="avi")
#     #pcm :s16le
# filepath=r'C:\Users\wy\Desktop\data\keda\2.wav'
# trans_mp3_to_wav(filepath)
# from pydub import AudioSegment
sound = AudioSegment.from_wav("C:\\Users\\wy\\Desktop\\data\\keda\\2.wav")
sound.export("C:\\Users\\wy\\Desktop\\data\\keda\\2.mp3", format="mp3")
