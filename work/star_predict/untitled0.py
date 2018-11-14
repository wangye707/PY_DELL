# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:52:00 2018

@author: sy
"""

from scipy import misc
import tensorflow as tf
import detect_face
import cv2
# import matplotlib.pyplot as plt
from PIL import Image
import os

# import scipy.misc
# %pylab inline
# fin = 'C:\\Users\\sy\\Desktop\\images'
# fout = 'C:\\Users\\sy\\Desktop\\image_裁剪'
def mtcnn_cut(fin,fout):

    fin = fin
    fout = fout
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    margin = 44
    frame_interval = 3
    batch_size = 1000
    image_size = 182
    input_image_size = 160

    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess,
                                                        r'.\20170512-110547')

    i = 0

    for file in os.listdir(fin):
        try:

            file_fullname = fin + '/' + file
            img = misc.imread(file_fullname)
            # i+= 1
            # img = misc.imread(image_path)
            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]  # 人脸数目
            print(nrof_faces)
            # print('找到人脸数目为：{}'.format(nrof_faces))

            # print(bounding_boxes)

            crop_faces = []
            if nrof_faces != 0:
                for face_position in bounding_boxes:
                    face_position = face_position.astype(int)
                    print(face_position[0:4])
                    cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]),
                                  (0, 255, 0), 2)
                    crop = img[face_position[1]:face_position[3],
                           face_position[0]:face_position[2], ]
                    # print(crop)
                    # crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)
                    crop_faces.append(crop)
                    img2 = Image.open(file_fullname)
                    a = face_position[0:4]
                    # print('crop_faces:',crop_faces)
                    # a = [face_position[0:4]]
                    box = (a)
                    roi = img2.crop(box)
                    i = roi.resize((250, 250))

                    out_path = fout + '/' + file

                    i.save(out_path)
                    print('success')
            else:
                pass
        except:
            pass

if __name__ == '__main__':
    fin = 'C:\\Users\\wy\\Desktop\\images'
    fout = 'C:\\Users\\wy\\Desktop\\image_裁剪'
    mtcnn_cut(fin,fout)
