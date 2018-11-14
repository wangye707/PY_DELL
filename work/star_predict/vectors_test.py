import os
import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
#import facenet
from imutils import paths
from tensorflow.python.platform import gfile
import re
from work.star_predict.untitled0 import mtcnn_cut
import shutil
from PIL import Image

from app import run_app

image_vector_list = None

def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def delete_file(path):
    # shutil.rmtree(path)
    # os.makedirs(path)
    ls = os.listdir(path)
    #print(type(ls))
    for i in ls:
        #if os._exists(i):
        c_path = os.path.join(path, i)
        # print(c_path)
        # print(c_path)
        os.remove(c_path)
        print('正在预处理...')
        # else:
        #     print('123')
        #     break

def prewhiten(x):  #
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y



import os
def name_read(path):

    name_list=[]
    for (root, dirs, files) in os.walk(path):  #列出目录下的所有文件和文件名
        for filename in files:
            #print(os.path.join(root,filename))
            pathload=str(os.path.join(root,filename))  #遍历的获取文件名
            #print(pathload)
            name_list.append(pathload)
        print(name_list)
    return name_list
def vector(name_list):
    image_vector=[]  #照片向量矩阵

    for i in range(len(name_list)):
        scaled_reshape = []
        image1 = scipy.misc.imread(name_list[i], mode='RGB')
        image1 = cv2.resize(image1, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        image1 = prewhiten(image1)
        scaled_reshape.append(image1.reshape(-1, image_size, image_size, 3))
        emb_array1 = np.zeros((1, embedding_size))
        emb_array1[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0], phase_train_placeholder: False})[0]
        #print(emb_array1)
        image_vector.append(emb_array1)
        if i%50 == 0:
            print('已读取',i)
       # sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0], phase_train_placeholder: False})[0]
    return image_vector


def user_image_packing(inputfile):   #图像压缩
    outputfile = r'.\datasets\user_image_packing'
    flag = 1024 * 1024

    for img in os.listdir(inputfile):
        file_fullname = inputfile + '/' + img
        im = Image.open(file_fullname)
        print(im.format, im.size, im.mode)
        size_tmp = os.path.getsize(file_fullname)
        (x, y) = im.size
        if size_tmp > flag:
            if x > 1920:
                x_s = 1920
                y_s = int(y * 1920 / x)
                out = im.resize((x_s, y_s), Image.ANTIALIAS)
                out.save(outputfile + '/' + img)
                print(out.format, out.size, out.mode)
        else:
            shutil.copy(inputfile + '/' + img, outputfile + '/' + img)
    return outputfile

# def user_image_vector(image_path):
#
#     image_path=name_read(image_path)  #图片文件夹转图片路径
#     #print('11111',user_path_list)
#     image_vector_list=vector(image_path)    # 明星库向量矩阵
#     #print('222',user_vector[0].shape)
#     while 1:
#             print('请输入文件路径')
#             user_path = input()
#             #delete_file(user_path)  # 清空用户输入图片  ./datasets/user_input_image
#             #pathin='./datasets/user_image_packing'
#             #delete_file(pathin)  # 清空压缩后图片 ./datasets/user_image
#             #pathout = './user_cut'
#             #delete_file(pathout)  # 清空用户的人脸对齐后图片  ./user_cut
#             if user_path == str(0):
#                 break
#             else:
#                 try:
#                     pathin=user_image_packing(user_path)
#                 except:
#                     print('路径异常')
#                     continue
#
#
#                 pathout = r'.\user_cut'
#
#                 mtcnn_cut(pathin,pathout)  #对输入图像裁剪
#
#                 user_path_list = name_read(pathout)
#
#                 name_vector = vector(user_path_list)
#
#                 '''清空预处理文件'''
#                 delete_file(user_path)  # 清空用户输入图片  ./datasets/user_input_image
#                 # pathin='./datasets/user_image_packing'
#                 delete_file(pathin)  # 清空压缩后图片 ./datasets/user_image
#                 # pathout = './user_cut'
#                 delete_file(pathout)  # 清空用户的人脸对齐后图片  ./user_cut
#
#
#                 print('相似图片')
#                 same_image = []  # 最终相似图片矩阵
#                 for i in range(len(image_vector_list)):
#                     # print(i)
#                     try:
#                         dist = np.sqrt(np.sum(np.square(name_vector - image_vector_list[i])))  # 两张图像的欧氏距离
#                     except:
#                         print('请重新输入图片')
#                         break
#                     else:
#                         same_image.append(dist)
#
#                 max_location = sorted(enumerate(same_image), key=lambda x: x[1], reverse=False)  # 返回
#
#                 for i in range(3):
#                     # print(max_location[i][0])    #反馈最小（最相似）的欧氏距离
#                     try:
#                         image_id = max_location[i][0]
#                         #print('相似图片路径:', image_path[image_id])
#                     except(IndexError):
#                         print('error')
#                         break
#                     else:
#                         print('相似图片路径:', image_path[image_id])

def inter(image_vector_list1=image_vector_list):



    user_path=r'.\datasets\user_input_image'
    pathout = r'.\user_cut'
    pathin = user_image_packing(user_path)
    mtcnn_cut(pathin, pathout)  # 对输入图像裁剪

    user_path_list = name_read(pathout)
    name_vector = vector(user_path_list)

    '''清空预处理文件'''
    delete_file(user_path)  # 清空用户输入图片  ./datasets/user_input_image
    # pathin='./datasets/user_image_packing'
    delete_file(pathin)  # 清空压缩后图片 ./datasets/user_image
    # pathout = './user_cut'
    delete_file(pathout)  # 清空用户的人脸对齐后图片  ./user_cut

    print('相似图片')
    same_image = []  # 最终相似图片矩阵
    for i in range(len(image_vector_list1)):
        # print(i)
        try:
            dist = np.sqrt(np.sum(np.square(name_vector - image_vector_list[i])))  # 两张图像的欧氏距离
        except:
            print('请重新输入图片')
            break
        else:
            same_image.append(dist)

    max_location = sorted(enumerate(same_image), key=lambda x: x[1], reverse=False)  # 返回
    # print(max_location[0])
    return max_location[0]

    #
    # print(id, "ddddddd")
    # # res = pathin[id]
    # # print(res, "rrrrrrrrrrrr")
    # return _image_path[id]

    # image_id = max_location[i][0]



# def compute(user_vector,image_path):
#
#     image_path_list = name_read(image_path)
#
#     base_vector = vector(image_path_list)  # 明星库向量矩阵
#
#     same_image = []  # 最终相似图片矩阵
#     for i in range(len(base_vector)):
#         # print(i)
#         dist = np.sqrt(np.sum(np.square(user_vector - base_vector[i])))  # 两张图像的欧氏距离
#         same_image.append(dist)
#
#     max_location = sorted(enumerate(same_image), key=lambda x: x[1], reverse=False)  # 返回
#
#     for i in range(3):
#         # print(max_location[i][0])    #反馈最小（最相似）的欧氏距离
#         image_id = max_location[i][0]
#         print('相似图片路径:', image_path_list[image_id])
#
# def main():
#     # delete_file(r'C:\Users\sy\Desktop\新建文件夹 (2)')
#     image_size = 200  # don't need equal to real image size, but this value should not small than this
#     modeldir = r'.\20170512-110547\20170512-110547.pb'  # change to your model dir
#
#     print('建立facenet embedding模型')
#     tf.Graph().as_default()
#     sess = tf.Session()
#
#     load_model(modeldir)
#     images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
#     embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#     phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
#     embedding_size = embeddings.get_shape()[1]
#     print('facenet embedding模型建立完毕')
#
#     # image_path=r'.\image_test'
#     #
#     # user_image_vector(image_path)
#     # image_path = r'.\user_cut'
#     # delete_file(image_path)
#     user_path = r'./datasets/user_input_image'
#     image_path = r'./image_test'
#     image_path = name_read(image_path)  # 图片文件夹转图片路径
#     # print('11111',user_path_list)
#     image_vector_list = vector(image_path)  # 明星库向量矩阵
#     # print('222',user_vector[0].shape)
#     return image_vector_list

if __name__ == '__main__':

    #delete_file(r'C:\Users\sy\Desktop\新建文件夹 (2)')
    image_size = 200 #don't need equal to real image size, but this value should not small than this
    modeldir = r'.\20170512-110547\20170512-110547.pb' #change to your model dir

    print('建立facenet embedding模型')
    tf.Graph().as_default()
    sess = tf.Session()

    load_model(modeldir)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    print('facenet embedding模型建立完毕')

    #image_path=r'.\image_test'

    #user_image_vector(image_path)

    user_path = r'./datasets/user_input_image'
    image_path = r'./image_test'
    image_path = name_read(image_path)  # 图片文件夹转图片路径
    # print('11111',user_path_list)
    image_vector_list = vector(image_path)  # 明星库向量矩阵
    # print('222',user_vector[0].shape)

    # inter(image_vector_list)
    run_app()