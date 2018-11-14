#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : dist_rate_test.py
# @Author: WangYe
# @Date  : 2018/10/24
# @Software: PyCharm
# 用随机数测试分布式各个节点速度
import tensorflow as tf

# Configuration of cluster
ps_hosts = [ "ps_host","192.168.1.114:1671,192.168.1.115:1672"]
worker_hosts = [ "worker_host","192.168.1.114:1672,192.168.1.115:1671" ]
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

def main(_):
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    server.join()

if __name__ == "__main__":
    tf.app.run()