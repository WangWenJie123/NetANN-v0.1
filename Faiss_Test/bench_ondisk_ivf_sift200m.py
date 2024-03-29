# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python2

import sys
import numpy as np
import csv
import os
import threading
import psutil
import queue
import time
# import mkl
# mkl.get_max_threads()
# sys.path.append('/home/wwj/Vector_DB_Acceleration/faiss/python')
from datasets import evaluate
import faiss

sift1M_base_dataPath = "/home/wwj/Vector_DB_Acceleration/vector_datasets/sift1B/bigann_base.bvecs"
sift1M_groundtruth_dataPath = "/home/wwj/Vector_DB_Acceleration/vector_datasets/sift1B/gnd/idx_200M.ivecs"
sift1M_learn_dataPath = "/home/wwj/Vector_DB_Acceleration/vector_datasets/sift1B/bigann_learn.bvecs"
sift1M_query_dataPath = "/home/wwj/Vector_DB_Acceleration/vector_datasets/sift1B/bigann_query.bvecs"

# sift1M_base_dataPath = "/home/wwj/Vector_DB_Acceleration/r740_vector_datasets/sift1B/bigann_base.bvecs"
# sift1M_groundtruth_dataPath = "/home/wwj/Vector_DB_Acceleration/r740_vector_datasets/sift1B/gnd/idx_200M.ivecs"
# sift1M_learn_dataPath = "/home/wwj/Vector_DB_Acceleration/r740_vector_datasets/sift1B/bigann_learn.bvecs"
# sift1M_query_dataPath = "/home/wwj/Vector_DB_Acceleration/r740_vector_datasets/sift1B/bigann_query.bvecs"

csv_log_path = "/home/wwj/Vector_DB_Acceleration/faiss/python/wwj_test_codes/eva_logs/local_sift200M_cpu_mmap.csv"

vbSize = 1000000000

dataset = "sift200M"
processor = "cpu"
# index_type = "IVF512,Flat"
# index_type = "IVF1024,Flat"
# index_type = "IVF2048,Flat"
# index_type = "IVF4096,Flat"
# index_type = "IVF512,PQ32"
# index_type = "IVF1024,PQ32"
# index_type = "IVF2048,PQ32"
index_type = "IVF4096,PQ32"
csv_log_title = ["dataset", "nprobe", "index_type", "processor", "search_latency/ms", "throughput/QPS", "R_1", "R_10", "R_100", "cpu_usage/%"]

#################################################################
# Small I/O functions
#################################################################


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def bvecs_read_learn(fname, c_contiguous=True):
   fv = np.fromfile(fname, dtype=np.byte)
   if fv.size == 0:
       return np.zeros(0, 0)
   dim = fv.view(np.int32)[0]  # 更改数据类型为int32,且查看该数组的第一个维度
   assert dim > 0
   fv = fv.reshape(-1, 4+dim)    #
   if not all(fv.view(np.int32)[:, 0] == dim):
       raise IOError("Non-uniform vector sizes in " + fname)
   fv = fv[:, 4:]
   if c_contiguous:
       fv = fv.copy()
   return fv

def bvecs_read_base(fname, c_contiguous=True):
   fv = np.fromfile(fname, dtype=np.byte)
   if fv.size == 0:
       return np.zeros(0, 0)
   dim = fv.view(np.int32)[0]  # 更改数据类型为int32,且查看该数组的第一个维度
   assert dim > 0
   fv = fv.reshape(-1, 4+dim)    #
   if not all(fv.view(np.int32)[:, 0] == dim):
       raise IOError("Non-uniform vector sizes in " + fname)
   fv = fv[:, 4:]
   if c_contiguous:
       fv = fv.copy()
   return fv[0:int(vbSize/10*2), :]

# def bvecs_read(fname):
#     x = np.memmap(fname, dtype='uint8', mode='r')
#     d = x[:4].view('int32')[0]
#     return x.reshape(-1, d + 4)[:, 4:]

class CPU_Monitor(threading.Thread):
    def __init__(self, q, ret_q):
        super(CPU_Monitor, self).__init__() 
        self.q = q
        self.ret_q = ret_q
        self.cpu_usage = 0
        
    def run(self):
        while self.q.get() == 0:
            self.cpu_usage = psutil.cpu_percent(None)
        
        self.ret_q.put(self.cpu_usage)

#################################################################
#  Main program
#################################################################

stage = int(sys.argv[1])

tmpdir = '/home/wwj/Vector_DB_Acceleration/vector_datasets/sift1B/trained_index_200M/'

# if stage == 0:
if 1:
    # train the index
    xt = bvecs_read_learn(sift1M_learn_dataPath)
    index = faiss.index_factory(xt.shape[1], index_type)
    print("training index")
    index.train(xt)
    print("write " + tmpdir + index_type + "trained.index")
    faiss.write_index(index, tmpdir + index_type + "trained.index")


# if 1 <= stage <= 4:
for stage in range(1, 5):
    # add 1/4 of the database to 4 independent indexes
    bno = stage - 1
    xb = bvecs_read_base(sift1M_base_dataPath)
    i0, i1 = int(bno * xb.shape[0] / 4), int((bno + 1) * xb.shape[0] / 4)
    index = faiss.read_index(tmpdir + index_type + "trained.index")
    print("adding vectors %d:%d" % (i0, i1))
    index.add_with_ids(xb[i0:i1], np.arange(i0, i1))
    # index.add(xb[i0:i1])
    print("write " + tmpdir + index_type +  "block_%d.index" % bno)
    faiss.write_index(index, tmpdir + index_type + "block_%d.index" % bno)


# if stage == 5:
if 1:
    # merge the images into an on-disk index
    # first load the inverted lists
    ivfs = []
    for bno in range(4):
        # the IO_FLAG_MMAP is to avoid actually loading the data thus
        # the total size of the inverted lists can exceed the
        # available RAM
        print("read " + tmpdir + index_type + "block_%d.index" % bno)
        index = faiss.read_index(tmpdir + index_type + "block_%d.index" % bno,
                                 faiss.IO_FLAG_MMAP)
        ivfs.append(index.invlists)

        # avoid that the invlists get deallocated with the index
        index.own_invlists = False

    # construct the output index
    index = faiss.read_index(tmpdir + index_type + "trained.index")

    # prepare the output inverted lists. They will be written
    # to merged_index.ivfdata
    invlists = faiss.OnDiskInvertedLists(
        index.nlist, index.code_size,
        tmpdir + index_type + "merged_index.ivfdata")

    # merge all the inverted lists
    ivf_vector = faiss.InvertedListsPtrVector()
    for ivf in ivfs:
        ivf_vector.push_back(ivf)

    print("merge %d inverted lists " % ivf_vector.size())
    ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())

    # now replace the inverted lists in the output index
    index.ntotal = ntotal
    index.replace_invlists(invlists)

    print("write " + tmpdir + index_type + "populated.index")
    faiss.write_index(index, tmpdir + index_type + "populated.index")


if 1:
    if not os.path.exists(csv_log_path):
        fd = open(csv_log_path, 'w')
        fd.close()
        print("create new csv log file")

    print("open csv log")
    csv_log_file = open(csv_log_path, 'a')
    csv_log_writer = csv.writer(csv_log_file)
    csv_log_writer.writerow(csv_log_title)
    
    for lnprobe in range(10):
        nprobe = 1 << lnprobe
        # perform a search from disk
        # print("read " + tmpdir + index_type + "populated.index")
        index_map_time_start = time.time()
        index = faiss.read_index(tmpdir + index_type + "populated.index")
        index_map_time_end = time.time()
        index_map_time = (index_map_time_end - index_map_time_start) * 1000.0
        if index_type != "Flat" and index_type != "PQ32" and index_type != "PCA80,Flat":
            index.nprobe = nprobe

        # load query vectors and ground-truth
        xq = bvecs_read_learn(sift1M_query_dataPath)
        gt = ivecs_read(sift1M_groundtruth_dataPath)
        
        q = queue.Queue()
        ret_q = queue.Queue()
        cpu_monitor = CPU_Monitor(q, ret_q)
        q.put(0)
        cpu_monitor.start()

        t, r = evaluate(index, xq, gt, 100)
        
        total_latency = index_map_time + t
        
        q.put(1)
        cpu_monitor.join()
        cpu_usage = ret_q.get()
        
        csv_log_data = [dataset, nprobe, index_type, processor, total_latency, 1.0/(total_latency/1000.0), r[1], r[10], r[100], cpu_usage]
        csv_log_writer.writerow(csv_log_data)

        print("nprobe=%4d %.3f ms recalls= %.4f %.4f %.4f" % (nprobe, total_latency, r[1], r[10], r[100]))
        print("cpu usage:", cpu_usage)
    
    csv_log_writer.writerow(['', '', '', '', '', '', '', '', '', ''])

    csv_log_file.close()
