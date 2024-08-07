'''
 * @Author: Wenjie Wang
 * @Date: 2024-03-27 14:39:10
'''
 
import sys
import numpy as np
import csv
import os
import threading
import psutil
import queue
import time
from datasets import evaluate, evaluate_without_recalls
import deep1B_text1B_dataset
import faiss

base_dataPath = "/home/wwj/Vector_DB_Acceleration/Vector_Datasets/Remote_Nvme_Vector_Datasets/text1B/local_nvme_text1B_base/base1B.fbin"
groundtruth_dataPath = "/home/wwj/Vector_DB_Acceleration/Vector_Datasets/Remote_Nvme_Vector_Datasets/text1B/groundtruth-public100K.ibin"
query_dataPath = "/home/wwj/Vector_DB_Acceleration/Vector_Datasets/Remote_Nvme_Vector_Datasets/text1B/query100K.fbin"

csv_log_path = "/home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/eva_logs/faiss_remote_text200M_cpu_disk.csv"

dataset = "text200M"
processor = "cpu"
# index_type = "IVF512,Flat"
# index_type = "IVF1024,Flat"
# index_type = "IVF2048,Flat"
index_type = "IVF4096,Flat"
csv_log_title = ["dataset", "nprobe", "index_type", "processor", "search_latency/ms", "throughput/QPS", "R_1", "R_10", "R_100", "cpu_usage/%"]

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

tmpdir = '/home/wwj/Vector_DB_Acceleration/Vector_Datasets/Remote_Nvme_Vector_Datasets/text1B/faiss_tarined_index/'

if 1:
    # train the index
    xt = deep1B_text1B_dataset.read_fbin(filename=base_dataPath, start_idx=0, chunk_size=200000000)
    print("xt_num: {}, xt_dim: {}".format(xt.shape[0], xt.shape[1]))
    index = faiss.index_factory(xt.shape[1], index_type)
    print("training index")
    index.train(xt)
    print("write " + tmpdir + index_type + "trained.index")
    faiss.write_index(index, tmpdir + index_type + "trained.index")

for stage in range(1, 5):
    bno = stage - 1
    xb = deep1B_text1B_dataset.read_fbin(filename=base_dataPath, start_idx=0, chunk_size=200000000)
    print("xb_num: {}, xb_dim: {}".format(xb.shape[0], xb.shape[1]))
    i0, i1 = int(bno * xb.shape[0] / 4), int((bno + 1) * xb.shape[0] / 4)
    index = faiss.read_index(tmpdir + index_type + "trained.index")
    print("adding vectors %d:%d" % (i0, i1))
    index.add_with_ids(xb[i0:i1], np.arange(i0, i1))
    # index.add(xb[i0:i1])
    print("write " + tmpdir + index_type +  "block_%d.index" % bno)
    faiss.write_index(index, tmpdir + index_type + "block_%d.index" % bno)


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
        xq = deep1B_text1B_dataset.read_fbin(filename=query_dataPath, start_idx=0, chunk_size=None)
        print("xq_num: {}, xq_dim: {}".format(xq.shape[0], xq.shape[1]))
        gt = deep1B_text1B_dataset.read_ibin(filename=groundtruth_dataPath, start_idx=0, chunk_size=None)
        
        q = queue.Queue()
        ret_q = queue.Queue()
        cpu_monitor = CPU_Monitor(q, ret_q)
        q.put(0)
        cpu_monitor.start()

        t, r = evaluate_without_recalls(index, xq, gt, 100)
        
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
