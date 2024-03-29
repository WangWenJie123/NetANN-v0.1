import numpy as np
import csv
import os
import threading
import psutil
import shutil
import time
import queue
import SPTAG

vector_dataset_path = '/app/wwj_test/remote_nvme_vector_datasets/sift1B/'
spfresh_index_path = '/app/wwj_test/SPFresh_Index'
csv_log_path = '/app/wwj_test/eva_logs/remote_spfresh_test_sift500M_cpu_disk.csv'

csv_log_title = ["dataset", "nprobe", "index_type", "processor", "search_latency/ms", "throughput/QPS", "R_1", "R_10", "R_100", "cpu_usage/%"]

DATASET = 'sift500M'
INDEX_TYPE = 'SPFresh_SPTAG'
PROCESSOR = 'CPU'
TOP_K = 100

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

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
        
def main():
    # create and open csv log file
    if not os.path.exists(csv_log_path):
        fd = open(csv_log_path, 'w')
        fd.close()
        print("create new csv log file")

    print("open csv log")
    csv_log_file = open(csv_log_path, 'a')
    csv_log_writer = csv.writer(csv_log_file)
    csv_log_writer.writerow(csv_log_title)
    
    # read base vectors
    xb = mmap_bvecs(vector_dataset_path + 'bigann_base.bvecs')
    xb = xb[:500 * 1000 * 1000]
    xb_num = xb.shape[0]
    xb_dim = xb.shape[1]
    
    # read query vectors
    xq = mmap_bvecs(vector_dataset_path + 'bigann_query.bvecs')
    # xq_num = xq.shape[0]
    xq_num = 2
    
    # create SPFresh metadata
    metadata = ''
    for i in range(xb_num):
        metadata += str(i) + '\n'
    
    # read groundtruth
    groundtruth = ivecs_read(vector_dataset_path + 'gnd/idx_%dM.ivecs' % 500)
        
    # create SPFresh index and save to disk
    index = SPTAG.AnnIndex('BKT', 'Float', xb_dim)

    index.SetBuildParam("NumberOfThreads", '28', "Index")

    index.SetBuildParam("DistCalcMethod", 'L2', "Index") 

    if (os.path.exists(spfresh_index_path)):
        shutil.rmtree(spfresh_index_path)
    if index.BuildWithMetaData(xb, metadata, xb_num, False, False):
        index.Save(spfresh_index_path) # Save the index to the disk

    os.listdir(spfresh_index_path)
    
    for lnprobe in range(1):
        # nprobe = 1 << lnprobe
        nprobe = 512
        
        # start search
        # Local index test on the vector search
        index_map_time_start = time.time()
        index = SPTAG.AnnIndex.Load(spfresh_index_path)
        index_map_time_end = time.time()
        index_map_time = (index_map_time_end - index_map_time_start) * 1000.0
        index.SetSearchParam('MaxCheck', str(nprobe), 'Index')
        index.SetSearchParam("NumberOfThreads", '110', "Index")
        
        # get cpu usage
        q = queue.Queue()
        ret_q = queue.Queue()
        cpu_monitor = CPU_Monitor(q, ret_q)
        q.put(0)
        cpu_monitor.start()
        
        # get avg search latency
        results = []
        t0 = time.time()
        for t in range(xq_num):
            result = index.SearchWithMetaData(xq[t], TOP_K)
            results.append(result[0])
        t1 = time.time()
        
        # get avg search latency
        q.put(1)
        cpu_monitor.join()
        cpu_usage = ret_q.get()
        
        latency = (t1 - t0) * 1000.0 / xq_num + index_map_time
        
        # calculate recalls
        recalls = {}
        j = 1
        while j <= TOP_K:
            # recalls[j] = (results[:, :j] == groundtruth[:, :1]).sum() / float(xq_num)
            recalls[j] = (results[0][:j] == groundtruth[:, :1]).sum() / float(xq_num)
            j *= 10
        
        # print and write evaluation data to csv log file
        csv_log_data = [DATASET, nprobe, INDEX_TYPE, PROCESSOR, latency, 1.0/(latency/1000.0), recalls[1], recalls[10], recalls[100], cpu_usage]
        csv_log_writer.writerow(csv_log_data)

        print("nprobe=%4d %.3f ms recalls= %.4f %.4f %.4f" % (nprobe, latency, recalls[1], recalls[10], recalls[100]))
        print("cpu usage:", cpu_usage)
        
    csv_log_file.close()

if __name__ == '__main__':
    main()