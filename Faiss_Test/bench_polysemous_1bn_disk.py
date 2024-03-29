# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import os
import csv
import sys
import time
import threading
import psutil
import queue
import numpy as np
import re
import faiss
from multiprocessing.dummy import Pool as ThreadPool
from datasets import ivecs_read

data_set_path = "/home/wwj/Vector_DB_Acceleration/remote_nvme_vector_datasets/sift1B/"
csv_log_path = "/home/wwj/Vector_DB_Acceleration/faiss/python/wwj_test_codes/eva_logs/remote_sift500M_cpu_mmap.csv"
csv_log_title = ["dataset", "nprobe", "index_type", "processor", "search_latency/ms", "throughput/QPS", "R_1", "R_10", "R_100", "cpu_usage/%"]

# we mem-map the biggest files to avoid having them in memory all at
# once


def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


#################################################################
# Bookkeeping
#################################################################

# index_type = "IVF512,Flat"
# index_type = "IVF1024,Flat"
# index_type = "IVF2048,Flat"
# index_type = "IVF4096,Flat"
# index_type = "IVF512,PQ32"
# index_type = "IVF1024,PQ32"
# index_type = "IVF2048,PQ32"
# index_type = "IVF4096,PQ32"

dbname        = sys.argv[1]
index_key     = sys.argv[2]
parametersets = sys.argv[3:]


tmpdir = '/home/wwj/Vector_DB_Acceleration/remote_nvme_vector_datasets/faiss_trained_index'

if not os.path.isdir(tmpdir):
    print("%s does not exist, creating it" % tmpdir)
    os.mkdir(tmpdir)


#################################################################
# Prepare dataset
#################################################################

print("Preparing dataset", dbname)

if dbname.startswith('SIFT'):
    # SIFT1M to SIFT1000M
    dbsize = int(dbname[4:-1])
    xb = mmap_bvecs(data_set_path + 'bigann_base.bvecs')
    xq = mmap_bvecs(data_set_path + 'bigann_query.bvecs')
    xt = mmap_bvecs(data_set_path + 'bigann_learn.bvecs')

    # trim xb to correct size
    xb = xb[:dbsize * 1000 * 1000]

    gt = ivecs_read(data_set_path + 'gnd/idx_%dM.ivecs' % dbsize)

elif dbname == 'Deep1B':
    xb = mmap_fvecs('deep1b/base.fvecs')
    xq = mmap_fvecs('deep1b/deep1B_queries.fvecs')
    xt = mmap_fvecs('deep1b/learn.fvecs')
    # deep1B's train is is outrageously big
    xt = xt[:10 * 1000 * 1000]
    gt = ivecs_read('deep1b/deep1B_groundtruth.ivecs')

else:
    print('unknown dataset', dbname, file=sys.stderr)
    sys.exit(1)


print("sizes: B %s Q %s T %s gt %s" % (
    xb.shape, xq.shape, xt.shape, gt.shape))

nq, d = xq.shape
nb, d = xb.shape
assert gt.shape[0] == nq


#################################################################
# Training
#################################################################

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

def choose_train_size(index_key):

    # some training vectors for PQ and the PCA
    n_train = 256 * 100

    if "IVF" in index_key:
        matches = re.findall('IVF([0-9]+)', index_key)
        ncentroids = int(matches[0])
        # n_train = max(n_train, 100 * ncentroids)
        n_train = min(n_train, 100 * ncentroids)
    elif "IMI" in index_key:
        matches = re.findall('IMI2x([0-9]+)', index_key)
        nbit = int(matches[0])
        n_train = max(n_train, 256 * (1 << nbit))
    return n_train


def get_trained_index():
    filename = "%s/%s_%s_trained.index" % (
        tmpdir, dbname, index_key)

    if not os.path.exists(filename):
        index = faiss.index_factory(d, index_key)

        n_train = choose_train_size(index_key)

        xtsub = xt[:n_train]
        print("Keeping %d train vectors" % xtsub.shape[0])
        # make sure the data is actually in RAM and in float
        xtsub = xtsub.astype('float32').copy()
        index.verbose = True

        t0 = time.time()
        index.train(xtsub)
        index.verbose = False
        print("train done in %.3f s" % (time.time() - t0))
        print("storing", filename)
        faiss.write_index(index, filename)
    else:
        print("loading", filename)
        index = faiss.read_index(filename)
    return index


#################################################################
# Adding vectors to dataset
#################################################################

def rate_limited_imap(f, l):
    'a thread pre-processes the next element'
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i[0], i[1]))
        if res:
            yield res.get()
        res = res_next
    yield res.get()


def matrix_slice_iterator(x, bs):
    " iterate over the lines of x in blocks of size bs"
    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]

    return rate_limited_imap(
        lambda i0, i1: x[i0:i1].astype('float32').copy(),
        block_ranges)


def get_populated_index():

    filename = "%s/%s_%s_populated.index" % (
        tmpdir, dbname, index_key)

    if not os.path.exists(filename):
        index = get_trained_index()
        i0 = 0
        t0 = time.time()
        for xs in matrix_slice_iterator(xb, 100000):
            i1 = i0 + xs.shape[0]
            print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
            sys.stdout.flush()
            index.add(xs)
            i0 = i1
        print()
        print("Add done in %.3f s" % (time.time() - t0))
        print("storing", filename)
        faiss.write_index(index, filename)
    else:
        print("loading", filename)
        index = faiss.read_index(filename)
    return index


#################################################################
# Perform searches
#################################################################

if not os.path.exists(csv_log_path):
    fd = open(csv_log_path, 'w')
    fd.close()
    print("create new csv log file")

print("open csv log")
csv_log_file = open(csv_log_path, 'a')
csv_log_writer = csv.writer(csv_log_file)
csv_log_writer.writerow(csv_log_title)

filename = "%s/%s_%s_populated.index" % (
        tmpdir, dbname, index_key)

if not os.path.exists(filename):
    index = get_trained_index()
    i0 = 0
    t0 = time.time()
    for xs in matrix_slice_iterator(xb, 100000):
        i1 = i0 + xs.shape[0]
        print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
        sys.stdout.flush()
        index.add(xs)
        i0 = i1
    print()
    print("Add done in %.3f s" % (time.time() - t0))
    print("storing", filename)
    faiss.write_index(index, filename)
    del index

if parametersets == ['autotune'] or parametersets == ['autotuneMT']:

    if parametersets == ['autotune']:
        faiss.omp_set_num_threads(56)

    # setup the Criterion object: optimize for 1-R@1
    crit = faiss.OneRecallAtRCriterion(nq, 1)
    # by default, the criterion will request only 1 NN
    crit.nnn = 100
    crit.set_groundtruth(None, gt.astype('int64'))

    # then we let Faiss find the optimal parameters by itself
    print("exploring operating points")

    t0 = time.time()
    op = ps.explore(index, xq, crit)
    print("Done in %.3f s, available OPs:" % (time.time() - t0))

    # opv is a C++ vector, so it cannot be accessed like a Python array
    opv = op.optimal_pts
    print("%-40s  1-R@1     time" % "Parameters")
    for i in range(opv.size()):
        opt = opv.at(i)
        print("%-40s  %.4f  %7.3f" % (opt.key, opt.perf, opt.t))

else:

    # we do queries in a single thread
    faiss.omp_set_num_threads(110)

    print(' ' * len(parametersets[0]), '\t', 'R@1    R@10   R@100     time    %pass')

    # for param in parametersets:
    for lnprobe in range(10):
        nprobe = 1 << lnprobe
        param = "nprobe=" + str(nprobe)
        
        index_map_time_start = time.time()
        index = faiss.read_index(filename)
        index_map_time_end = time.time()
        index_map_time = (index_map_time_end - index_map_time_start) * 1000.0

        ps = faiss.ParameterSpace()
        ps.initialize(index)

        # make sure queries are in RAM
        xq = xq.astype('float32').copy()

        # a static C++ object that collects statistics about searches
        ivfpq_stats = faiss.cvar.indexIVFPQ_stats
        
        print(param, '\t', end=' ')
        sys.stdout.flush()
        ps.set_index_parameters(index, param)
        
        q = queue.Queue()
        ret_q = queue.Queue()
        cpu_monitor = CPU_Monitor(q, ret_q)
        q.put(0)
        cpu_monitor.start()
        
        t0 = time.time()
        ivfpq_stats.reset()
        D, I = index.search(xq, 100)
        t1 = time.time()
        
        q.put(1)
        cpu_monitor.join()
        cpu_usage = ret_q.get()
        
        recalls = []
        for rank in 1, 10, 100:
            n_ok = (I[:, :rank] == gt[:, :1]).sum()
            print("%.4f" % (n_ok / float(nq)), end=' ')
            recalls.append(n_ok / float(nq))
        print("%8.3f  " % ((t1 - t0) * 1000.0 / nq + index_map_time), end=' \n')
        t = (t1 - t0) * 1000.0 / nq + index_map_time
        # print("%5.2f" % (ivfpq_stats.n_hamming_pass * 100.0 / ivfpq_stats.ncode))
        
        csv_log_data = ['sift500M', param, index_key, 'cpu', t, 1.0/(t/1000.0), recalls[0], recalls[1], recalls[2], cpu_usage]
        csv_log_writer.writerow(csv_log_data)
        
        del index
