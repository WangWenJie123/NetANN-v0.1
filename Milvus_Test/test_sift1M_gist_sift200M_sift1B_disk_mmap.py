import sys
import numpy as np
import time
import csv
import queue
import psutil
import threading
import os

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

# This example shows how to:
#   1. connect to Milvus server
#   2. create a collection
#   3. insert entities
#   4. create index
#   5. search

csv_log_path = '/home/wwj/Vector_DB_Acceleration/Milvus/eva_logs/'
csv_log_title = ["dataset", "nprobe", "index_type", "processor", "search_latency/ms", "throughput/QPS", "R_1", "R_10", "R_100", "cpu_usage/%"]

sift1M_DATASET_PATH = '/home/wwj/Vector_DB_Acceleration/vector_datasets/sift1M/'
sift1M_learn_file = 'sift_learn.fvecs'
sift1M_base_file = 'sift_base.fvecs'
sift1M_query_file = 'sift_query.fvecs'
sift1M_gt_file = 'sift_groundtruth.ivecs'

gist_DATASET_PATH = '/home/wwj/Vector_DB_Acceleration/vector_datasets/gist/'
gist_learn_file = 'gist_learn.fvecs'
gist_base_file = 'gist_base.fvecs'
gist_query_file = 'gist_query.fvecs'
gist_gt_file = 'gist_groundtruth.ivecs'

sift200M_DATASET_PATH = '/home/wwj/Vector_DB_Acceleration/vector_datasets/sift1B/'
sift200M_learn_file = 'bigann_learn.bvecs'
sift200M_base_file = 'bigann_base.bvecs'
sift200M_query_file = 'bigann_query.bvecs'
sift200M_gt_file = 'gnd/idx_1000M.ivecs'

_HOST = '127.0.0.1'
_PORT = '19530'

# Const names
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'

# Index parameters
_METRIC_TYPE = 'L2'
_INDEX_TYPE = 'IVF_FLAT'

NLIST_LIST = [512, 1024, 2048, 4096]
NPROB_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

vbSize = 1000000000

# dataset read functions
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

# dataset mmap functions
def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_base_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:int(1000000000/5), 4:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

# def load_sift1M():
def load_sift1M(sift_learn, sift_base, sift_query, sift_gt):
    print("Loading sift1M...", end='', file=sys.stderr)
    # xt = fvecs_read(sift_learn)
    xb = fvecs_read(sift_base)
    xq = fvecs_read(sift_query)
    gt = ivecs_read(sift_gt)
    print("done", file=sys.stderr)

    return xb, xq, gt

# def mmap_sift1M():
def mmap_sift1M(sift_learn, sift_base, sift_query, sift_gt):
    print("Mapping sift1M...", end='', file=sys.stderr)
    # xt = mmap_fvecs(sift_learn)
    xb = mmap_fvecs(sift_base)
    xq = mmap_fvecs(sift_query)
    gt = ivecs_read(sift_gt)
    print("done", file=sys.stderr)

    return xb, xq, gt

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
   return fv[0:int(vbSize/5), :]

def bvecs_read_query(fname, c_contiguous=True):
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

# Create a Milvus connection
def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=_HOST, port=_PORT)
    print(f"\nList connections:")
    print(connections.list_connections())

# Create a collection named 'dataset name'
def create_collection(name, id_field, vector_field, _DIM):
    field1 = FieldSchema(name=id_field, dtype=DataType.INT64, description="int64", is_primary=True)
    
    if(name == "sift200M"):
        field2 = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, description="float vector", dim=_DIM,
                         is_primary=False)
    else:
        field2 = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, description="float vector", dim=_DIM,
                         is_primary=False)
    
    schema = CollectionSchema(fields=[field1, field2], description="collection description")
    collection = Collection(name=name, data=None, schema=schema, properties={"collection.ttl.seconds": 10000})
    print("\ncollection created:", name)
    return collection


def has_collection(name):
    return utility.has_collection(name)

# Drop a collection in Milvus
def drop_collection(name):
    collection = Collection(name)
    collection.drop()
    print("\nDrop collection: {}".format(name))

# List all collections in Milvus
def list_collections():
    print("\nlist collections:")
    print(utility.list_collections())

def insert(collection, xb_num, xb_dim, xb_vector_data, xq_num, xq_dim, xq_vector_data):
    batch_size = 10000
    for batch in range(0, xb_num, batch_size):
        xb_data = [
            [i for i in range(batch, batch + batch_size)],
            [[xb_vector_data[id][index] for index in range(xb_dim)] for id in range(batch, batch + batch_size)],
        ]
        collection.insert(xb_data)
    print("xb data size is ", len(xb_data[1]))
    
    xq_data = [
        [i for i in range(xq_num)],
        [[xq_vector_data[id][index] for index in range(xq_dim)] for id in range(xq_num)],
    ]
    print("xq data size is ", len(xq_data[1]))
    
    return xq_data[1]

def get_entity_num(collection):
    print("\nThe number of entity:")
    print(collection.num_entities)

def create_index(collection, filed_name, _NLIST):
    index_param = {
        "index_type": _INDEX_TYPE,
        "params": {"nlist": _NLIST},
        "metric_type": _METRIC_TYPE}
    collection.create_index(filed_name, index_param)
    print("\nCreated index:\n{}".format(collection.index().params))

def drop_index(collection):
    collection.drop_index()
    print("\nDrop index sucessfully")

def load_collection(collection):
    collection.load()

def release_collection(collection):
    collection.release()

def search(collection, vector_field, id_field, search_vectors, _NPROBE, _TOPK):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
        "limit": _TOPK,
        "expr": "id_field >= 0"}
    
    search_s_time = time.time()
    results = collection.search(**search_param)
    search_e_time = time.time()
    
    topK_idList = []
    for i, result in enumerate(results):
        tmp_topK_idList = []
        for j, res in enumerate(result):
            tmp_topK_idList.append(res.id)
        
        topK_idList.append(tmp_topK_idList)
    
    # for i, result in enumerate(results):
    #     print("\nSearch result for {}th vector: ".format(i))
    #     for j, res in enumerate(result):
    #         print("Top {}: {}".format(j, res))
    #         print(res.id)
    
    return (search_e_time - search_s_time) * 1000.0 / len(search_vectors), np.asarray(topK_idList, dtype=object)

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

def set_properties(collection):
    collection.set_properties(properties={"collection.ttl.seconds": 10000})

def main():
    # set parameters
    if(len(sys.argv) < 5):
        print("Please give search parameters: dataset_name, type, processor, storage")
        exit(1)
    _COLLECTION_NAME = sys.argv[1]
    stor_type = sys.argv[2]
    processor = sys.argv[3]
    storage = sys.argv[4]
    
    csv_log_file = csv_log_path + stor_type + '_' + _COLLECTION_NAME + '_' + processor + '_' + storage + '.csv'
    if not os.path.exists(csv_log_path):
        fd = open(csv_log_path, 'w')
        fd.close()
        print("create new csv log file")

    print("open csv log")
    csv_log_fd = open(csv_log_file, 'a')
    csv_log_writer = csv.writer(csv_log_fd)
    csv_log_writer.writerow(csv_log_title)
    
    # get vectors from dataset files
    xb = []
    xq = []
    gt = []
    if(sys.argv[1] == 'sift1M'):
        xb, xq, gt = mmap_sift1M(sift_base=sift1M_DATASET_PATH+sift1M_base_file, sift_query=sift1M_DATASET_PATH+sift1M_query_file, sift_learn=sift1M_DATASET_PATH+sift1M_learn_file, sift_gt=sift1M_DATASET_PATH+sift1M_gt_file)
        
    if(sys.argv[1] == 'gist'):
        xb, xq, gt = mmap_sift1M(sift_base=gist_DATASET_PATH+gist_base_file, sift_query=gist_DATASET_PATH+gist_query_file, sift_learn=gist_DATASET_PATH+gist_learn_file, sift_gt=gist_DATASET_PATH+gist_gt_file)
        
    if(sys.argv[1] == 'sift200M'):
        xb = mmap_base_bvecs(sift200M_DATASET_PATH + sift200M_base_file)
        xq = mmap_bvecs(sift200M_DATASET_PATH + sift200M_query_file)
        gt = ivecs_read(sift200M_DATASET_PATH + sift200M_gt_file)
        
        xb = xb.astype(np.float32)
        xq = xq.astype(np.float32)
    
    _NUM = xb.shape[0]
    _DIM = xb.shape[1]
    
    xq_num = xq.shape[0]
    xq_dim = xq.shape[1]

    for _NLIST in NLIST_LIST:
        # create a connection
        create_connection()

        # drop collection if the collection exists
        if has_collection(_COLLECTION_NAME):
            drop_collection(_COLLECTION_NAME)
        if has_collection("sift1M"):
            drop_collection("sift1M")
        if has_collection("gist"):
            drop_collection("gist")
        if has_collection("sift200M"):
            drop_collection("sift200M")

        # create collection
        collection = create_collection(_COLLECTION_NAME, _ID_FIELD_NAME, _VECTOR_FIELD_NAME, _DIM)

        # alter ttl properties of collection level
        set_properties(collection)

        # show collections
        list_collections()

        # insert _NUM vectors with _DIM dimension
        xq_vectors = insert(collection, xb_num=_NUM, xb_dim=_DIM, xb_vector_data=xb, xq_num=xq_num, xq_dim=xq_dim, xq_vector_data=xq)
        # print("xq_vec num: ", len(xq_vectors))

        collection.flush()
        # get the number of entities
        get_entity_num(collection)
    
        # create index
        create_index(collection, _VECTOR_FIELD_NAME, _NLIST)
        
        for _NPROBE in NPROB_LIST:
            # load data to memory
            load_collection(collection)
            
            q = queue.Queue()
            ret_q = queue.Queue()
            cpu_monitor = CPU_Monitor(q, ret_q)
            q.put(0)
            cpu_monitor.start()

            # search
            _TOPK = 100
            avg_search_latency, topK_idList = search(collection, _VECTOR_FIELD_NAME, _ID_FIELD_NAME, xq_vectors, _NPROBE, _TOPK)
            
            q.put(1)
            cpu_monitor.join()
            cpu_usage = ret_q.get()
            
            recalls = {}
            i = 1
            while i <= _TOPK:
                recalls[i] = (topK_idList[:, :i] == gt[:, :1]).sum() / float(xq_num)
                i *= 10
                
            print("nprobe=%4d; %.3f ms; recalls= %.4f, %.4f, %.4f" % (_NPROBE, avg_search_latency, recalls[1], recalls[10], recalls[100]))
            print("cpu usage:", cpu_usage)
            
            # write log data to csv file
            csv_log_data = [_COLLECTION_NAME, _NPROBE, 'IVF'+str(_NLIST)+','+'FLAT', processor, avg_search_latency, 1.0/(avg_search_latency/1000.0), recalls[1], recalls[10], recalls[100], cpu_usage]
            csv_log_writer.writerow(csv_log_data)
            
        # release memory
        release_collection(collection)

        # drop collection index
        drop_index(collection)

        # drop collection
        drop_collection(_COLLECTION_NAME)
    
        csv_log_writer.writerow(['', '', '', '', '', '', '', '', '', ''])
    
if __name__ == '__main__':
    main()