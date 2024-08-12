import sys
sys.path.append(r"../Faiss_Test")
import numpy as np
import time
import csv
import queue
import psutil
import threading
import os
import deep1B_text1B_dataset
import subprocess

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

base_dataPath = "/home/wwj/Vector_DB_Acceleration/Vector_Datasets/Remote_Nvme_Vector_Datasets/text1B/local_nvme_text1B_base/base1B.fbin"
groundtruth_dataPath = "/home/wwj/Vector_DB_Acceleration/Vector_Datasets/Remote_Nvme_Vector_Datasets/text1B/groundtruth-public100K.ibin"
query_dataPath = "/home/wwj/Vector_DB_Acceleration/Vector_Datasets/Remote_Nvme_Vector_Datasets/text1B/query100K.fbin"

csv_log_path = "/home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/eva_logs/"
csv_log_title = ["dataset", "nprobe", "index_type", "processor", "search_latency/ms", "throughput/QPS", "R_1", "R_10", "R_100", "cpu_usage/%"]

_HOST = '127.0.0.1'
_PORT = '19530'

# Const names
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'

# Index parameters
_METRIC_TYPE = 'L2'
_INDEX_TYPE = 'IVF_FLAT'

# NLIST_LIST = [512, 1024, 2048, 4096]
NLIST_LIST = [2048, 4096]
NPROB_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

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
    batch_size = 1000
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
    if(sys.argv[1] == 'text10M'):
        xb = deep1B_text1B_dataset.read_fbin(filename=base_dataPath, start_idx=0, chunk_size=10000000)
        xq = deep1B_text1B_dataset.read_fbin(filename=query_dataPath, start_idx=0, chunk_size=1000)
        gt = deep1B_text1B_dataset.read_ibin(filename=groundtruth_dataPath, start_idx=0, chunk_size=1000)
        print("xb_num: {}, xb_dim: {}".format(xb.shape[0], xb.shape[1]))
        print("xq_num: {}, xq_dim: {}".format(xq.shape[0], xq.shape[1]))
        
    if(sys.argv[1] == 'text200M'):
        xb = deep1B_text1B_dataset.read_fbin(filename=base_dataPath, start_idx=0, chunk_size=200000000)
        xq = deep1B_text1B_dataset.read_fbin(filename=query_dataPath, start_idx=0, chunk_size=1000)
        gt = deep1B_text1B_dataset.read_ibin(filename=groundtruth_dataPath, start_idx=0, chunk_size=1000)
        print("xb_num: {}, xb_dim: {}".format(xb.shape[0], xb.shape[1]))
        print("xq_num: {}, xq_dim: {}".format(xq.shape[0], xq.shape[1]))
    
    _NUM = xb.shape[0]
    _DIM = xb.shape[1]
    
    xq_num = xq.shape[0]
    xq_dim = xq.shape[1]
    
    print("***milvus testing start!***")
    # start milvus standalone docker
    start_command = 'bash ./standalone_embed.sh start'
    process = subprocess.run(start_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)

    for _NLIST in NLIST_LIST:
        # create a connection
        create_connection()

        # drop collection if the collection exists
        if has_collection(_COLLECTION_NAME):
            drop_collection(_COLLECTION_NAME)
        if has_collection("text10M"):
            drop_collection("text10M")
        if has_collection("text200M"):
            drop_collection("text200M")

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
            load_vectors_s = time.time()
            load_collection(collection)
            load_vectors_e = time.time()
            load_time = (load_vectors_e - load_vectors_s) * 1000.0
            
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
                
            print("nprobe=%4d; %.3f ms; recalls= %.4f, %.4f, %.4f" % (_NPROBE, avg_search_latency + load_time, recalls[1], recalls[10], recalls[100]))
            print("cpu usage:", cpu_usage)
            
            # write log data to csv file
            csv_log_data = [_COLLECTION_NAME, _NPROBE, 'IVF'+str(_NLIST)+','+'FLAT', processor, avg_search_latency + load_time, 1.0/((avg_search_latency+load_time)/1000.0), recalls[1], recalls[10], recalls[100], cpu_usage]
            csv_log_writer.writerow(csv_log_data)
            
        # release memory
        release_collection(collection)

        # drop collection index
        drop_index(collection)

        # drop collection
        drop_collection(_COLLECTION_NAME)
    
        csv_log_writer.writerow(['', '', '', '', '', '', '', '', '', ''])
    
    # stop Milvus
    stop_command = 'bash ./standalone_embed.sh stop'
    process = subprocess.run(stop_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
        
    # delete Milvus data
    delete_command = 'bash ./standalone_embed.sh delete'
    process = subprocess.run(delete_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
    print("***milvus testing end!***")
    
if __name__ == '__main__':
    main()