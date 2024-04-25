import faiss
import numpy as np

def read_bvecs_base(file_path, fseek_enable=False):
    """
    读取.bvecs文件并返回一个numpy数组。(专用于读取base数据, 1B切分为200M)
    
    参数:
    file_path -- .bvecs文件的路径
    
    返回:
    vectors -- 包含所有向量的numpy数组
    """
    vectors = []
    with open(file_path, 'rb') as f:
        for i in range(250000000):
            # 读取向量维度，每个维度用4个字节的整数表示
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # 文件结束
            dim = int.from_bytes(dim_bytes, byteorder='little')
            
            # if fseek_enable == True and i == 0:
            #     f.seek(250000000*dim, 0)
            
            # 读取向量数据，每个特征用1个字节表示
            vector = np.frombuffer(f.read(dim), dtype=np.uint8)
            vectors.append(vector)
    
    return np.array(vectors)


def read_bvecs(file_path):
    """
    读取.bvecs文件并返回一个numpy数组。
    
    参数:
    file_path -- .bvecs文件的路径
    
    返回:
    vectors -- 包含所有向量的numpy数组
    """
    vectors = []
    with open(file_path, 'rb') as f:
        while True:
            # 读取向量维度，每个维度用4个字节的整数表示
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # 文件结束
            dim = int.from_bytes(dim_bytes, byteorder='little')
            
            # 读取向量数据，每个特征用1个字节表示
            vector = np.frombuffer(f.read(dim), dtype=np.uint8)
            vectors.append(vector)
    
    return np.array(vectors)

def get_invlist(invlists, l):
    """ returns the inverted lists content as a pair of (list_ids, list_codes).
    The codes are reshaped to a proper size
    """
    invlists = faiss.downcast_InvertedLists(invlists)
    ls = invlists.list_size(l)
    list_ids = np.zeros(ls, dtype='int64')
    ids = codes = None
    try:
        ids = invlists.get_ids(l)
        if ls > 0:
            faiss.memcpy(faiss.swig_ptr(list_ids), ids, list_ids.nbytes)
        codes = invlists.get_codes(l)
        if invlists.code_size != faiss.InvertedLists.INVALID_CODE_SIZE:
            list_codes = np.zeros((ls, invlists.code_size), dtype='uint8')
        else:
            # it's a BlockInvertedLists
            npb = invlists.n_per_block
            bs = invlists.block_size
            ls_round = (ls + npb - 1) // npb
            list_codes = np.zeros((ls_round, bs // npb, npb), dtype='uint8')
        if ls > 0:
            faiss.memcpy(faiss.swig_ptr(list_codes), codes, list_codes.nbytes)
    finally:
        if ids is not None:
            invlists.release_ids(l, ids)
        if codes is not None:
            invlists.release_codes(l, codes)
    return list_ids


def get_invlist_sizes(invlists):
    """ return the array of sizes of the inverted lists """
    return np.array([
        invlists.list_size(i)
        for i in range(invlists.nlist)
    ], dtype='int64')

# 使用示例
learn_file_name = '/home/wwj/Vector_DB_Acceleration/SPTAG/wwj_test/remote_nvme_vector_datasets/sift1B/bigann_learn.bvecs'
base_file_name = '/home/wwj/Vector_DB_Acceleration/SPTAG/wwj_test/remote_nvme_vector_datasets/sift1B/bigann_base.bvecs'
output_file_name = '/home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/NetANN_Vector_Datasets/sift1B/sift1B_512_invlists_128dim_indexs_FIXED.csv'
    
# add batch vectors
# 4 * 250000000 = 1B
for i in range(4):
    xb_learn_second = read_bvecs(learn_file_name)
    xb_base_second = read_bvecs_base(base_file_name, True)

    d_second = 128
    nb_second = int(xb_base_second.size / d_second)
    print(nb_second)
    nlist_second = 512

    quantizer_second = faiss.IndexFlatL2(d_second)  # the other index
    ivfindex_second = faiss.IndexIVFFlat(quantizer_second, d_second, nlist_second)

    faiss.omp_set_num_threads(64)
    ivfindex_second.train(xb_learn_second)
    ivfindex_second.add(xb_base_second)

    with open(output_file_name, 'a') as f:
        for i in range(nlist_second):
            invlist_second = get_invlist(ivfindex_second.invlists, i)
            f.write(','.join(map(str, invlist_second)) + '\n')