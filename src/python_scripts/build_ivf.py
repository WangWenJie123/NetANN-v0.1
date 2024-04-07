import faiss
import numpy as np

def read_fvecs(file_name):
    """
    读取.fvecs文件的函数
    :param file_name: .fvecs文件的路径
    :return: 包含文件中所有向量的NumPy数组
    """
    vectors = []
    with open(file_name, 'rb') as f:
        while True:
            # 读取向量维度（每个维度4字节）
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # 文件结束
            dim = int.from_bytes(dim_bytes, byteorder='little')
            # 读取向量数据（每个数据项4字节）
            vector = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            vectors.append(vector)
    return np.array(vectors, ndmin=2)


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
learn_file_name = '/home/wxr/vector_dataset/sift1M/sift_learn.fvecs'
base_file_name = '/home/wxr/vector_dataset/sift1M/sift_base.fvecs'
output_file_name = '/home/wxr/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/fixed_ivflist/sift1M/sift1M_512_invlists_128dim_indexs_FIXED.csv'
xb_learn = read_fvecs(learn_file_name)
xb_base = read_fvecs(base_file_name)

d = 128
nb = int(xb_base.size / d)
nlist = 512

quantizer = faiss.IndexFlatL2(d)  # the other index
ivfindex = faiss.IndexIVFFlat(quantizer, d, nlist)

ivfindex.train(xb_learn)
ivfindex.add(xb_base)

with open(output_file_name, 'w') as f:
    for i in range(nlist):
        invlist = get_invlist(ivfindex.invlists, i)
        f.write(','.join(map(str, invlist)) + '\n')

# faiss.write_index(ivfindex, "large.index")