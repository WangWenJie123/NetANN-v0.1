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

# 使用示例
base_file_name = '/home/wwj/Vector_DB_Acceleration/SPTAG/wwj_test/remote_nvme_vector_datasets/sift1M/sift_base.fvecs'
query_file_name = '/home/wwj/Vector_DB_Acceleration/SPTAG/wwj_test/remote_nvme_vector_datasets/sift1M/sift_query.fvecs'

xb_base = read_fvecs(base_file_name)

d = 128
xb_num = int(xb_base.size / d)
sparse_vectors_num = 0

vec_sparsity = 0
for i in range(xb_num):
    for j in range(d):
        if xb_base[i][j] == 0.0:
            vec_sparsity += 1
    vec_sparsity = vec_sparsity / d * 100
    if vec_sparsity > 50.0:
        sparse_vectors_num += 1
        # print("vec[{index}] sparsity: {sparsity}%".format(index = i, sparsity = vec_sparsity))
    vec_sparsity = 0

print("sparse vectors in database: ", sparse_vectors_num / xb_num * 100, "%")