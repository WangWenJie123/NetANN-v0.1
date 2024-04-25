import os
import argparse

def readVectorData(source_file, vector_ind: int, vector_dim: int) -> bytes:
    source_file.seek(vector_ind * vector_dim * 4)
    return source_file.read(vector_dim * 4)

# 创建解析器
parser = argparse.ArgumentParser(description='处理传入的参数。')
# 添加参数
parser.add_argument('--source_file', type=str, help='Xb数据文件路径')
parser.add_argument('--target_data', type=str, help='数据重排后Xb数据文件路径(输出)')
parser.add_argument('--vector_dim', type=int, help='向量维数')
parser.add_argument('--nlist', type=int, help='IVF聚类中心数量')
parser.add_argument('--ivf_index_file', type=str, help='IVF索引数据文件路径')
parser.add_argument('--target_index_map', type=str, help='新旧索引映射文件位置(输出)')
parser.add_argument('--target_cluster_nav', type=str, help='标记每个聚类在数据重排后的开始位置(输出)')
# 解析参数
args = parser.parse_args()

source_file = open(args.source_file, 'rb')

# 获取source_file大小, 根据大小以及vector_dim可算出包含向量个数
source_file_size = source_file.seek(0, os.SEEK_END)
vector_num = int(source_file_size / (args.vector_dim * 4))

# 读取聚类中心索引
invlists = []
with open(args.ivf_index_file, 'r') as f:
    invlists = f.readlines()

i=0
for invlist in invlists:
    i = i + 1
    if invlist == '\n':
        print(i)
        invlists.remove(invlist)

assert len(invlists) == args.nlist

# 转换聚类中心索引为int数组格式
invlists = [[int(x) for x in invlist.split(',')] for invlist in invlists]

# 生成每个聚类在数据重组后, 对应的起始位置(单位: 向量个数)
cluster_nav = [0] * args.nlist
for i in range(1, args.nlist):
    cluster_nav[i] = cluster_nav[i - 1] + len(invlists[i - 1])

# 新旧ID映射
index_map = [0] * vector_num

# 从第0个聚类开始, 逐个聚类处理
for i in range(args.nlist):
    for j in range(len(invlists[i])):
        # 新 -> 旧 ID映射
        index_map[cluster_nav[i] + j] = invlists[i][j]
        # 读取向量数据
        vector_data = readVectorData(source_file, invlists[i][j], args.vector_dim)
        # 在指定位置写入向量数据
        with open(args.target_data, 'ab') as f:
            f.write(vector_data)

# 保存新旧ID映射
with open(args.target_index_map, 'wb') as f:
    for ind in index_map:
        f.write(ind.to_bytes(4, byteorder='little'))

# 保存聚类开头地址
with open(args.target_cluster_nav, 'wb') as f:
    for nav in cluster_nav:
        f.write(nav.to_bytes(4, byteorder='little'))


source_file.close()
