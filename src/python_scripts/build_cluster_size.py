import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='处理传入的参数。')
# 添加参数
parser.add_argument('--target_cluster_size', type=str, help='统计每个聚类包含向量数量文件路径(输出)')
parser.add_argument('--nlist', type=int, help='IVF聚类中心数量')
parser.add_argument('--ivf_index_file', type=str, help='IVF索引数据文件路径')

# 解析参数
args = parser.parse_args()

# 读取聚类中心索引
invlists = []
with open(args.ivf_index_file, 'r') as f:
    invlists = f.readlines()

assert len(invlists) == args.nlist

# 转换聚类中心索引为int数组格式
invlists = [[int(x) for x in invlist.split(',')] for invlist in invlists]

# 将每个聚类包含向量数量写入文件
with open(args.target_cluster_size, 'wb') as f:
    for invlist in invlists:
        f.write(len(invlist).to_bytes(4, byteorder='little'))