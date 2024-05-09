#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <algorithm>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_queue.h"

#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <unistd.h>
#include <time.h>
#include "p2p_ssd.h"
#include "read_vector_datasets.h"
#include "rapidcsv.h"
#include "kernel_manager.h"

struct taskInfo
{
    int cluster_id;
    int cluster_size;
    int original_index;
    bool operator < (const taskInfo& other) const
    {
        return cluster_size > other.cluster_size;
    }
};

#define BENCHMARK_TEST


#define TEST_SEARCH_VEC_NUM 1
std::string vector_dataset_path = "/home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/nvme_vector_datasets/";
std::string sift1M_xq_vec_fname = "sift_query.fvecs";
std::string gist_xq_vec_fname = "gist_query.fvecs";
std::string sift200M_xq_vec_fname = "bigann_query.bvecs";

#define SEARCH_TOPK_VEC_KERNEL_NUM 16

int main(int argc, char *argv[])
{
    sda::utils::CmdLineParser parser;

    // IVF索引——聚类中心向量数据
    std::string cluster_features_path;
    // IVF索引——每个聚类中含有的向量ID
    std::string cluster_invlists_indexs_path;
    // 测试数据——查询向量数据
    std::string xq_vector_path;
    // 向量数据集——向量数据
    std::string xb_vector_features_path;
    // 数据重组——新旧ID转换
    std::string index_map_path;
    // 数据重组——聚类起始位置
    std::string cluster_nav_path;
    // IVF索引——每个聚类中包含向量个数
    std::string cluster_size_path;

    size_t xq_vec_dim = 0;
    size_t xq_vec_num = 0;

    int cluster_features_fd;
    size_t mmap_cluster_features_len;
    int *mmap_cluster_features = nullptr;
    int xb_vector_features_fd;
    size_t mmap_xb_vector_features_len;


    // time recorder
    std::chrono::high_resolution_clock::time_point centroid_kernelStart, centroid_kernelEnd;
    unsigned long centroid_kernelTime;
    double centroid_dnsduration;

    std::chrono::high_resolution_clock::time_point search_topK_vec_kernelStart, search_topK_vec_kernelEnd;
    unsigned long search_topK_vec_kernelTime;
    double search_topK_vec_dnsduration;

    std::chrono::high_resolution_clock::time_point clusters_vecLoad_Start, clusters_vecLoad_End;
    unsigned long clusters_vecLoad_Time;
    double clusters_vecLoad_dnsduration;

    std::chrono::high_resolution_clock::time_point xbBase_vecLoad_Start, xbBase_vecLoad_End;
    unsigned long xbBase_vecLoad_Time;
    double xbBase_vecLoad_dnsduration;

    std::chrono::high_resolution_clock::time_point distribute_topK_Start, distribute_topK_End;
    unsigned long distribute_topK_Time;
    double distribute_topK_dnsduration;

    double avg_e2e_dnsduration_sum;
    double avg_search_dnsduration_sum;

    // config running args
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "0");
    parser.addSwitch("--data_set", "-s", "vector dataset", "");
    parser.addSwitch("--num_cluster", "-c", "number of cluster", "0");
    parser.addSwitch("--dim", "-m", "dim of vector", "0");
    parser.addSwitch("--nprobe", "-p", "number of cluster used to vector similarity search", "0");
    parser.addSwitch("--topk", "-k", "number of vectors selected", "0");

    parser.parse(argc, argv);

    // read running args
    cluster_features_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("num_cluster") + std::string("_cluster_") + parser.value("dim") + std::string("dim_features.dat");
    if (parser.value("data_set") == std::string("sift1M"))
    {
        xq_vector_path = vector_dataset_path + parser.value("data_set") + std::string("/") + sift1M_xq_vec_fname;
    }
    if (parser.value("data_set") == std::string("gist"))
    {
        xq_vector_path = vector_dataset_path + parser.value("data_set") + std::string("/") + gist_xq_vec_fname;
    }
    if (parser.value("data_set") == std::string("sift200M") || parser.value("data_set") == std::string("sift500M"))
    {
        xq_vector_path = vector_dataset_path + parser.value("data_set") + std::string("/") + sift200M_xq_vec_fname;
    }
    cluster_invlists_indexs_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("num_cluster") + std::string("_invlists_") + parser.value("dim") + std::string("dim_indexs.csv");

    if (parser.value("data_set") == "sift1M")
    {
        xb_vector_features_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("num_cluster") + std::string("_") + parser.value("dim") + std::string("dim_xbVec_features_reorg.dat");
        index_map_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("num_cluster") + std::string("_") + parser.value("dim") + std::string("dim_reorg_indexmap.dat");
        cluster_nav_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("num_cluster") + std::string("_") + parser.value("dim") + std::string("dim_reorg_cluster_nav.dat");
    }
    else if (parser.value("data_set") == "sift200M" || parser.value("data_set") == "sift500M")
    {
        xb_vector_features_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("num_cluster") + std::string("_") + parser.value("dim") + std::string("dim_xbVec_features_reorg.dat");
        index_map_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("num_cluster") + std::string("_") + parser.value("dim") + std::string("dim_reorg_indexmap.dat");
        cluster_nav_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("num_cluster") + std::string("_") + parser.value("dim") + std::string("dim_reorg_cluster_nav.dat");        
    }
    else if (parser.value("data_set") == "gist")
    {
        xb_vector_features_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("num_cluster") + std::string("_") + parser.value("dim") + std::string("dim_xbVec_features_reorg.dat");
        index_map_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("num_cluster") + std::string("_") + parser.value("dim") + std::string("dim_reorg_indexmap.dat");
        cluster_nav_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("num_cluster") + std::string("_") + parser.value("dim") + std::string("dim_reorg_cluster_nav.dat");        
    }
    else
    {
        xb_vector_features_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("dim") + std::string("dim_xbVec_features.dat");
    }

    cluster_size_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("num_cluster") + std::string("_") + parser.value("dim") + std::string("dim_cluster_size.dat");

    int *xq_vec_int = nullptr;

    // Load Query Data (For Benchmark)
    datasetInfo query_dataset_info = load_query_vec_data(parser.value("data_set"), xq_vector_path);
    xq_vec_dim = query_dataset_info.vec_dim;
    xq_vec_num = query_dataset_info.vec_num;
    xq_vec_int = query_dataset_info.vec_data_int;

    int VECTOR_DIM = std::stoi(parser.value("dim"));
    int NUM_CENTROID = std::stoi(parser.value("num_cluster"));
    int NPROBE = std::stoi(parser.value("nprobe"));
    int TOPK = std::stoi(parser.value("topk"));

    std::string vecSearCentroidsBinaryFile = parser.value("xclbin_file");
    int fpgaDevice_index = stoi(parser.value("device_id"));

    if (argc < 3)
    {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    std::cout << "Open the device" << fpgaDevice_index << std::endl;
    auto fpgaDevice = xrt::device(fpgaDevice_index);
    std::cout << "Load the xclbin " << vecSearCentroidsBinaryFile << std::endl;
    auto vecSearCentroids_uuid = fpgaDevice.load_xclbin(vecSearCentroidsBinaryFile);

    // Compute the size of array in bytes
    size_t xqVector_size_in_bytes = VECTOR_DIM * sizeof(int);
    size_t centroid_size_in_bytes = NUM_CENTROID * VECTOR_DIM * sizeof(int);
    size_t outCentroid_id_size_in_bytes = NPROBE * sizeof(int);

    // Creating one vector_search_centroids_top kernel for searching nearist Centroids
    auto fpga_vecSearchCentroid_kernl = xrt::kernel(fpgaDevice, vecSearCentroids_uuid, "vector_search_centroids_top");

    // allocate vector_search_centroids_top kernel buffer on fpga memory
    xrt::bo::flags flag = xrt::bo::flags::p2p;
    std::cout << "Creating FPGA Buffers (M_AXI Interface)" << std::endl;
    auto fpga_XqVector = xrt::bo(fpgaDevice, xqVector_size_in_bytes, fpga_vecSearchCentroid_kernl.group_id(0));
    auto fpga_CentroidsVector = xrt::bo(fpgaDevice, centroid_size_in_bytes, flag, fpga_vecSearchCentroid_kernl.group_id(1));
    auto fpga_oputCentroids_id = xrt::bo(fpgaDevice, outCentroid_id_size_in_bytes, fpga_vecSearchCentroid_kernl.group_id(2));

    // Map the contents of the buffer object into host memory
    auto fpga_XqVector_map = fpga_XqVector.map<int *>();
    auto fpga_CentroidsVector_map = fpga_CentroidsVector.map<int *>();
    auto fpga_oputCentroids_id_map = fpga_oputCentroids_id.map<int *>();

    std::fill(fpga_XqVector_map, fpga_XqVector_map + VECTOR_DIM, 0);
    std::fill(fpga_CentroidsVector_map, fpga_CentroidsVector_map + NUM_CENTROID * VECTOR_DIM, 0);
    std::fill(fpga_oputCentroids_id_map, fpga_oputCentroids_id_map + NPROBE, 0);

    // Benchmark过程中，数据集不会变化，因此IVF索引对应的聚类中心向量数据也不会变化，提前读取后就不用在每个新查询中读取。
    {
        cluster_features_fd = open(cluster_features_path.c_str(), O_RDONLY | O_DIRECT);
        if (cluster_features_fd < 0)
        {
            std::cerr << "ERROR: fopen " << cluster_features_path << " failed: " << std::endl;
            return EXIT_FAILURE;
        }
        mmap_cluster_features_len = lseek(cluster_features_fd, 0, SEEK_END);
        lseek(cluster_features_fd, 0, SEEK_SET);


        clusters_vecLoad_Start = std::chrono::high_resolution_clock::now();

        pread(cluster_features_fd, (void *)fpga_CentroidsVector_map, mmap_cluster_features_len, 0);

        clusters_vecLoad_End = std::chrono::high_resolution_clock::now();

        clusters_vecLoad_Time = std::chrono::duration_cast<std::chrono::nanoseconds>(clusters_vecLoad_End - clusters_vecLoad_Start).count();
        clusters_vecLoad_dnsduration = (double)clusters_vecLoad_Time;
        std::cout << "clusters_vecLoad excution latency:\t" << std::setprecision(3) << std::fixed << clusters_vecLoad_dnsduration << " ns" << std::endl;
        std::cout << std::endl;
    }

    // IVF索引CSV
    // std::cout << "Loading CSV\n";
    // rapidcsv::Document invlist_index_csv(cluster_invlists_indexs_path.c_str() ,rapidcsv::LabelParams(-1, -1));
    // std::cout << "Complete CSV\n";

    // 向量数据文件
    xb_vector_features_fd = open(xb_vector_features_path.c_str(), O_RDONLY | O_DIRECT);
    if (xb_vector_features_fd < 0)
    {
        std::cerr << "ERROR: open " << xb_vector_features_path << " failed!" << std::endl;
        return EXIT_FAILURE;
    }
    mmap_xb_vector_features_len = lseek(xb_vector_features_fd, 0, SEEK_END);
    lseek(xb_vector_features_fd, 0, SEEK_SET);

    // 汇总排序kernel
    auto fpga_distribute_topK_kernel = xrt::kernel(fpgaDevice, vecSearCentroids_uuid, "distribute_topK_top");

    xrt::bo inTopK_disList = xrt::bo(fpgaDevice, NPROBE * TOPK * sizeof(int), fpga_distribute_topK_kernel.group_id(0));
    xrt::bo outTopK_vecId = xrt::bo(fpgaDevice, TOPK * sizeof(int), fpga_distribute_topK_kernel.group_id(1));
    // xrt::bo sort_tmp = xrt::bo(fpgaDevice, NPROBE * TOPK * sizeof(int), fpga_distribute_topK_kernel.group_id(2));
    int *inTopK_disList_map = inTopK_disList.map<int *>(); // Map
    int *outTopK_vecId_map = outTopK_vecId.map<int *>(); // Map
    int inTopK_idList[NPROBE * TOPK] {}; // 不用放入FPGA中计算, 仅用于最终结果展示

    // 聚类起始位置数据读取
    int cluster_nav_fd = open(cluster_nav_path.c_str(), O_RDONLY);
    if (cluster_nav_fd < 0)
    {
        std::cerr << "ERROR: open " << cluster_nav_path << " failed!" << std::endl;
        return EXIT_FAILURE;
    }
    int *cluster_nav_data = new int[NUM_CENTROID];
    read(cluster_nav_fd, (void *)cluster_nav_data, NUM_CENTROID * sizeof(int));

    // 新旧ID映射数据读取
    int index_map_fd = open(index_map_path.c_str(), O_RDONLY);
    if (index_map_fd < 0)
    {
        std::cerr << "ERROR: open " << index_map_path << " failed!" << std::endl;
        return EXIT_FAILURE;
    }
    int *index_map_data = new int[mmap_xb_vector_features_len / sizeof(int) / VECTOR_DIM];
    read(index_map_fd, (void *)index_map_data, mmap_xb_vector_features_len / VECTOR_DIM);

    // 聚类内向量数量数据读取
    int cluster_size_fd = open(cluster_size_path.c_str(), O_RDONLY);
    if (cluster_size_fd < 0)
    {
        std::cerr << "ERROR: open " << cluster_size_path << " failed!" << std::endl;
        return EXIT_FAILURE;
    }
    int *cluster_size_data = new int[NUM_CENTROID];
    read(cluster_size_fd, (void *)cluster_size_data, NUM_CENTROID * sizeof(int));


    // 聚类内距离计算kernel
    searchTopK_KernelManager vecTopkSearch_managers[SEARCH_TOPK_VEC_KERNEL_NUM] = {
        searchTopK_KernelManager("1", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid),
        searchTopK_KernelManager("2", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid),
        searchTopK_KernelManager("3", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid),
        searchTopK_KernelManager("4", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid),
        searchTopK_KernelManager("5", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid),
        searchTopK_KernelManager("6", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid),
        searchTopK_KernelManager("7", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid),
        searchTopK_KernelManager("8", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid),
        searchTopK_KernelManager("9", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid),
        searchTopK_KernelManager("10", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid),
        searchTopK_KernelManager("11", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid),
        searchTopK_KernelManager("12", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid),
        searchTopK_KernelManager("13", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid),
        searchTopK_KernelManager("14", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid),
        searchTopK_KernelManager("15", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid),
        searchTopK_KernelManager("16", TOPK, VECTOR_DIM, parser.value("data_set"), xb_vector_features_fd, cluster_nav_data, inTopK_idList, inTopK_disList_map, cluster_size_data, fpgaDevice, vecSearCentroids_uuid)
    };

    for (size_t i = 0; i < TEST_SEARCH_VEC_NUM; i++)
    {
        memcpy((void *)fpga_XqVector_map, (void *)(xq_vec_int + i * xq_vec_dim), xqVector_size_in_bytes);

        auto run = xrt::run(fpga_vecSearchCentroid_kernl);
        run.set_arg(0, fpga_XqVector);
        run.set_arg(1, fpga_CentroidsVector);
        run.set_arg(2, fpga_oputCentroids_id);
        run.set_arg(3, NUM_CENTROID);
        run.set_arg(4, VECTOR_DIM);
        run.set_arg(5, NPROBE);

        xrt::queue main_queue;

        fpga_XqVector.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Execution of the kernel
        std::cout << "Execution of the fpga_vecSearchCentroid_kernl" << std::endl;
        centroid_kernelStart = std::chrono::high_resolution_clock::now();

        run.start();
        run.wait();

        centroid_kernelEnd = std::chrono::high_resolution_clock::now();
        centroid_kernelTime = std::chrono::duration_cast<std::chrono::nanoseconds>(centroid_kernelEnd - centroid_kernelStart).count();
        centroid_dnsduration = (double)centroid_kernelTime;
        std::cout << "fpga_vecSearchCentroid_kernl excution latency:\t" << std::setprecision(3) << std::fixed << centroid_dnsduration << " ns" << std::endl;
        std::cout << std::endl;

        fpga_oputCentroids_id.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        search_topK_vec_kernelStart = std::chrono::high_resolution_clock::now();

        std::vector<taskInfo> taskInfos(NPROBE);
        for (int j = 0; j < NPROBE; j++)
        {
            taskInfos[j].cluster_id = fpga_oputCentroids_id_map[j];
            taskInfos[j].cluster_size = cluster_size_data[fpga_oputCentroids_id_map[j]];
            taskInfos[j].original_index = j;
        }
        sort(taskInfos.begin(), taskInfos.end());

        // 设置searchTopk Managers Init
        #pragma omp parallel for num_threads(SEARCH_TOPK_VEC_KERNEL_NUM)
        for (int j = 0; j < SEARCH_TOPK_VEC_KERNEL_NUM; j++)
        {
            // 分配kernel上计算的聚类
            std::vector<int> tasks, original_indexs;
            int round = 0;
            while (true)
            {
                int target_index = round * SEARCH_TOPK_VEC_KERNEL_NUM;
                if (round % 2 == 0)
                {
                    target_index += j;
                }
                else
                {
                    target_index += SEARCH_TOPK_VEC_KERNEL_NUM - j - 1;
                }
                if (target_index >= NPROBE)
                {
                    break;
                }
                ++round;
                tasks.push_back(taskInfos[target_index].cluster_id);
                original_indexs.push_back(taskInfos[target_index].original_index);
            }
            // 打印分配结果
            // std::cout << "searchTopK_vec_top kernel " << j << " tasks: \n";
            // for (int k = 0; k < tasks.size(); k++)
            // {
            //     std::cout << tasks[k] << ":\t" << cluster_size_data[tasks[k]] << std::endl;
            // }

            
            vecTopkSearch_managers[j].Init(tasks, original_indexs, xq_vec_int + i * xq_vec_dim);
        }

        // // 设置searchTopk Managers Init
        // for (int j = 0; j < SEARCH_TOPK_VEC_KERNEL_NUM; j++)
        // {
        //     // 分配kernel上计算的聚类
        //     std::vector<int> tasks, original_indexs;
        //     for (int m = j; m < NPROBE; m += SEARCH_TOPK_VEC_KERNEL_NUM)
        //     {
        //         tasks.push_back(fpga_oputCentroids_id_map[m]);
        //         original_indexs.push_back(m);
        //     }
            
        //     // 打印分配结果
        //     std::cout << "searchTopK_vec_top kernel " << j << " tasks: \n";
        //     for (int k = 0; k < tasks.size(); k++)
        //     {
        //         std::cout << tasks[k] << ":\t" << cluster_size_data[tasks[k]] << std::endl;
        //     }

        //     vecTopkSearch_managers[j].Init(tasks, original_indexs, xq_vec_int + i * xq_vec_dim);
        // }

        // 主线程开始轮询Manager是否完成计算, 并控制Load Compute Store流水的推进
        int kernel_complete_cnt = 0;
        bool kernel_complete[SEARCH_TOPK_VEC_KERNEL_NUM] {false};

        while (kernel_complete_cnt != SEARCH_TOPK_VEC_KERNEL_NUM)
        {
            #pragma omp parallel for num_threads(SEARCH_TOPK_VEC_KERNEL_NUM)
            for (int j = 0; j < SEARCH_TOPK_VEC_KERNEL_NUM; j++)
            {
                if (kernel_complete[j])
                    continue;
                vecTopkSearch_managers[j].Next();
                if (vecTopkSearch_managers[j].isEnd())
                {
                    kernel_complete[j] = true;
                    ++kernel_complete_cnt;
                }
            }
        }

        search_topK_vec_kernelEnd = std::chrono::high_resolution_clock::now();
        search_topK_vec_kernelTime = std::chrono::duration_cast<std::chrono::nanoseconds>(search_topK_vec_kernelEnd - search_topK_vec_kernelStart).count();
        search_topK_vec_dnsduration = (double)search_topK_vec_kernelTime;
        std::cout << "search_topK_vec_top excution latency:\t" << std::setprecision(3) << std::fixed << search_topK_vec_dnsduration << " ns" << std::endl;
        std::cout << std::endl;

        // 计算完毕, 将结果同步到FPGA上, 准备下一阶段的计算
        inTopK_disList.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // running distribute_topK kernel to get final topK nearist vectors
        size_t topK_dis_list_size_in_bytes = NPROBE * TOPK * sizeof(int);
        size_t topK_vecId_size_in_bytes = TOPK * sizeof(int);

        // set distribute_topK_top kernel parameters
        xrt::run distribute_topK_run;
        distribute_topK_run = xrt::run(fpga_distribute_topK_kernel);
        distribute_topK_run.set_arg(0, inTopK_disList);
        distribute_topK_run.set_arg(1, outTopK_vecId);
        // distribute_topK_run.set_arg(2, sort_tmp);
        distribute_topK_run.set_arg(2, NPROBE);
        distribute_topK_run.set_arg(3, TOPK);

        // Execution of the kernel
        std::cout << "Execution of the distribute_topK_kernel" << std::endl;
        distribute_topK_Start = std::chrono::high_resolution_clock::now();

        distribute_topK_run.start();
        distribute_topK_run.wait();

        distribute_topK_End = std::chrono::high_resolution_clock::now();
        distribute_topK_Time = std::chrono::duration_cast<std::chrono::nanoseconds>(distribute_topK_End - distribute_topK_Start).count();
        distribute_topK_dnsduration = (double)distribute_topK_Time;
        std::cout << "distribute_topK_kernel excution latency:\t" << std::setprecision(3) << std::fixed << distribute_topK_dnsduration << " ns" << std::endl;
        std::cout << std::endl;

        outTopK_vecId.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        // get final nearist topK vectors
        // std::cout << "Final selected topK nearest neighbor vector..." << std::endl;
        // for(int i = 0; i < TOPK; i++)
        // {
        //     // std::cout << "vector index= " << selected_cluster_topK_vecId[outTopK_vecId_map[i]] << ", ";
        //     std::cout << "vector index= " << outTopK_vecId_map[i] << ", ";

        //     // std::cout << std::dec << "vector dis= " << inTopK_disList_map[outTopK_vecId_map[i]] << ", ";

        //     if(i % 8 == 0 && i != 0)
        //     {
        //         std::cout << std::endl;
        //     }
        // }
        // std::cout << std::endl;
        // std::cout << std::endl;

        // output e2e search latency
        double e2e_latency = centroid_dnsduration + xbBase_vecLoad_dnsduration + search_topK_vec_dnsduration + distribute_topK_dnsduration;
        double search_latency = centroid_dnsduration + search_topK_vec_dnsduration + distribute_topK_dnsduration;

        std::cout << "e2e latency: " << e2e_latency << " ns" << std::endl;
        std::cout << "e2e latency: " << e2e_latency / 1000 << " us" << std::endl;
        std::cout << "e2e latency: " << e2e_latency / 1000000 << " ms" << std::endl;
        std::cout << std::endl;

        std::cout << "search latency: " << search_latency << " ns" << std::endl;
        std::cout << "search latency: " << search_latency / 1000 << " us" << std::endl;
        std::cout << "search latency: " << search_latency / 1000000 << " ms" << std::endl;

        avg_e2e_dnsduration_sum += e2e_latency;
        avg_search_dnsduration_sum += search_latency;

        sleep(1);
    }

    std::cout << std::endl;
    std::cout << "avg e2e latency: " << avg_e2e_dnsduration_sum / TEST_SEARCH_VEC_NUM << " ns" << std::endl;
    std::cout << "avg e2e latency: " << avg_e2e_dnsduration_sum / TEST_SEARCH_VEC_NUM / 1000 << " us" << std::endl;
    std::cout << "avg e2e latency: " << avg_e2e_dnsduration_sum / TEST_SEARCH_VEC_NUM / 1000000 << " ms" << std::endl;
    std::cout << std::endl;

    std::cout << "avg search latency: " << avg_search_dnsduration_sum / TEST_SEARCH_VEC_NUM << " ns" << std::endl;
    std::cout << "avg search latency: " << avg_search_dnsduration_sum / TEST_SEARCH_VEC_NUM / 1000 << " us" << std::endl;
    std::cout << "avg search latency: " << avg_search_dnsduration_sum / TEST_SEARCH_VEC_NUM / 1000000 << " ms" << std::endl;

    close(cluster_features_fd);
    close(xb_vector_features_fd);

    return EXIT_SUCCESS;
}
