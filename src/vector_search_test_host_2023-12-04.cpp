#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <sys/mman.h>
#include <thread>

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

// #define WRITE_TEST_DATA
// #define DEBUG
#define BENCHMARK_TEST

#ifdef DEBUG
#define VECTOR_DIM 128
#define NUM_CENTROID 512
#define NPROBE 256
#define NUMCU 256
#endif

#ifdef DEBUG
std::string out_CentroidsVector_file_path = "/home/wwj/nvme_ssd_test_dir/vector_p2p_test/CentroidsVector.dat";
#endif

#ifdef BENCHMARK_TEST
#define TEST_SEARCH_VEC_NUM 1
std::string vector_dataset_path = "/home/wwj/nvme_ssd_test_dir/";
std::string sift1M_xq_vec_fname = "sift_query.fvecs";
std::string gist_xq_vec_fname = "gist_query.fvecs";
#endif

int main(int argc, char* argv[])
{
    sda::utils::CmdLineParser parser;

#ifdef BENCHMARK_TEST
    std::string cluster_features_path;
    std::string cluster_invlists_indexs_path;
    std::string xq_vector_path;
    std::string xb_vector_features_path;

    size_t xq_vec_dim = 0;
    size_t xq_vec_num = 0;
#endif

#ifdef DEBUG
    std::ofstream fout;
    FILE* nvmeFd;
#endif

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

#ifdef BENCHMARK_TEST
    parser.addSwitch("--data_set", "-s", "vector dataset", "");
    parser.addSwitch("--num_cluster", "-c", "number of cluster", "0");
    parser.addSwitch("--dim", "-m", "dim of vector", "0");
    parser.addSwitch("--nprobe", "-p", "number of cluster used to vector similarity search", "0");
    parser.addSwitch("--topk", "-k", "number of vectors selected", "0");
#endif

    parser.parse(argc, argv);

    // read running args
#ifdef BENCHMARK_TEST
    cluster_features_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("num_cluster") + std::string("_cluster_") + parser.value("dim") + std::string("dim_features.dat");
    if(parser.value("data_set") == std::string("sift1M"))
    {   xq_vector_path = vector_dataset_path + parser.value("data_set") + std::string("/") + sift1M_xq_vec_fname;
    }
    if(parser.value("data_set") == std::string("gist"))
    {   xq_vector_path = vector_dataset_path + parser.value("data_set") + std::string("/") + gist_xq_vec_fname;
    }
    cluster_invlists_indexs_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("num_cluster") + std::string("_invlists_") + parser.value("dim") + std::string("dim_indexs.csv");
    xb_vector_features_path = vector_dataset_path + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("/") + parser.value("data_set") + std::string("_") + parser.value("num_cluster") + std::string("_") + parser.value("dim") + std::string("dim_xbVec_features.dat");

    int* xq_vec_int = nullptr;
    float* xq_vec = load_fvecs_data(xq_vector_path.c_str(), &xq_vec_dim, &xq_vec_num);
    xq_vec_int = (int*)malloc(sizeof(int) * xq_vec_num * xq_vec_dim);
    for(size_t i = 0; i < xq_vec_num; i++)
    {
        for(size_t j = 0; j < xq_vec_dim; j++)
        {
            xq_vec_int[i * xq_vec_dim + j] = int(xq_vec[i * xq_vec_dim + j]);
        }
    }

    int VECTOR_DIM = std::stoi(parser.value("dim"));
    int NUM_CENTROID = std::stoi(parser.value("num_cluster"));
    int NPROBE = std::stoi(parser.value("nprobe"));
    int TOPK = std::stoi(parser.value("topk"));
#endif

    std::string vecSearCentroidsBinaryFile = parser.value("xclbin_file");
    int fpgaDevice_index = stoi(parser.value("device_id"));

    if(argc < 3)
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

    // Creating multi-vecTopkSearch kernels for asymmetrical searching in Centroids
    // xrt::kernel fpga_vecTopkSearch_kernl[NPROBE];
    // for(int i = 0; i < NPROBE; i++)
    // {
    //     fpga_vecTopkSearch_kernl[i] = xrt::kernel(fpgaDevice, vecSearCentroids_uuid, "search_topK_vec_top");
    // }
    auto fpga_vecTopkSearch_kernl = xrt::kernel(fpgaDevice, vecSearCentroids_uuid, "search_topK_vec_top");

    // Creating one distribute_topK kernel for getting final topK nearist vectors
    auto fpga_distribute_topK_kernel = xrt::kernel(fpgaDevice, vecSearCentroids_uuid, "distribute_topK_top");
    
    // allocate vector_search_centroids_top kernel buffer on fpga memory
    xrt::bo::flags flag = xrt::bo::flags::p2p;
    std::cout << "Creating FPGA Buffers (M_AXI Interface)" << std::endl;
    auto fpga_XqVector = xrt::bo(fpgaDevice, xqVector_size_in_bytes, fpga_vecSearchCentroid_kernl.group_id(0));
    auto fpga_CentroidsVector = xrt::bo(fpgaDevice, centroid_size_in_bytes, flag, fpga_vecSearchCentroid_kernl.group_id(1));
    auto fpga_oputCentroids_id = xrt::bo(fpgaDevice, outCentroid_id_size_in_bytes, fpga_vecSearchCentroid_kernl.group_id(2));

    // Map the contents of the buffer object into host memory
    auto fpga_XqVector_map = fpga_XqVector.map<int*>();
    auto fpga_CentroidsVector_map = fpga_CentroidsVector.map<int*>();
    auto fpga_oputCentroids_id_map = fpga_oputCentroids_id.map<int*>();

    std::fill(fpga_XqVector_map, fpga_XqVector_map + VECTOR_DIM, 0);
    std::fill(fpga_CentroidsVector_map, fpga_CentroidsVector_map + NUM_CENTROID * VECTOR_DIM, 0);
    std::fill(fpga_oputCentroids_id_map, fpga_oputCentroids_id_map + NPROBE, 0);

#ifdef BENCHMARK_TEST
    for(size_t i = 0; i < TEST_SEARCH_VEC_NUM; i++)
    {
    memcpy((void*)fpga_XqVector_map, (void*)(xq_vec_int + i * xq_vec_dim), xqVector_size_in_bytes);
    // std::cout << "Xq vector features..." << std::endl;
    // for(size_t i = 0; i < xq_vec_dim; i++)
    // {
    //     std::cout << fpga_XqVector_map[i] << ", ";
    //     if(i % 8 == 0 && i != 0)
    //     {
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;
#endif

#ifdef DEBUG
    for(int i = 0; i < VECTOR_DIM; i++)
    {
        fpga_XqVector_map[i] = int(i * 4);
    }
#endif

#if defined(WRITE_TEST_DATA) && defined(DEBUG)
    // Create the test vector data and write CentroidsVector to nvme disk file
    int tem_data = 0;
    fout.open(out_CentroidsVector_file_path, std::ios::app | std::ios::binary);
    if(fout.is_open() == false)
    {
        std::cout << "out file open failure!" << std::endl;
    }
    for(int i = 0; i < NUM_CENTROID; i++)
    {
        for(int j = 0; j < VECTOR_DIM; j++)
        {
            if(i % 7 == 0)
            {
                // fpga_CentroidsVector_map[i * VECTOR_DIM + j] = j;
                fout.write((const char*)&j, sizeof(int));
            }
            else
            {
                // fpga_CentroidsVector_map[i * VECTOR_DIM + j] = j * 4;
                tem_data = j * 4;
                fout.write((const char*)&tem_data, sizeof(int));
            }
        }
    }
    fout.close();
#endif

    auto run = xrt::run(fpga_vecSearchCentroid_kernl);
    run.set_arg(0, fpga_XqVector);
    run.set_arg(1, fpga_CentroidsVector);
    run.set_arg(2, fpga_oputCentroids_id);
    run.set_arg(3, NUM_CENTROID);
    run.set_arg(4, VECTOR_DIM);
    run.set_arg(5, NPROBE);

    xrt::queue main_queue;

    // read CentroidsVector from nvme disk file
#ifdef DEBUG
    nvmeFd = fopen(out_CentroidsVector_file_path.c_str(), "rb");
    if(nvmeFd == NULL) {
        std::cerr << "ERROR: fopen " << out_CentroidsVector_file_path << "failed: " << std::endl;
        return EXIT_FAILURE;
    }
    if(fread((void*)fpga_CentroidsVector_map, sizeof(int), NUM_CENTROID * VECTOR_DIM, nvmeFd) <= 0) {
        std::cerr << "ERR: fread failed: "
                  << " error: " << strerror(errno) << std::endl;
        exit(EXIT_FAILURE);
    }
    fclose(nvmeFd);
#endif

#ifdef BENCHMARK_TEST
    int cluster_features_fd;
    cluster_features_fd = open(cluster_features_path.c_str(), O_RDONLY | O_DIRECT);
    if(cluster_features_fd < 0) {
        std::cerr << "ERROR: fopen " << cluster_features_path << " failed: " << std::endl;
        return EXIT_FAILURE;
    }
    int mmap_cluster_features_len = lseek(cluster_features_fd, 0, SEEK_END);
    int* mmap_cluster_features = (int*)mmap(NULL, mmap_cluster_features_len, PROT_READ, MAP_SHARED, cluster_features_fd, 0);

    clusters_vecLoad_Start = std::chrono::high_resolution_clock::now();
    memcpy((void*)fpga_CentroidsVector_map, (void*)mmap_cluster_features, NUM_CENTROID * VECTOR_DIM * sizeof(int));
    clusters_vecLoad_End = std::chrono::high_resolution_clock::now();
    clusters_vecLoad_Time = std::chrono::duration_cast<std::chrono::nanoseconds>(clusters_vecLoad_End - clusters_vecLoad_Start).count();
    clusters_vecLoad_dnsduration = (double)clusters_vecLoad_Time;
    std::cout << "clusters_vecLoad excution latency:\t" << std::setprecision(3) << std::fixed << clusters_vecLoad_dnsduration << " ns" << std::endl;
    std::cout << std::endl;
    close(cluster_features_fd);
    munmap(mmap_cluster_features, mmap_cluster_features_len);
#endif

    auto fpga_XqVector_event = main_queue.enqueue([&fpga_XqVector] { fpga_XqVector.sync(XCL_BO_SYNC_BO_TO_DEVICE); });

    // Execution of the kernel
    std::cout << "Execution of the fpga_vecSearchCentroid_kernl" << std::endl;
    centroid_kernelStart = std::chrono::high_resolution_clock::now();
    
    main_queue.enqueue([&run] {
        run.start();
        run.wait();
    });
    
    centroid_kernelEnd = std::chrono::high_resolution_clock::now();
    centroid_kernelTime = std::chrono::duration_cast<std::chrono::nanoseconds>(centroid_kernelEnd - centroid_kernelStart).count();
    centroid_dnsduration = (double)centroid_kernelTime;
    std::cout << "fpga_vecSearchCentroid_kernl excution latency:\t" << std::setprecision(3) << std::fixed << centroid_dnsduration << " ns" << std::endl;
    std::cout << std::endl;

    auto fpga_oputCentroids_id_event = main_queue.enqueue([&fpga_oputCentroids_id] { fpga_oputCentroids_id.sync(XCL_BO_SYNC_BO_FROM_DEVICE); });
    fpga_oputCentroids_id_event.wait();

    // Output the result
    std::cout << "The fpga_vecSearchCentroid_kernl compute result:" << std::endl;
    for(int i = 0; i < NPROBE; i++)
    {
        std::cout << fpga_oputCentroids_id_map[i] << ", ";
        if((i + 1) % 8 == 0 && i != 0)
        {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    // get selected invlists index
    rapidcsv::Document invlists_indx_csv(cluster_invlists_indexs_path.c_str(), rapidcsv::LabelParams(-1, -1));
    std::vector<int> selected_clusters_invlists_item;
    std::vector<std::vector<int>> selected_clusters_invlists_sum;
    for(int i = 0; i < NPROBE; i++)
    {
        selected_clusters_invlists_item = invlists_indx_csv.GetRow<int>(fpga_oputCentroids_id_map[i]);
        selected_clusters_invlists_sum.push_back(selected_clusters_invlists_item);
    }
    
    // output some result for validation
    // std::cout << "Some selected clusters invlists..." << std::endl;
    // int count = 0;
    // std::vector<int> selected_clusters_invlists_item_test = selected_clusters_invlists_sum.at(0);
    // for(auto item = selected_clusters_invlists_item_test.begin(); item != selected_clusters_invlists_item_test.end(); ++item)
    // {
    //     count++;
    //     std::cout << std::dec << *item << ",";
    //     if(count % 8 == 0 && count != 0)
    //     {
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;

    // start search topK vectors
#ifdef BENCHMARK_TEST
    // allocate search_topK_vec_top kernel buffer on fpga memory
    size_t xb_vectors_num[NPROBE];
    std::vector<int> invlists_items[NPROBE];
    size_t xb_vectors_size_in_bytes[NPROBE];
    size_t outTopKVectors_id_size_in_bytes = TOPK * sizeof(int);
    size_t outTopKVectors_dis_size_in_bytes = TOPK * sizeof(int);

    for(int i = 0; i < NPROBE; i++)
    {
        xb_vectors_num[i] = selected_clusters_invlists_sum[i].size();
        invlists_items[i] = selected_clusters_invlists_sum.at(i);
        xb_vectors_size_in_bytes[i] = xb_vectors_num[i] * VECTOR_DIM * sizeof(int);
    }
    
    xrt::bo search_topK_vec_XqVector[NPROBE];
    xrt::bo search_topK_vec_xbVectors[NPROBE];
    xrt::bo search_topK_vec_out_topK_Vectors[NPROBE];
    xrt::bo search_topK_vec_out_topK_dis[NPROBE];

    for(int i = 0; i < NPROBE; i++)
    {
        search_topK_vec_XqVector[i] = xrt::bo(fpgaDevice, xqVector_size_in_bytes, fpga_vecTopkSearch_kernl.group_id(0));
        search_topK_vec_xbVectors[i] = xrt::bo(fpgaDevice, xb_vectors_size_in_bytes[i], flag, fpga_vecTopkSearch_kernl.group_id(1));
        search_topK_vec_out_topK_Vectors[i] = xrt::bo(fpgaDevice, outTopKVectors_id_size_in_bytes, fpga_vecTopkSearch_kernl.group_id(2));
        search_topK_vec_out_topK_dis[i] = xrt::bo(fpgaDevice, outTopKVectors_dis_size_in_bytes, fpga_vecTopkSearch_kernl.group_id(0));
    }

    // Map the contents of the buffer object into host memory
    int* search_topK_vec_XqVector_map[NPROBE] = {nullptr};
    int* search_topK_vec_xbVectors_map[NPROBE] = {nullptr};
    int* search_topK_vec_out_topK_Vectors_map[NPROBE] = {nullptr};
    int* search_topK_vec_out_topK_dis_map[NPROBE] = {nullptr};
    for(int i = 0; i < NPROBE; i++)
    {
        search_topK_vec_XqVector_map[i] = search_topK_vec_XqVector[i].map<int*>();
        search_topK_vec_xbVectors_map[i] = search_topK_vec_xbVectors[i].map<int*>();
        search_topK_vec_out_topK_Vectors_map[i] = search_topK_vec_out_topK_Vectors[i].map<int*>();
        search_topK_vec_out_topK_dis_map[i] = search_topK_vec_out_topK_dis[i].map<int*>();

        std::fill(search_topK_vec_XqVector_map[i], search_topK_vec_XqVector_map[i] + VECTOR_DIM, 0);
        std::fill(search_topK_vec_xbVectors_map[i], search_topK_vec_xbVectors_map[i] + xb_vectors_num[i] * VECTOR_DIM, 0);
        std::fill(search_topK_vec_out_topK_Vectors_map[i], search_topK_vec_out_topK_Vectors_map[i] + TOPK, 0);
        std::fill(search_topK_vec_out_topK_dis_map[i], search_topK_vec_out_topK_dis_map[i] + TOPK, 0);

        memcpy((void*)search_topK_vec_XqVector_map[i], (void*)xq_vec_int, xqVector_size_in_bytes);
    }

    // set search_topK_vec_top kernel parameters
    xrt::run vecTopkSearch_run[NPROBE];
    for(int i = 0; i < NPROBE; i++)
    {
        vecTopkSearch_run[i] = xrt::run(fpga_vecTopkSearch_kernl);
        vecTopkSearch_run[i].set_arg(0, search_topK_vec_XqVector[i]);
        vecTopkSearch_run[i].set_arg(1, search_topK_vec_xbVectors[i]);
        vecTopkSearch_run[i].set_arg(2, search_topK_vec_out_topK_Vectors[i]);
        vecTopkSearch_run[i].set_arg(3, xb_vectors_num[i]);
        vecTopkSearch_run[i].set_arg(4, VECTOR_DIM);
        vecTopkSearch_run[i].set_arg(5, TOPK);
        vecTopkSearch_run[i].set_arg(6, search_topK_vec_out_topK_dis[i]);
    }
    
    xrt::queue vecTopkSearch_main_queue[NPROBE];

    // p2p read xb base vectors from NVMe SSDs (mmap)
    int xb_vector_features_fd;
    xb_vector_features_fd = open(xb_vector_features_path.c_str(), O_RDONLY | O_DIRECT);
    if(xb_vector_features_fd < 0)
    {
        std::cerr << "ERROR: open " << xb_vector_features_path << " failed!" << std::endl;
        return EXIT_FAILURE;
    }
    size_t mmap_xb_vector_features_len = lseek(xb_vector_features_fd, 0, SEEK_END);
    int* mmap_xb_vector_features = (int*)mmap(NULL, mmap_xb_vector_features_len, PROT_READ, MAP_SHARED, xb_vector_features_fd, 0);
    
    xbBase_vecLoad_Start = std::chrono::high_resolution_clock::now();
    // #pragma omp parallel for reduction(+:sum)
    // #pragma omp parallel
    for(int cluster_index = 0; cluster_index < NPROBE; cluster_index++)
    {
        #pragma omp for
        for(size_t i = 0; i < xb_vectors_num[cluster_index]; i++)
        {
            memcpy((void*)(search_topK_vec_xbVectors_map[cluster_index] + i * VECTOR_DIM), (void*)(mmap_xb_vector_features + invlists_items[cluster_index][i] * VECTOR_DIM), VECTOR_DIM * sizeof(int));
        }
    }
    xbBase_vecLoad_End = std::chrono::high_resolution_clock::now();
    xbBase_vecLoad_Time = std::chrono::duration_cast<std::chrono::nanoseconds>(xbBase_vecLoad_End - xbBase_vecLoad_Start).count();
    xbBase_vecLoad_dnsduration = (double)xbBase_vecLoad_Time;
    std::cout << "xbBase_vecLoad excution latency:\t" << std::setprecision(3) << std::fixed << xbBase_vecLoad_dnsduration << " ns" << std::endl;
    std::cout << std::endl;

    close(xb_vector_features_fd);
    munmap(mmap_xb_vector_features, mmap_xb_vector_features_len);

    // run search_topK_vec_top kernel
    xrt::queue::event search_topK_vec_XqVector_event[NPROBE];
    #pragma omp parallel
    for(int i = 0; i < NPROBE; i++)
    {
        search_topK_vec_XqVector_event[i] = vecTopkSearch_main_queue[i].enqueue([&search_topK_vec_XqVector, i] { search_topK_vec_XqVector[i].sync(XCL_BO_SYNC_BO_TO_DEVICE); });
        // search_topK_vec_XqVector[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }    

    std::cout << "Execution of the search_topK_vec_top" << std::endl;
    search_topK_vec_kernelStart = std::chrono::high_resolution_clock::now();
    
    // #pragma omp parallel
    for(int i = 0; i < NPROBE; i++)
    {
        vecTopkSearch_main_queue[i].enqueue([&vecTopkSearch_run, i] {
        vecTopkSearch_run[i].start();
        vecTopkSearch_run[i].wait();
        });
        // vecTopkSearch_run[i].start();
    }

    // for(int i = 0; i < NPROBE; i++)
    // {
    //     vecTopkSearch_main_queue[i].enqueue([&vecTopkSearch_run, i] {
    //     vecTopkSearch_run[i].wait();
    //     });
    //     // vecTopkSearch_run[i].wait();
    // }
    
    search_topK_vec_kernelEnd = std::chrono::high_resolution_clock::now();
    search_topK_vec_kernelTime = std::chrono::duration_cast<std::chrono::nanoseconds>(search_topK_vec_kernelEnd - search_topK_vec_kernelStart).count();
    search_topK_vec_dnsduration = (double)search_topK_vec_kernelTime;
    std::cout << "search_topK_vec_top excution latency:\t" << std::setprecision(3) << std::fixed << search_topK_vec_dnsduration << " ns" << std::endl;
    std::cout << std::endl;

    xrt::queue::event search_topK_vec_out_topK_Vectors_event[NPROBE];
    xrt::queue::event search_topK_vec_out_topK_Dis_event[NPROBE];
    #pragma omp parallel
    for(int i = 0; i < NPROBE; i++)
    {
        search_topK_vec_out_topK_Vectors_event[i] = vecTopkSearch_main_queue[i].enqueue([&search_topK_vec_out_topK_Vectors, i] { search_topK_vec_out_topK_Vectors[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE); });
        search_topK_vec_out_topK_Vectors_event[i].wait();
        // search_topK_vec_out_topK_Vectors[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        search_topK_vec_out_topK_Dis_event[i] = vecTopkSearch_main_queue[i].enqueue([&search_topK_vec_out_topK_dis, i] { search_topK_vec_out_topK_dis[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE); });
        search_topK_vec_out_topK_Dis_event[i].wait();
        // search_topK_vec_out_topK_dis[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }

    // get topK nearest neighbor vector index
    // int selected_cluster_topK_vecId[NPROBE * TOPK];
    // std::cout << "Selected topK nearest neighbor vector..." << std::endl;
    // for(int cluster_index = 0; cluster_index < NPROBE; cluster_index++)
    // {
    //     for (int i = 0; i < TOPK; i++)
    //     {
    //         std::cout << "vector index= " << invlists_items[cluster_index][search_topK_vec_out_topK_Vectors_map[cluster_index][i]] << ", ";
    //         selected_cluster_topK_vecId[cluster_index * TOPK + i] = invlists_items[cluster_index][search_topK_vec_out_topK_Vectors_map[cluster_index][i]];

    //         std::cout << std::dec << "vector dis= " << search_topK_vec_out_topK_dis_map[cluster_index][i] << ", ";

    //         if(i % 8 == 0 && i != 0)
    //         {
    //             std::cout << std::endl;
    //         }
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // running distribute_topK kernel to get final topK nearist vectors
    size_t topK_dis_list_size_in_bytes = NPROBE * TOPK * sizeof(int);
    size_t topK_vecId_size_in_bytes = TOPK * sizeof(int);

    xrt::bo inTopK_disList;
    xrt::bo outTopK_vecId;

    inTopK_disList = xrt::bo(fpgaDevice, topK_dis_list_size_in_bytes, fpga_distribute_topK_kernel.group_id(0));
    outTopK_vecId = xrt::bo(fpgaDevice, topK_vecId_size_in_bytes, fpga_distribute_topK_kernel.group_id(1));

    // Map the contents of the buffer object into host memory
    auto inTopK_disList_map = inTopK_disList.map<int*>();
    auto outTopK_vecId_map = outTopK_vecId.map<int*>();

    std::fill(inTopK_disList_map, inTopK_disList_map + NPROBE * TOPK, 0);
    std::fill(outTopK_vecId_map, outTopK_vecId_map + TOPK, 0);

    #pragma omp parallel
    for(int i = 0; i < NPROBE; i++)
    {
        memcpy((void*)(inTopK_disList_map + i * TOPK), (void*)(search_topK_vec_out_topK_dis_map[i]), TOPK * sizeof(int));
    }
    
    // set distribute_topK_top kernel parameters
    xrt::run distribute_topK_run;
    distribute_topK_run = xrt::run(fpga_distribute_topK_kernel);
    distribute_topK_run.set_arg(0, inTopK_disList);
    distribute_topK_run.set_arg(1, outTopK_vecId);
    distribute_topK_run.set_arg(2, NPROBE);
    distribute_topK_run.set_arg(3, TOPK);

    // running distribute_topK kernel
    xrt::queue distribute_topK_main_queue;
    auto distribute_topK_inDis_event = distribute_topK_main_queue.enqueue([&inTopK_disList] { inTopK_disList.sync(XCL_BO_SYNC_BO_TO_DEVICE); });

    // Execution of the kernel
    std::cout << "Execution of the distribute_topK_kernel" << std::endl;
    distribute_topK_Start = std::chrono::high_resolution_clock::now();
    
    distribute_topK_main_queue.enqueue([&distribute_topK_run] {
        distribute_topK_run.start();
        distribute_topK_run.wait();
    });

    distribute_topK_End = std::chrono::high_resolution_clock::now();
    distribute_topK_Time = std::chrono::duration_cast<std::chrono::nanoseconds>(distribute_topK_End - distribute_topK_Start).count();
    distribute_topK_dnsduration = (double)distribute_topK_Time;
    std::cout << "distribute_topK_kernel excution latency:\t" << std::setprecision(3) << std::fixed << distribute_topK_dnsduration << " ns" << std::endl;
    std::cout << std::endl;

    auto distribute_topK_outId_event = distribute_topK_main_queue.enqueue([&outTopK_vecId] { outTopK_vecId.sync(XCL_BO_SYNC_BO_FROM_DEVICE); });
    distribute_topK_outId_event.wait();

    // get final nearist topK vectors
    // std::cout << "Final selected topK nearest neighbor vector..." << std::endl;
    // for(int i = 0; i < TOPK; i++)
    // {
    //     std::cout << "vector index= " << selected_cluster_topK_vecId[outTopK_vecId_map[i]] << ", ";

    //     std::cout << std::dec << "vector dis= " << inTopK_disList_map[outTopK_vecId_map[i]] << ", ";

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
#endif
    
    return EXIT_SUCCESS;
}
