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
#include "csv_log.h"

// define read type
#define SEQ_READ    1
#define RANDOM_READ 2

// define log file path
std::string csv_log_path = "/home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/eva_logs/";

class CSV_Test_Disk_IO_LOGer
{
    public:
        CSV_Test_Disk_IO_LOGer(const char* logger_file);

        ~CSV_Test_Disk_IO_LOGer();

    public:
        void write_to_csv(const char* datasets, const char* read_type, const char* vec_data_type, int vec_dim, int read_vec_num, double search_latency, double bandwidth);
    
    private:
        std::ofstream file;
};

CSV_Test_Disk_IO_LOGer::CSV_Test_Disk_IO_LOGer(const char* logger_file)
{
    file = std::ofstream(logger_file, std::ios::app);
    file << "datasets" << ',' << "read_type" << ',' << "vec_data_type" << ',' << "vec_dim" << ',' << "read_vec_num" << ',' << "search_latency/ms" << "bandwidth/MB" << std::endl;
}

CSV_Test_Disk_IO_LOGer::~CSV_Test_Disk_IO_LOGer()
{
    file.close();
}

void CSV_Test_Disk_IO_LOGer::write_to_csv(const char* datasets, const char* read_type, const char* vec_data_type, int vec_dim, int read_vec_num, double search_latency, double bandwidth)
{
    file << datasets << ',' << read_type << ',' << vec_data_type << ',' << std::to_string(vec_dim) << ',' << std::to_string(read_vec_num) << ',' << std::to_string(search_latency) << ',' << std::to_string(bandwidth) << std::endl;
}

int get_random_vecid(int min, int max)
{
    srand((unsigned)time(NULL));
    return rand() % (max - min + 1) + min;
}

uint8_t test_cop_cpu_rssd(uint8_t type, const char* fname)
{
    int nvme_fd  = -1;

    std::chrono::high_resolution_clock::time_point readVec_start, readVec_end, readVec_start_1, readVec_end_1;

    // open nvme ssd vector file
    nvme_fd = open(fname, O_RDONLY | O_DIRECT);
    if(nvme_fd < 0)
    {
        std::cerr << "ERROR: open " << fname << " failed!" << std::endl;
        return EXIT_FAILURE;
    }

    // allocate buffer on host memory
    size_t buffer_size_bytes = 10000 * 128 * sizeof(int);
    int* vec_buffer = (int*)malloc(buffer_size_bytes);
    std::fill(vec_buffer, vec_buffer + 10000 * 128, 0);

    switch (type)
    {
    case SEQ_READ: { // seq read
        std::string log_fname = csv_log_path + std::string("test_cop_cpu_rssd_Seq_Read.csv");
        CSV_Test_Disk_IO_LOGer eva_logger(log_fname.c_str());

        for(int vec_num = 1; vec_num <= 10000; vec_num *= 2)
        {
            size_t readVecSize_bytes = vec_num * 128 * sizeof(int);
            readVec_start = std::chrono::high_resolution_clock::now();
            if(pread(nvme_fd, (void*)vec_buffer, readVecSize_bytes, 0) == -1)
            {
                std::cout << "CPU: ssd to host_mem, err line: " << __LINE__ << std::endl;
            }
            readVec_end = std::chrono::high_resolution_clock::now();
            unsigned long readVec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(readVec_end - readVec_start).count();

            double dnsduration = (double)readVec_time;
            double dsduration = dnsduration / ((double)1000000000);
            double read_bandwidth = (readVecSize_bytes / dsduration) / ((double)1024 * 1024);

            eva_logger.write_to_csv("sift1B", "cop_cpu_rssd_Seq_Read", "int", 128, vec_num, dsduration * 1000, read_bandwidth);
        }
        break;
    }
    
    case RANDOM_READ: { // random read
        std::string log_fname_1 = csv_log_path + std::string("test_cop_cpu_rssd_Random_Read.csv");
        CSV_Test_Disk_IO_LOGer eva_logger_1(log_fname_1.c_str());

        for(int vec_num = 1; vec_num <= 10000; vec_num *= 2)
        {
            int read_vec_index[vec_num];
            for(int index = 0; index < vec_num; index++)
            {
                read_vec_index[index] = get_random_vecid(0, 1000);
            }

            size_t readVecSize_bytes_1 = 128 * sizeof(int);

            readVec_start_1 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for(int read_index = 0; read_index < vec_num; read_index++)
            {
                // if(pread(nvme_fd, (void*)(vec_buffer + read_index * 128), readVecSize_bytes_1, read_vec_index[read_index] * 128) == -1)
                if(pread(nvme_fd, (void*)(vec_buffer + read_index * 128), readVecSize_bytes_1, 0) == -1)
                {
                    std::cout << "CPU: ssd to host_mem, err line: " << __LINE__ << std::endl;
                }
            }
            readVec_end_1 = std::chrono::high_resolution_clock::now();
            unsigned long readVec_time_1 = std::chrono::duration_cast<std::chrono::nanoseconds>(readVec_end_1 - readVec_start_1).count();

            double dnsduration_1 = (double)readVec_time_1;
            double dsduration_1 = dnsduration_1 / ((double)1000000000);
            double read_bandwidth_1 = (readVecSize_bytes_1 * vec_num / dsduration_1) / ((double)1024 * 1024);

            eva_logger_1.write_to_csv("sift1B", "cop_cpu_rssd_Random_Read", "int", 128, vec_num, dsduration_1 * 1000, read_bandwidth_1);
        }
        break;
    }

    default: {
        std::cout << "Please give read type 0 or 1 !" << std::endl;
        break;
    }
    }

    free(vec_buffer);

    return 0;
}

uint8_t test_cop_gpu_lssd(uint8_t type, const char* fname)
{
    int nvme_fd  = -1;

    std::chrono::high_resolution_clock::time_point readVec_start, readVec_end, readVec_start_1, readVec_end_1;

    // open nvme ssd vector file
    nvme_fd = open(fname, O_RDONLY | O_DIRECT);
    if(nvme_fd < 0)
    {
        std::cerr << "ERROR: open " << fname << " failed!" << std::endl;
        return EXIT_FAILURE;
    }

    // allocate buffer on host memory
    size_t buffer_size_bytes = 10000 * 128 * sizeof(int);
    int* vec_buffer = (int*)malloc(buffer_size_bytes);
    std::fill(vec_buffer, vec_buffer + 10000 * 128, 0);

    // allocate buffer on GPU memory
    int* gpu_buffer = NULL;
    cudaMallocManaged(&gpu_buffer, 10000 * 128);

    switch (type)
    {
    case SEQ_READ: { // seq read
        std::string log_fname = csv_log_path + std::string("test_cop_gpu_lssd_Seq_Read.csv");
        CSV_Test_Disk_IO_LOGer eva_logger(log_fname.c_str());

        for(int vec_num = 1; vec_num <= 10000; vec_num *= 2)
        {
            size_t readVecSize_bytes = vec_num * 128 * sizeof(int);
            readVec_start = std::chrono::high_resolution_clock::now();
            if(pread(nvme_fd, (void*)vec_buffer, readVecSize_bytes, 0) == -1)
            {
                std::cout << "CPU: ssd to host_mem, err line: " << __LINE__ << std::endl;
            }
            cudaError_t cudaStatus = cudaMemcpy(gpu_buffer, vec_buffer, vec_num * 128, cudaMemcpyHostToDevice);
            readVec_end = std::chrono::high_resolution_clock::now();
            unsigned long readVec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(readVec_end - readVec_start).count();

            double dnsduration = (double)readVec_time;
            double dsduration = dnsduration / ((double)1000000000);
            double read_bandwidth = (readVecSize_bytes / dsduration) / ((double)1024 * 1024);

            eva_logger.write_to_csv("sift1B", "cop_gpu_lssd_Seq_Read", "int", 128, vec_num, dsduration * 1000, read_bandwidth);
        }
        break;
    }
    
    case RANDOM_READ: { // random read
        std::string log_fname_1 = csv_log_path + std::string("test_cop_gpu_lssd_Random_Read.csv");
        CSV_Test_Disk_IO_LOGer eva_logger_1(log_fname_1.c_str());

        for(int vec_num = 1; vec_num <= 10000; vec_num *= 2)
        {
            int read_vec_index[vec_num];
            for(int index = 0; index < vec_num; index++)
            {
                read_vec_index[index] = get_random_vecid(0, 1000);
            }

            size_t readVecSize_bytes_1 = 128 * sizeof(int);

            readVec_start_1 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for(int read_index = 0; read_index < vec_num; read_index++)
            {
                // if(pread(nvme_fd, (void*)(vec_buffer + read_index * 128), readVecSize_bytes_1, read_vec_index[read_index] * 128) == -1)
                if(pread(nvme_fd, (void*)(vec_buffer + read_index * 128), readVecSize_bytes_1, 0) == -1)
                {
                    std::cout << "CPU: ssd to host_mem, err line: " << __LINE__ << std::endl;
                }
            }
            cudaError_t cudaStatus = cudaMemcpy(gpu_buffer, vec_buffer, vec_num * 128, cudaMemcpyHostToDevice);
            readVec_end_1 = std::chrono::high_resolution_clock::now();
            unsigned long readVec_time_1 = std::chrono::duration_cast<std::chrono::nanoseconds>(readVec_end_1 - readVec_start_1).count();

            double dnsduration_1 = (double)readVec_time_1;
            double dsduration_1 = dnsduration_1 / ((double)1000000000);
            double read_bandwidth_1 = (readVecSize_bytes_1 * vec_num / dsduration_1) / ((double)1024 * 1024);

            eva_logger_1.write_to_csv("sift1B", "cop_gpu_lssd_Random_Read", "int", 128, vec_num, dsduration_1 * 1000, read_bandwidth_1);
        }
        break;
    }

    default: {
        std::cout << "Please give read type 0 or 1 !" << std::endl;
        break;
    }
    }

    free(vec_buffer);
    cudaFree(gpu_buffer);

    return 0;
}

uint8_t test_cop_gpu_rssd(uint8_t type, const char* fname)
{
    int nvme_fd  = -1;

    std::chrono::high_resolution_clock::time_point readVec_start, readVec_end, readVec_start_1, readVec_end_1;

    // open nvme ssd vector file
    nvme_fd = open(fname, O_RDONLY | O_DIRECT);
    if(nvme_fd < 0)
    {
        std::cerr << "ERROR: open " << fname << " failed!" << std::endl;
        return EXIT_FAILURE;
    }

    // allocate buffer on host memory
    size_t buffer_size_bytes = 10000 * 128 * sizeof(int);
    int* vec_buffer = (int*)malloc(buffer_size_bytes);
    std::fill(vec_buffer, vec_buffer + 10000 * 128, 0);

    // allocate buffer on GPU memory
    int* gpu_buffer = NULL;
    cudaMallocManaged(&gpu_buffer, 10000 * 128);

    switch (type)
    {
    case SEQ_READ: { // seq read
        std::string log_fname = csv_log_path + std::string("test_cop_gpu_rssd_Seq_Read.csv");
        CSV_Test_Disk_IO_LOGer eva_logger(log_fname.c_str());

        for(int vec_num = 1; vec_num <= 10000; vec_num *= 2)
        {
            size_t readVecSize_bytes = vec_num * 128 * sizeof(int);
            readVec_start = std::chrono::high_resolution_clock::now();
            if(pread(nvme_fd, (void*)vec_buffer, readVecSize_bytes, 0) == -1)
            {
                std::cout << "CPU: ssd to host_mem, err line: " << __LINE__ << std::endl;
            }
            cudaError_t cudaStatus = cudaMemcpy(gpu_buffer, vec_buffer, vec_num * 128, cudaMemcpyHostToDevice);
            readVec_end = std::chrono::high_resolution_clock::now();
            unsigned long readVec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(readVec_end - readVec_start).count();

            double dnsduration = (double)readVec_time;
            double dsduration = dnsduration / ((double)1000000000);
            double read_bandwidth = (readVecSize_bytes / dsduration) / ((double)1024 * 1024);

            eva_logger.write_to_csv("sift1B", "cop_gpu_rssd_Seq_Read", "int", 128, vec_num, dsduration * 1000, read_bandwidth);
        }
        break;
    }
    
    case RANDOM_READ: { // random read
        std::string log_fname_1 = csv_log_path + std::string("test_cop_gpu_rssd_Random_Read.csv");
        CSV_Test_Disk_IO_LOGer eva_logger_1(log_fname_1.c_str());

        for(int vec_num = 1; vec_num <= 10000; vec_num *= 2)
        {
            int read_vec_index[vec_num];
            for(int index = 0; index < vec_num; index++)
            {
                read_vec_index[index] = get_random_vecid(0, 1000);
            }

            size_t readVecSize_bytes_1 = 128 * sizeof(int);

            readVec_start_1 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for(int read_index = 0; read_index < vec_num; read_index++)
            {
                // if(pread(nvme_fd, (void*)(vec_buffer + read_index * 128), readVecSize_bytes_1, read_vec_index[read_index] * 128) == -1)
                if(pread(nvme_fd, (void*)(vec_buffer + read_index * 128), readVecSize_bytes_1, 0) == -1)
                {
                    std::cout << "CPU: ssd to host_mem, err line: " << __LINE__ << std::endl;
                }
            }
            cudaError_t cudaStatus = cudaMemcpy(gpu_buffer, vec_buffer, vec_num * 128, cudaMemcpyHostToDevice);
            readVec_end_1 = std::chrono::high_resolution_clock::now();
            unsigned long readVec_time_1 = std::chrono::duration_cast<std::chrono::nanoseconds>(readVec_end_1 - readVec_start_1).count();

            double dnsduration_1 = (double)readVec_time_1;
            double dsduration_1 = dnsduration_1 / ((double)1000000000);
            double read_bandwidth_1 = (readVecSize_bytes_1 * vec_num / dsduration_1) / ((double)1024 * 1024);

            eva_logger_1.write_to_csv("sift1B", "cop_gpu_rssd_Random_Read", "int", 128, vec_num, dsduration_1 * 1000, read_bandwidth_1);
        }
        break;
    }

    default: {
        std::cout << "Please give read type 0 or 1 !" << std::endl;
        break;
    }
    }

    free(vec_buffer);
    cudaFree(gpu_buffer);

    return 0;
}

uint8_t test_NetANN_cpu(uint8_t type, const char* fname, xrtDeviceHandle device, xrt::kernel& krnl)
{
    int nvme_fd  = -1;

    std::chrono::high_resolution_clock::time_point readVec_start, readVec_end, readVec_start_1, readVec_end_1;

    // open nvme ssd vector file
    nvme_fd = open(fname, O_RDONLY | O_DIRECT);
    if(nvme_fd < 0)
    {
        std::cerr << "ERROR: open " << fname << " failed!" << std::endl;
        return EXIT_FAILURE;
    }

    // allocate buffer on FPGA
    size_t fpga_p2p_buffer_size_bytes = 10000 * 128 * sizeof(int);
    auto fpga_p2p_buffer = xrt::bo(device, fpga_p2p_buffer_size_bytes, krnl.group_id(0));

    // map fpga buffer to host memory
    auto fpga_p2p_buffer_map = fpga_p2p_buffer.map<int*>();
    std::fill(fpga_p2p_buffer_map, fpga_p2p_buffer_map + 10000 * 28, 0);

    switch (type)
    {
    case SEQ_READ: { // seq read
        std::string log_fname = csv_log_path + std::string("test_NetANN_cpu_Seq_Read.csv");
        CSV_Test_Disk_IO_LOGer eva_logger(log_fname.c_str());

        for(int vec_num = 1; vec_num <= 10000; vec_num *= 2)
        {
            size_t readVecSize_bytes = vec_num * 128 * sizeof(int);
            readVec_start = std::chrono::high_resolution_clock::now();
            if(pread(nvme_fd, (void*)fpga_p2p_buffer_map, readVecSize_bytes, 0) == -1)
            {
                std::cout << "CPU: ssd to fpga_map_host_mem, err line: " << __LINE__ << std::endl;
            }
            fpga_p2p_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            readVec_end = std::chrono::high_resolution_clock::now();
            unsigned long readVec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(readVec_end - readVec_start).count();

            double dnsduration = (double)readVec_time;
            double dsduration = dnsduration / ((double)1000000000);
            double read_bandwidth = (readVecSize_bytes / dsduration) / ((double)1024 * 1024);

            eva_logger.write_to_csv("sift1B", "NetANN_cpu_Seq_Read", "int", 128, vec_num, dsduration * 1000, read_bandwidth);
        }
        break;
    }
    
    case RANDOM_READ: { // random read
        std::string log_fname_1 = csv_log_path + std::string("test_NetANN_cpu_Random_Read.csv");
        CSV_Test_Disk_IO_LOGer eva_logger_1(log_fname_1.c_str());

        for(int vec_num = 1; vec_num <= 10000; vec_num *= 2)
        {
            int read_vec_index[vec_num];
            for(int index = 0; index < vec_num; index++)
            {
                read_vec_index[index] = get_random_vecid(0, 1000);
            }

            size_t readVecSize_bytes_1 = 128 * sizeof(int);

            readVec_start_1 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for(int read_index = 0; read_index < vec_num; read_index++)
            {
                // if(pread(nvme_fd, (void*)(fpga_p2p_buffer_map + read_index * 128), readVecSize_bytes_1, read_vec_index[read_index] * 128) == -1)
                if(pread(nvme_fd, (void*)(fpga_p2p_buffer_map + read_index * 128), readVecSize_bytes_1, 0) == -1)
                {
                    std::cout << "CPU: ssd to fpga_map_host_mem, err line: " << __LINE__ << std::endl;
                }
            }
            fpga_p2p_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            readVec_end_1 = std::chrono::high_resolution_clock::now();
            unsigned long readVec_time_1 = std::chrono::duration_cast<std::chrono::nanoseconds>(readVec_end_1 - readVec_start_1).count();

            double dnsduration_1 = (double)readVec_time_1;
            double dsduration_1 = dnsduration_1 / ((double)1000000000);
            double read_bandwidth_1 = (readVecSize_bytes_1 * vec_num / dsduration_1) / ((double)1024 * 1024);

            eva_logger_1.write_to_csv("sift1B", "NetANN_cpu_Random_Read", "int", 128, vec_num, dsduration_1 * 1000, read_bandwidth_1);
        }
        break;
    }

    default: {
        std::cout << "Please give read type 0 or 1 !" << std::endl;
        break;
    }
    }

    return 0;
}

uint8_t test_NetANN_R(uint8_t type, const char* fname, xrtDeviceHandle device, xrt::kernel& krnl)
{
    int nvme_fd  = -1;

    std::chrono::high_resolution_clock::time_point readVec_start, readVec_end, readVec_start_1, readVec_end_1;

    // open nvme ssd vector file
    nvme_fd = open(fname, O_RDONLY | O_DIRECT);
    if(nvme_fd < 0)
    {
        std::cerr << "ERROR: open " << fname << " failed! " << std::endl;
        return EXIT_FAILURE;
    }

    // allocate buffer on FPGA
    size_t fpga_p2p_buffer_size_bytes = 10000 * 128 * sizeof(int);
    xrt::bo::flags flags = xrt::bo::flags::p2p;
    auto fpga_p2p_buffer = xrt::bo(device, fpga_p2p_buffer_size_bytes, flags, krnl.group_id(0));

    // map fpga buffer to host memory
    auto fpga_p2p_buffer_map =  fpga_p2p_buffer.map<int*>();
    std::fill(fpga_p2p_buffer_map, fpga_p2p_buffer_map + 10000 * 28, 0);

    switch (type)
    {
    case SEQ_READ: { // seq read
        std::string log_fname = csv_log_path + std::string("test_NetANN_R_Seq_Read.csv");
        CSV_Test_Disk_IO_LOGer eva_logger(log_fname.c_str());

        for(int vec_num = 1; vec_num <= 10000; vec_num *= 2)
        {
            size_t readVecSize_bytes = vec_num * 128 * sizeof(int);
            readVec_start = std::chrono::high_resolution_clock::now();
            if(pread(nvme_fd, (void*)fpga_p2p_buffer_map, readVecSize_bytes, 0) == -1)
            {
                std::cout << "P2P: ssd to fpga, err line: " << __LINE__ << std::endl;
            }
            readVec_end = std::chrono::high_resolution_clock::now();
            unsigned long readVec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(readVec_end - readVec_start).count();

            double dnsduration = (double)readVec_time;
            double dsduration = dnsduration / ((double)1000000000);
            double read_bandwidth = (readVecSize_bytes / dsduration) / ((double)1024 * 1024);

            eva_logger.write_to_csv("sift1B", "NetANN_R_Seq_Read", "int", 128, vec_num, dsduration * 1000, read_bandwidth);
        }
        break;
    }
    
    case RANDOM_READ: { // random read
        std::string log_fname_1 = csv_log_path + std::string("test_NetANN_R_Random_Read.csv");
        CSV_Test_Disk_IO_LOGer eva_logger_1(log_fname_1.c_str());

        for(int vec_num = 1; vec_num <= 10000; vec_num *= 2)
        {
            int read_vec_index[vec_num];
            for(int index = 0; index < vec_num; index++)
            {
                read_vec_index[index] = get_random_vecid(0, 1000);
            }

            size_t readVecSize_bytes_1 = 128 * sizeof(int);

            readVec_start_1 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for(int read_index = 0; read_index < vec_num; read_index++)
            {
                // if(pread(nvme_fd, (void*)(fpga_p2p_buffer_map + read_index * 128), readVecSize_bytes_1, read_vec_index[read_index] * 128) == -1)
                if(pread(nvme_fd, (void*)(fpga_p2p_buffer_map + read_index * 128), readVecSize_bytes_1, 0) == -1)
                {
                    std::cout << "P2P: ssd to fpga, err line: " << __LINE__ << std::endl;
                }
            }
            readVec_end_1 = std::chrono::high_resolution_clock::now();
            unsigned long readVec_time_1 = std::chrono::duration_cast<std::chrono::nanoseconds>(readVec_end_1 - readVec_start_1).count();

            double dnsduration_1 = (double)readVec_time_1;
            double dsduration_1 = dnsduration_1 / ((double)1000000000);
            double read_bandwidth_1 = (readVecSize_bytes_1 * vec_num / dsduration_1) / ((double)1024 * 1024);

            eva_logger_1.write_to_csv("sift1B", "NetANN_R_Random_Read", "int", 128, vec_num, dsduration_1 * 1000, read_bandwidth_1);
        }
        break;
    }

    default: {
        std::cout << "Please give read type 0 or 1 !" << std::endl;
        break;
    }
    }

    return 0;
}

int main(int argc, char* argv[])
{
    if(argc < 4)
    {
        std::cout << "Please give test parameters:\n(1) test_type: 1(Seq Read) or 2(Random Read).\n(2) vector dataset path.\n(3)xclbin file path." << std::endl;

        return EXIT_FAILURE;
    }

    uint8_t read_type = *(argv[1]) - '0';
    const char* vec_dataset_path = argv[2];
    if(read_type == 1)
    {
        std::cout << "Seq Read Vector in Path: " << vec_dataset_path << std::endl;
    }
    else if(read_type == 2)
    {
        std::cout << "Random Read Vector in Path: " << vec_dataset_path << std::endl;
    }

    std::string xclbin_file = std::string(argv[3]);
    std::cout << "Open the device" << '0' << std::endl;
    auto fpgaDevice = xrt::device(0);
    std::cout << "Load the xclbin: " << xclbin_file << std::endl;
    auto fpga_test_kernel_uuid = fpgaDevice.load_xclbin(xclbin_file);

     // creating FPGA test kernel
    auto fpga_test_kernl = xrt::kernel(fpgaDevice, fpga_test_kernel_uuid, "vector_search_centroids_top");

    // test_NetANN_R(read_type, vec_dataset_path, fpgaDevice, fpga_test_kernl);
    // test_NetANN_cpu(read_type, vec_dataset_path, fpgaDevice, fpga_test_kernl);
    // test_cop_gpu_rssd(read_type, vec_dataset_path);
    test_cop_gpu_lssd(read_type, vec_dataset_path);
    // test_cop_cpu_rssd(read_type, vec_dataset_path);

    return 0;
}
