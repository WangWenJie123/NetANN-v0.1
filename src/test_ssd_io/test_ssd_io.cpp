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
#define SEQ_READ        1
#define RANDOM_READ     2

// define log file path
std::string csv_log_path = "/home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/eva_logs/";

class CSV_Test_Disk_IO_LOGer:
{
    public:
        CSV_Test_Disk_IO_LOGer(const char* logger_file)
        {
            file = std::ofstream(logger_file, std::ios::app);
            file << "datasets" << ',' << "read_type" << ',' << "vec_data_type" << ',' << "vec_dim" << ',' << "read_vec_num" << ',' << "search_latency/ms" << std::endl;
        }

        ~CSV_Test_Disk_IO_LOGers()
        {
            file.close()
        }

    public:
        void write_to_csv(const char* datasets, const char* read_type, const char* vec_data_type, int vec_dim, int read_vec_num, double search_latency)
        {
            file << datasets << ',' << read_type << ',' << vec_data_type << ',' << std::to_string(vec_dim) << ',' << std::to_string(read_vec_num) << ',' << std::to_string(search_latency) << std::endl;
        }
    
    private:
        std::ofstream file;
}

uint8_t test_cop-cpu-rssd(uint8_t type, const char* fname)
{
    return 0;
}

uint8_t test_cop-gpu-lssd(uint8_t type, const char* fname)
{
    return 0;
}

uint8_t test_cop-gpu-rssd(uint8_t type, const char* fname)
{
    return 0;
}

uint8_t test_NetANN-cpu(uint8_t type, const char* fname)
{
    return 0;
}

uint8_t test_NetANN_R(uint8_t type, const char* fname, xrtDeviceHandle device, xrt::kernel& krnl)
{
    int nvme_fd  = -1;

    // open nvme ssd vector file
    nvme_fd = open(fname, O_RDWR | O_DIRECT);
    if(nvme_fd < 0)
    {
        std::cerr << "ERROR: open " << fname << "failed: " << std::endl;
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
    case SEQ_READ: // seq read
        std::string log_fname = csv_log_path + std::string("test_NetANN_R_Seq_Read.csv");
        CSV_Test_Disk_IO_LOGer eva_logger(log_fname.c_str());

        
        break;
    
    case RANDOM_READ: // random read

        break;

    default: // seq read
        break;
    }

    return 0;
}

int main(int argc, char* argv[])
{
    printf("hello....\n");

    return 0;
}