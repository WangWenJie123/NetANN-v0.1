/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <unistd.h>
#include <time.h>
#include "p2p_ssd.h"

#define DATA_SIZE 1024 * 1024
#define INCR_VALUE 10
#define num_cu_cuda_cpy 4
#define num_cu_ssd_cpy 4

#define ssd_file1 "/home/wwj/nvme_ssd_test_dir/sample1.txt"
#define ssd_file2 "/home/wwj/nvme_ssd_test_dir/sample2.txt"
#define ssd_file3 "/home/wwj/nvme_ssd_test_dir/sample3.txt"
#define ssd_file4 "/home/wwj/nvme_ssd_test_dir/sample4.txt"


int main(int argc, char** argv) {
    // Command Line Parser
    sda::utils::CmdLineParser parser;

    int inc = INCR_VALUE;
    int size = DATA_SIZE;

    // cuda
    int* host_ptr[num_cu_cuda_cpy];
    int* gpu_ptr[num_cu_cuda_cpy];
    cudaStream_t dataTransStream[num_cu_cuda_cpy];

    // ssd file fd
    int nvmeFd[num_cu_ssd_cpy] = {-1};

    // time recorder
    std::chrono::high_resolution_clock::time_point p2pReadStart, p2pReadEnd, kernelStart, kernelEnd;
    unsigned long p2pReadTime, kernelTime;
    double dnsduration, dsduration, gbpersec, kernel_dnsduration;

    // Switches
    //**************//"<Full Arg>",  "<Short Arg>", "<Description>", "<Default>"
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "0");
    parser.parse(argc, argv);

    // Read settings
    std::string binaryFile = parser.value("xclbin_file");
    int device_index = stoi(parser.value("device_id"));

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    auto device_name = device.get_info<xrt::info::device::name>();
    char* xcl_mode = getenv("XCL_EMULATION_MODE");
    if ((xcl_mode != nullptr) && !strcmp(xcl_mode, "hw_emu")) {
        if (device_name.find("2018") != std::string::npos) {
            std::cout << "[INFO]: The example is not supported for " << device_name
                      << " for hw_emu. Please try other flows." << '\n';
            return EXIT_SUCCESS;
        }
    }
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    // allocate FPGA kernel for GPU P2P
    xrt::kernel krnl[num_cu_cuda_cpy];
    for(int i = 0; i < num_cu_cuda_cpy; i++) {
        krnl[i] = xrt::kernel(device, uuid, "adder");
    }

    // allocate FPGA kernel for SSD P2P
    xrt::kernel krnl_ssd[num_cu_ssd_cpy];
    for(int i = 0; i < num_cu_ssd_cpy; i++) {
        krnl_ssd[i] = xrt::kernel(device, uuid, "adder");
    }


    size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
    size_t vector_size_bytes_ssd[num_cu_ssd_cpy] = {0};
    for(int i = 0; i < num_cu_ssd_cpy; i++) {
        vector_size_bytes_ssd[i] = sizeof(int) * DATA_SIZE;
    }

    // Allocate Buffer in FPGA Global Memory for GPU P2P
    xrt::bo::flags flags = xrt::bo::flags::p2p;
    std::cout << "Allocate Buffer in FPGA Global Memory for GPU P2P\n";
    xrt::bo p2p_bo0[num_cu_cuda_cpy];
    xrt::bo bo_out[num_cu_cuda_cpy];
    for(int i = 0; i < num_cu_cuda_cpy; i++) {
        p2p_bo0[i] = xrt::bo(device, vector_size_bytes, flags, krnl[i].group_id(0));
        bo_out[i] = xrt::bo(device, vector_size_bytes, krnl[i].group_id(1));
    }

    // Allocate Buffer in FPGA Global Memory for GPU P2P
    std::cout << "Allocate Buffer in FPGA Global Memory for SSD P2P\n";
    xrt::bo p2p_bo0_ssd[num_cu_ssd_cpy];
    xrt::bo bo_out_ssd[num_cu_ssd_cpy];
    for(int i = 0; i < num_cu_ssd_cpy; i++) {
        p2p_bo0_ssd[i] = xrt::bo(device, vector_size_bytes, flags, krnl_ssd[i].group_id(0));
        bo_out_ssd[i] = xrt::bo(device, vector_size_bytes, krnl_ssd[i].group_id(1));
    }

    // allocate host 
    for(int i = 0; i < num_cu_cuda_cpy; i++) {
        host_ptr[i] = (int*) malloc(vector_size_bytes);
        for (int j = 0; j < size; ++j) {
            host_ptr[i][j] = 0;
        }
    }

    // create cuda streams
    for(int i = 0; i < num_cu_cuda_cpy; i++) {
        cudaStreamCreate(&dataTransStream[i]);
    }

    // allocate gpu 
    for(int i = 0; i < num_cu_cuda_cpy; i++) {
        cudaMalloc((void**)&gpu_ptr[i], vector_size_bytes);
    }
#pragma omp parallel for
    for(int i = 0; i < num_cu_cuda_cpy; i++) {
        cudaMemcpyAsync(gpu_ptr[i], host_ptr[i], vector_size_bytes, 
                        cudaMemcpyHostToDevice, dataTransStream[i]);
        cudaStreamSynchronize(dataTransStream[i]);
    }

    // Map the contents of the buffer object into host memory for GPU P2P
    int* p2p_bo0_map[num_cu_cuda_cpy] = {nullptr};
    int* bo_out_map[num_cu_cuda_cpy] = {nullptr};
    for(int i = 0; i < num_cu_cuda_cpy; i++) {
        p2p_bo0_map[i] = p2p_bo0[i].map<int*>();
        bo_out_map[i] = bo_out[i].map<int*>();
        std::fill(p2p_bo0_map[i], p2p_bo0_map[i] + size, 0);
        std::fill(bo_out_map[i], bo_out_map[i] + size, 0);
    }

    // Map the contents of the buffer object into host memory for SSD P2P
    int* p2p_bo0_map_ssd[num_cu_ssd_cpy] = {nullptr};
    int* bo_out_map_ssd[num_cu_ssd_cpy] = {nullptr};
    for(int i = 0; i < num_cu_ssd_cpy; i++) {
        p2p_bo0_map_ssd[i] = p2p_bo0_ssd[i].map<int*>();
        bo_out_map_ssd[i] = bo_out_ssd[i].map<int*>();
        std::fill(p2p_bo0_map_ssd[i], p2p_bo0_map_ssd[i] + size, 0);
        std::fill(bo_out_map_ssd[i], bo_out_map_ssd[i] + size, 0);
    }

    // Create the test data
    int* bufReference[num_cu_cuda_cpy];
    for(int i = 0; i < num_cu_cuda_cpy; i++) {
        bufReference[i] = (int*) malloc(vector_size_bytes);
        for (int j = 0; j < size; ++j) {
            bo_out_map[i][j] = 0;
            bufReference[i][j] = host_ptr[i][j] + inc;
        }
    }
    std::cout << "Now start P2P Read from GPU to FPGA buffers\n" << std::endl;
    p2pReadStart = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for(int i = 0; i < num_cu_cuda_cpy; i++) {
        cudaMemcpyAsync(p2p_bo0_map[i], gpu_ptr[i], vector_size_bytes, 
                        cudaMemcpyDeviceToHost, dataTransStream[i]);
        cudaStreamSynchronize(dataTransStream[i]);
    }
    p2pReadEnd = std::chrono::high_resolution_clock::now();
    p2pReadTime = std::chrono::duration_cast<std::chrono::nanoseconds>(p2pReadEnd - p2pReadStart).count();
    dnsduration = (double)p2pReadTime;
    dsduration = dnsduration / ((double)10e9);
    gbpersec = (vector_size_bytes * num_cu_cuda_cpy / dsduration) / ((double)1024 * 1024 * 1024);
    std::cout << "GPU->FPGA\t" << std::setprecision(3) << std::fixed << dnsduration << "ns" << std::endl;
    std::cout << "GPU->FPGA\t" << std::setprecision(3) << std::fixed << gbpersec << "GB/s" << std::endl;

    std::cout << "\nExecution of the kernel\n";
    xrt::run run[num_cu_cuda_cpy];
    kernelStart = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < num_cu_cuda_cpy; i++) {
        run[i] = krnl[i](p2p_bo0[i], bo_out[i], inc, size);
    }

    for(int i = 0; i < num_cu_cuda_cpy; i++) {
        run[i].wait();
    }
    kernelEnd = std::chrono::high_resolution_clock::now();
    kernelTime = std::chrono::duration_cast<std::chrono::nanoseconds>(kernelEnd - kernelStart).count();
    kernel_dnsduration = (double)kernelTime;
    std::cout << "FPGA Kernel\t" << std::setprecision(3) << std::fixed << kernel_dnsduration << "ns" << std::endl;

    // Get the output;
    std::cout << "Get the output data from the device" << std::endl;
    for(int i = 0; i < num_cu_cuda_cpy; i++) {
        bo_out[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }

    // Validate our results
    for(int i = 0; i < num_cu_cuda_cpy; i++) {
        if (std::memcmp(bo_out_map[i], bufReference[i], size))
            throw std::runtime_error("Value read back does not match reference");
    }

    // delete cuda streams
    for(int i = 0; i < num_cu_cuda_cpy; i++) {
        cudaStreamDestroy(dataTransStream[i]);
    }

    // test performance for FPGA P2P SSD
    nvmeFd[0] = open(ssd_file1, O_RDWR | O_DIRECT | O_ASYNC);
    if (nvmeFd[0] < 0) {
        std::cerr << "ERROR: open " << ssd_file1 << " failed!" << std::endl;
        return EXIT_FAILURE;
    }
    nvmeFd[1] = open(ssd_file2, O_RDWR | O_DIRECT | O_ASYNC);
    if (nvmeFd[1] < 0) {
        std::cerr << "ERROR: open " << ssd_file2 << " failed!" << std::endl;
        return EXIT_FAILURE;
    }
    nvmeFd[2] = open(ssd_file3, O_RDWR | O_DIRECT | O_ASYNC);
    if (nvmeFd[2] < 0) {
        std::cerr << "ERROR: open " << ssd_file3 << " failed!" << std::endl;
        return EXIT_FAILURE;
    }
    nvmeFd[3] = open(ssd_file4, O_RDWR | O_DIRECT | O_ASYNC);
    if (nvmeFd[3] < 0) {
        std::cerr << "ERROR: open " << ssd_file4 << " failed!" << std::endl;
        return EXIT_FAILURE;
    }

    int ssd_file_offset[num_cu_ssd_cpy] = {0};
    p2p_fpga_to_ssd(nvmeFd, (void**)p2p_bo0_map_ssd, vector_size_bytes_ssd, 
                    num_cu_ssd_cpy, ssd_file_offset);
    std::cout << std::endl;
    p2p_ssd_to_fpga(nvmeFd, (void**)p2p_bo0_map_ssd, vector_size_bytes_ssd, 
                    num_cu_ssd_cpy, ssd_file_offset);

    (void)close(nvmeFd[0]);
    (void)close(nvmeFd[1]);
    (void)close(nvmeFd[2]);
    (void)close(nvmeFd[3]);

    std::cout << "TEST PASSED\n";
    return 0;
}