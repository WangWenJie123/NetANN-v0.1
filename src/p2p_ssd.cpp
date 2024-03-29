#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "p2p_ssd.h"

void p2p_fpga_to_ssd(int* nvmeFd, void** fpga_hostMap_addr, size_t* vector_size_bytes, int parallel_num, int* offset)
{
    // time recorder
    std::chrono::high_resolution_clock::time_point p2pStart, p2pEnd;
    unsigned long p2pTime;
    double dnsduration, dsduration, gbpersec;
    size_t total_trans_size_bytes = 0;

    p2pStart = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for(int i = 0; i < parallel_num; i++)
    {
        if(pwrite(nvmeFd[i], fpga_hostMap_addr[i], vector_size_bytes[i], offset[i]) == -1)
        {
            std::cout << "P2P: fpga to ssd, err thread index: " << i << ", line: " << __LINE__ << std::endl;
        }
    }
    p2pEnd = std::chrono::high_resolution_clock::now();
    p2pTime = std::chrono::duration_cast<std::chrono::nanoseconds>(p2pEnd - p2pStart).count();
    dnsduration = (double)p2pTime;
    dsduration = dnsduration / ((double)1000000000);
    for(int i = 0; i < parallel_num; i++)
    {
        total_trans_size_bytes += vector_size_bytes[i];
    }
    gbpersec = (total_trans_size_bytes / dsduration) / ((double)1024 * 1024 * 1024);
    std::cout << "FPGA->SSD\t" << std::setprecision(3) << std::fixed << dnsduration << "ns" << std::endl;
    std::cout << "FPGA->SSD\t" << std::setprecision(3) << std::fixed << gbpersec << "GB/s" << std::endl;
}

void p2p_ssd_to_fpga(int* nvmeFd, void** fpga_hostMap_addr, size_t* vector_size_bytes, int parallel_num, int* offset)
{
    // time recorder
    std::chrono::high_resolution_clock::time_point p2pStart, p2pEnd;
    unsigned long p2pTime;
    double dnsduration, dsduration, gbpersec;
    size_t total_trans_size_bytes = 0;

    p2pStart = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for(int i = 0; i < parallel_num; i++)
    {
        if(pread(nvmeFd[i], fpga_hostMap_addr[i], vector_size_bytes[i], offset[i]) == -1)
        {
            std::cout << "P2P: ssd to fpga, err thread index: " << i << ", line: " << __LINE__ << std::endl;
        }
    }
    p2pEnd = std::chrono::high_resolution_clock::now();
    p2pTime = std::chrono::duration_cast<std::chrono::nanoseconds>(p2pEnd - p2pStart).count();
    dnsduration = (double)p2pTime;
    dsduration = dnsduration / ((double)1000000000);
    for(int i = 0; i < parallel_num; i++)
    {
        total_trans_size_bytes += vector_size_bytes[i];
    }
    gbpersec = (total_trans_size_bytes / dsduration) / ((double)1024 * 1024 * 1024);
    std::cout << "SSD->FPGA\t" << std::setprecision(3) << std::fixed << dnsduration << "ns" << std::endl;
    std::cout << "SSD->FPGA\t" << std::setprecision(3) << std::fixed << gbpersec << "GB/s" << std::endl;
}