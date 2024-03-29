#ifndef __P2P_SSD__
#define __P2P_SSD__

void p2p_fpga_to_ssd(int* nvmeFd, void** fpga_hostMap_addr, size_t* vector_size_bytes, int parallel_num, int* offset);
void p2p_ssd_to_fpga(int* nvmeFd, void** fpga_hostMap_addr, size_t* vector_size_bytes, int parallel_num, int* offset);

#endif