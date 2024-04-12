#!/bin/bash

source /opt/xilinx/xrt/setup.sh
source /tools/Xilinx/Vitis/2022.2/settings64.sh
make all TARGET=sw_emu PLATFORM=xilinx_u200_gen3x16_xdma_2_202110_1
export XCL_EMULATION_MODE=sw_emu
./hello_world_xrt -x ./build_dir.sw_emu.xilinx_u200_gen3x16_xdma_2_202110_1/vadd.xclbin