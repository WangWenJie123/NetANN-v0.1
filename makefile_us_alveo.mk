#
# Copyright 2019-2021 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# makefile-generator v1.0.3
#

############################## Help Section ##############################
ifneq ($(findstring Makefile, $(MAKEFILE_LIST)), Makefile)
help:
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make test PLATFORM=<FPGA platform>"
	$(ECHO) "      Command to run the application. This is same as 'run' target but does not have any makefile dependency."
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) ""
	$(ECHO) "  make host"
	$(ECHO) "      Command to build host application."
	$(ECHO) ""
endif

############################## Setting up Project Variables ##############################
TARGET := hw
include ./utils.mk

TEMP_DIR := ./_x.$(TARGET).$(XSA)
BUILD_DIR := ./build_dir.$(TARGET).$(XSA)

# LINK_OUTPUT := $(BUILD_DIR)/adder.link.xclbin
LINK_OUTPUT := $(BUILD_DIR)/vector_search_kernels.link.xclbin
PACKAGE_OUT = ./package.$(TARGET)

VPP_PFLAGS := 
# CMD_ARGS = -x $(BUILD_DIR)/adder.xclbin -f ./data/sample.txt
# CMD_ARGS = -x $(BUILD_DIR)/vector_search_kernels.xclbin -f ./data/sample.txt
CMD_ARGS = $(BUILD_DIR)/vector_search_kernels.xclbin
include config.mk

CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_VIVADO)/include -Wall -O0 -g -std=c++17
LDFLAGS += -L$(XILINX_XRT)/lib -pthread -lOpenCL

########################## Checking if PLATFORM in allowlist #######################
PLATFORM_BLOCKLIST += zc u25_ u30 vck aws 201910 u50_gen3x16_xdma_2019 
PLATFORM_ALLOWLIST += samsung_u2 

############################## Setting up Host Variables ##############################
#Include Required Host Source Files
CXXFLAGS += -I$(XF_PROJ_ROOT)/common/includes/cmdparser
CXXFLAGS += -I$(XF_PROJ_ROOT)/common/includes/logger
CXXFLAGS += -I./src
# HOST_SRCS += $(XF_PROJ_ROOT)/common/includes/cmdparser/cmdlineparser.cpp $(XF_PROJ_ROOT)/common/includes/logger/logger.cpp ./src/p2p_ssd.cpp ./src/host.cpp 
HOST_SRCS += $(XF_PROJ_ROOT)/common/includes/cmdparser/cmdlineparser.cpp $(XF_PROJ_ROOT)/common/includes/logger/logger.cpp ./src/p2p_ssd.cpp ./src/read_vector_datasets.cpp ./src/csv_log.cpp ./src/read_vector_datasets.cpp ./src/kernel_manager.cpp ./src/vector_search_test_host_pipline.cpp 

HOST_SRCS_TEST_SSD_IO += $(XF_PROJ_ROOT)/common/includes/cmdparser/cmdlineparser.cpp $(XF_PROJ_ROOT)/common/includes/logger/logger.cpp ./src/p2p_ssd.cpp ./src/read_vector_datasets.cpp ./src/csv_log.cpp ./src/test_ssd_io/test_ssd_io.cpp 

HOST_SRCS_TEST_SEARCH += $(XF_PROJ_ROOT)/common/includes/cmdparser/cmdlineparser.cpp $(XF_PROJ_ROOT)/common/includes/logger/logger.cpp ./src/p2p_ssd.cpp ./src/read_vector_datasets.cpp ./src/csv_log.cpp ./src/read_vector_datasets.cpp ./src/kernel_manager.cpp ./src/vector_search_test_host.cpp 

# Host compiler global settings
CXXFLAGS += -fmessage-length=0
CXXFLAGS += -I/usr/local/cuda-12.1/include
LDFLAGS += -lrt -lstdc++ 
LDFLAGS += -luuid -lxrt_coreutil

############################## Setting up Kernel Variables ##############################
# Kernel compiler global settings
VPP_FLAGS += -t $(TARGET) --platform $(PLATFORM) --save-temps 
VPP_FLAGS += --config vadd_pcie.cfg
VPP_LDFLAGS += --config vadd_pcie.cfg

# EXECUTABLE = ./GPU_FPGA_P2P_Test
VPP_FLAGS_vector_search_centroids_top_hls +=  --config ./vector_search_centroids_top_hls.cfg
VPP_LDFLAGS_vector_search_centroids_top += --config ./advanced.cfg --config ./vector_search_centroids_top.cfg

EXECUTABLE = ./vector_search_test
EXECUTABLE_TEST_SEARCH = ./vector_search_test_search
EXECUTABLE_TEST_SSD_IO = ./test_ssd_io
EMCONFIG_DIR = $(TEMP_DIR)

############################## Setting Targets ##############################
.PHONY: all clean cleanall docs emconfig
all: check-platform check-device check-vitis $(EXECUTABLE) $(BUILD_DIR)/vector_search_kernels.xclbin emconfig
# all: check-platform check-device check-vitis $(EXECUTABLE) $(BUILD_DIR)/adder.xclbin emconfig

.PHONY: host
host: $(EXECUTABLE)

.PHONY: host_test_search
host_test_search: $(EXECUTABLE_TEST_SEARCH)

.PHONY: host_test_ssd_io
host_test_ssd_io: $(EXECUTABLE_TEST_SSD_IO)

.PHONY: build
build: check-vitis check-device $(BUILD_DIR)/vector_search_kernels.xclbin

.PHONY: xclbin
xclbin: build

############################## Setting Rules for Binary Containers (Building Kernels) ##############################
############################## Building Memory Version Kernels ##############################
# $(TEMP_DIR)/vector_search_centroids_top.xo: src/vector_search_centroids_top.cpp
# 	mkdir -p $(TEMP_DIR)
# 	v++ $(VPP_FLAGS) -c $(VPP_FLAGS_vector_search_centroids_top_hls) -k vector_search_centroids_top --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
# $(TEMP_DIR)/distribute_topK_top.xo: src/distribute_topK_top.cpp
# 	mkdir -p $(TEMP_DIR)
# 	v++ $(VPP_FLAGS) -c $(VPP_FLAGS_vector_search_centroids_top_hls) -k distribute_topK_top --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
# $(TEMP_DIR)/search_topK_vec_top.xo: src/search_topK_vec_top.cpp
# 	mkdir -p $(TEMP_DIR)
# 	v++ $(VPP_FLAGS) -c $(VPP_FLAGS_vector_search_centroids_top_hls) -k search_topK_vec_top --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'

############################## Building Stream Version Kernels ##############################
$(TEMP_DIR)/vector_search_centroids_top.xo: src/stream_search_kernels/vector_search_centroids_top.cpp
	mkdir -p $(TEMP_DIR)
	v++ $(VPP_FLAGS) -c $(VPP_FLAGS_vector_search_centroids_top_hls) -k vector_search_centroids_top --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(TEMP_DIR)/distribute_topK_top.xo: src/stream_search_kernels/distribute_topK_top.cpp
	mkdir -p $(TEMP_DIR)
	v++ $(VPP_FLAGS) -c $(VPP_FLAGS_vector_search_centroids_top_hls) -k distribute_topK_top --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'
$(TEMP_DIR)/search_topK_vec_top.xo: src/stream_search_kernels/search_topK_vec_top.cpp
	mkdir -p $(TEMP_DIR)
	v++ $(VPP_FLAGS) -c $(VPP_FLAGS_vector_search_centroids_top_hls) -k search_topK_vec_top --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<'


$(BUILD_DIR)/vector_search_kernels.xclbin: $(TEMP_DIR)/vector_search_centroids_top.xo $(TEMP_DIR)/distribute_topK_top.xo $(TEMP_DIR)/search_topK_vec_top.xo
	mkdir -p $(BUILD_DIR)
	v++ $(VPP_FLAGS) -l $(VPP_LDFLAGS) --temp_dir $(TEMP_DIR) $(VPP_LDFLAGS_vector_search_centroids_top) -o'$(LINK_OUTPUT)' $(+)
	v++ -p $(LINK_OUTPUT) $(VPP_FLAGS) --package.out_dir $(PACKAGE_OUT) -o $(BUILD_DIR)/vector_search_kernels.xclbin

############################## Setting Rules for Host (Building Host Executable) ##############################
$(EXECUTABLE): $(HOST_SRCS) | check-xrt
		g++ -o $@ $^ $(CXXFLAGS) $(LDFLAGS) -L/usr/local/cuda-12.1/lib64 -lcudart -fopenmp

$(EXECUTABLE_TEST_SSD_IO): $(HOST_SRCS_TEST_SSD_IO) | check-xrt
		g++ -o $@ $^ $(CXXFLAGS) $(LDFLAGS) -L/usr/local/cuda-12.1/lib64 -lcudart -fopenmp

$(EXECUTABLE_TEST_SEARCH): $(HOST_SRCS_TEST_SEARCH) | check-xrt
		g++ -o $@ $^ $(CXXFLAGS) $(LDFLAGS) -L/usr/local/cuda-12.1/lib64 -lcudart -fopenmp

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(PLATFORM) --od $(EMCONFIG_DIR)

############################## Setting Essential Checks and Running Rules ##############################
run: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	cp -rf $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) $(CMD_ARGS)
else
	$(EXECUTABLE) $(CMD_ARGS)
endif

.PHONY: test
test: $(EXECUTABLE)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	XCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) $(CMD_ARGS)
else
	$(EXECUTABLE) $(CMD_ARGS)
endif

############################## Cleaning Rules ##############################
# Cleaning stuff
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

clean_test_ssd_io:
	-$(RMDIR) $(EXECUTABLE_TEST_SSD_IO)

clean_test_search:
	-$(RMDIR) $(EXECUTABLE_TEST_SEARCH)

cleanall: clean
	-$(RMDIR) build_dir*
	-$(RMDIR) package.*
	-$(RMDIR) _x* *xclbin.run_summary qemu-memory-_* emulation _vimage pl* start_simulation.sh *.xclbin

