#include <stdint.h>
#include <hls_math.h>
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"

#define MAX_VECTOR_DIM 1024
#define MAX_NPROBE 4096
#define MAX_CENTROIDS_NUM 4096

#define DWIDTH 32

auto constexpr DATA_WIDTH = 512;
auto constexpr c_widthInBytes = DATA_WIDTH / 8;
auto constexpr c_widthInInt = c_widthInBytes / 4;

typedef ap_axiu<DWIDTH, 0, 0, 0> pkt;
using TYPE = ap_int<DATA_WIDTH>;

static void compute_nearest(int outL2dis[MAX_CENTROIDS_NUM], int local_oputCentroids_id[MAX_CENTROIDS_NUM], unsigned int numCentroids, unsigned int nprobe)
{
    int sort_tmp[MAX_CENTROIDS_NUM];
#pragma HLS ARRAY_RESHAPE variable=sort_tmp type=block factor=2 dim=1

// #pragma HLS pipeline II=1

    // 并行全比排序，选出top-k(nprobe)个聚类中心
    for(int i = 0; i < numCentroids; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min = MAX_CENTROIDS_NUM max = MAX_CENTROIDS_NUM
// #pragma HLS pipeline off
#pragma HLS unroll factor=2
        for(int j = 0; j < numCentroids; j++)
        {
#pragma HLS LOOP_TRIPCOUNT min = MAX_CENTROIDS_NUM max = MAX_CENTROIDS_NUM
// #pragma HLS pipeline off
#pragma HLS unroll factor=2

            if(outL2dis[i] < outL2dis[j]) sort_tmp[i] += 1;
        }
    }

    unsigned int needed_id = 0;
    unsigned int out_cent_idex = 0;
    needed_id = numCentroids - nprobe;
    for(int i = 0; i < numCentroids; i++ )
    {
#pragma HLS LOOP_TRIPCOUNT min = MAX_CENTROIDS_NUM max = MAX_CENTROIDS_NUM
// #pragma HLS pipeline off
#pragma HLS unroll factor=2

        if(sort_tmp[i] >= needed_id)
        {
            local_oputCentroids_id[out_cent_idex] = i;
            // local_oputCentroids_id << i;
            out_cent_idex += 1;
            if(out_cent_idex >= nprobe) break;
        }
    }
}

extern "C" 
{

// void vector_search_centroids_top(int* mem_xqVector, int* mem_CentroidsVector, int* oputCentroids_id, unsigned int numCentroids, unsigned int dim, unsigned int nprobe
// // , hls::stream<pkt>& out_xq_vector
// )
void vector_search_centroids_top(TYPE* mem_xqVector, TYPE* mem_CentroidsVector, int* oputCentroids_id, unsigned int numCentroids, unsigned int dim, unsigned int nprobe
// , hls::stream<pkt>& out_xq_vector
)
{
// #pragma HLS INTERFACE m_axi port = mem_xqVector offset = slave bundle = gmem0
// #pragma HLS INTERFACE m_axi port = mem_CentroidsVector offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = mem_xqVector offset = slave bundle = gmem0 max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = mem_CentroidsVector offset = slave bundle = gmem0 max_write_burst_length = 64 num_write_outstanding = 16
#pragma HLS INTERFACE m_axi port = oputCentroids_id offset = slave bundle = gmem0
#pragma HLS INTERFACE s_axilite port = mem_xqVector
#pragma HLS INTERFACE s_axilite port = mem_CentroidsVector
#pragma HLS INTERFACE s_axilite port = oputCentroids_id
#pragma HLS INTERFACE s_axilite port = numCentroids
#pragma HLS INTERFACE s_axilite port = dim
#pragma HLS INTERFACE s_axilite port = nprobe
#pragma HLS INTERFACE ap_ctrl_chain port = return

#pragma HLS cache port=mem_xqVector lines=64 depth=128 
#pragma HLS cache port=mem_CentroidsVector lines=64 depth=128 

    // static hls::stream<int> local_xqVector("local_xqVector");
    // static hls::stream<int> local_centroidsVector("local_centroidsVector");
    // static hls::stream<int> local_oputCentroids_id("local_oputCentroids_id");

    int local_xqVector[MAX_VECTOR_DIM];
    int local_centroidsVector[MAX_VECTOR_DIM];
    int local_oputCentroids_id[MAX_NPROBE];
    // int outL2dis_tmp[MAX_CENTROIDS_NUM];
    int outL2dis[MAX_CENTROIDS_NUM];

#pragma HLS ARRAY_RESHAPE variable=local_xqVector type=block factor=2 dim=1
#pragma HLS ARRAY_RESHAPE variable=local_centroidsVector type=block factor=2 dim=1
#pragma HLS ARRAY_RESHAPE variable=local_oputCentroids_id type=block factor=2 dim=1
#pragma HLS ARRAY_RESHAPE variable=outL2dis type=block factor=2 dim=1
// #pragma HLS bind_storage variable=outL2dis_tmp impl=lutram
// #pragma HLS bind_storage variable=outL2dis impl=lutram

// #pragma HLS pipeline II=1
// #pragma HLS dataflow

    uint32_t read_q_vec_times = dim / c_widthInInt;
    TYPE temp, temp_1;

    // for(int i = 0; i < dim; i++)
    for(int i = 0; i < read_q_vec_times; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min = MAX_VECTOR_DIM max = MAX_VECTOR_DIM
// #pragma HLS pipeline off
#pragma HLS unroll factor=2

        // local_xqVector[i] = mem_xqVector[i];
        // local_xqVector << mem_xqVector[i];
        // pkt xq_vec;
        // xq_vec.data = local_xqVector[i];
        // out_xq_vector.write(xq_vec);

        temp = mem_xqVector[i];
        for (int j = 0; j < 16; j++) 
        {
            local_xqVector[i * 16 + j] = (temp >> (32 * j)) & 0xFFFFFFFF;
        }
    }

    for(int i = 0; i < numCentroids; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min = MAX_CENTROIDS_NUM max = MAX_CENTROIDS_NUM
// #pragma HLS pipeline II=1
#pragma HLS unroll factor=2

        // for(int j = 0; j < dim; j++)
        for(int j = 0; j < read_q_vec_times; j++)
        {
#pragma HLS LOOP_TRIPCOUNT min = MAX_VECTOR_DIM max = MAX_VECTOR_DIM
// #pragma HLS pipeline off
#pragma HLS unroll factor=2

            // local_centroidsVector[j] = mem_CentroidsVector[i * dim + j];
            // local_centroidsVector << mem_CentroidsVector[i * dim + j];

            temp_1 = mem_CentroidsVector[i * read_q_vec_times + j];
            for (int m = 0; m < 16; m++) 
            {
                local_centroidsVector[j * 16 + m] = (temp_1 >> (32 * m)) & 0xFFFFFFFF;
            }
        }

        for(int z = 0; z < dim; z++)
        {
#pragma HLS LOOP_TRIPCOUNT min = MAX_VECTOR_DIM max = MAX_VECTOR_DIM
// #pragma HLS pipeline off
#pragma HLS unroll factor=2

            outL2dis[i] += pow((local_xqVector[z] - local_centroidsVector[z]), 2);
            // outL2dis_tmp[i] += pow((local_xqVector.read() - local_centroidsVector.read()), 2);
        }

        outL2dis[i] = sqrt(outL2dis[i]);
    }
    
    compute_nearest(outL2dis, local_oputCentroids_id, numCentroids, nprobe);
    
    for(int i = 0; i < nprobe; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min = MAX_NPROBE max = MAX_NPROBE 
// #pragma HLS pipeline off
#pragma HLS unroll factor=2

        oputCentroids_id[i] = local_oputCentroids_id[i];
        // oputCentroids_id[i] = local_oputCentroids_id.read();
    }
}

}
