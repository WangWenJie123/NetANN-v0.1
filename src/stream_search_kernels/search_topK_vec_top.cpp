#include <stdint.h>
#include <hls_math.h>
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"

#define MAX_VECTOR_DIM 1024
#define DWIDTH 32

auto constexpr DATA_WIDTH = 512;
auto constexpr c_widthInBytes = DATA_WIDTH / 8;
auto constexpr c_widthInInt = c_widthInBytes / 4;

using TYPE = ap_int<DATA_WIDTH>;

/*
 * read xq vector from fpga memory  
*/
static void read_xq_vector(TYPE* mem_xqVector, int local_xqVector[MAX_VECTOR_DIM], unsigned int dim)
{
    unsigned int read_q_vec_times = dim / c_widthInInt;
    TYPE temp;

    for(int i = 0; i < read_q_vec_times; i++)
    {
#pragma HLS PIPELINE II = 1
        temp = mem_xqVector[i];
        for (int j = 0; j < 16; j++) 
        {
#pragma HLS PIPELINE II = 1
            local_xqVector[i * 16 + j] = (temp >> (32 * j)) & 0xFFFFFFFF;
        }
    }
}

/*
 * read base vector from fpga memory
*/
static void read_base_vector(TYPE* mem_CentroidsVector, hls::stream<int>& local_centroidsVector, unsigned int numCentroids, unsigned int dim)
{
    unsigned int read_q_vec_times = dim / c_widthInInt;
    TYPE temp;

    for(int i = 0; i < numCentroids; i++)
    {
// #pragma HLS unroll factor=8
#pragma HLS PIPELINE II = 1
        for(int j = 0; j < read_q_vec_times; j++)
        {
            temp = mem_CentroidsVector[i * read_q_vec_times + j];
            for (int m = 0; m < 16; m++) 
            {
#pragma HLS PIPELINE II = 1
                local_centroidsVector.write_nb((temp >> (32 * m)) & 0xFFFFFFFF);
            }
        }
    }
}

/*
 * compute l2 distance
*/
static void compute_l2(int local_xqVector[MAX_VECTOR_DIM], hls::stream<int>& local_centroidsVector, unsigned int numCentroids, unsigned int dim, hls::stream<int>& outL2dis_1, hls::stream<int>& outL2dis_2)
{
    int outL2dis_tmp;
    int local_centroidsVector_tmp;
    int dif_tmp;

    for(int i = 0; i < numCentroids; i++)
    {
// #pragma HLS unroll factor=8
#pragma HLS PIPELINE II = 1
        outL2dis_tmp = 0;
        dif_tmp = 0;
        for(int j = 0; j < dim; j++)
        {
#pragma HLS unroll factor=128
#pragma HLS PIPELINE II = 1 
            local_centroidsVector.read_nb(local_centroidsVector_tmp);
            dif_tmp = abs(local_xqVector[j] - local_centroidsVector_tmp);
            // outL2dis_tmp += pow(dif_tmp, 2);
            outL2dis_tmp += dif_tmp;
        }
        // outL2dis_tmp = sqrt(outL2dis_tmp);
        outL2dis_1.write_nb(outL2dis_tmp);
        outL2dis_2.write_nb(outL2dis_tmp);
    }
}

/*
 * parallel sorting
*/
static void parallel_sort(hls::stream<int>& outL2dis_1, hls::stream<int>& outL2dis_2, hls::stream<int>& sort_tmpStream, hls::stream<int>& outL2dis, unsigned int numCentroids)
{
    int sort_tmp;
    int outL2dis_tmp_1, outL2dis_tmp_2;

    // parallel sorting
    for(int i = 0; i < numCentroids; i++)
    {
// #pragma HLS unroll factor=8
#pragma HLS PIPELINE II = 1
        sort_tmp = 0;
        outL2dis_1.read_nb(outL2dis_tmp_1);
        for(int j = 0; j < numCentroids; j++)
        {
#pragma HLS PIPELINE II = 1
            outL2dis_2.read_nb(outL2dis_tmp_2);
            if(outL2dis_tmp_1 < outL2dis_tmp_2) sort_tmp += 1;
        }
        sort_tmpStream.write_nb(sort_tmp);
        outL2dis.write_nb(outL2dis_tmp_1);
    }
}

/*
 * select top k
*/
static void select_topK(hls::stream<int>& sort_tmpStream, hls::stream<int>& local_oputCentroids_id, hls::stream<int>& outL2dis, hls::stream<int>& outSelectedL2dis, unsigned int numCentroids, unsigned int nprobe)
{
    unsigned int needed_id = numCentroids - nprobe;
    unsigned int out_cent_idex = 0;
    int sort_tmp, outSelectedL2dis_tmp;

    for(int i = 0; i < numCentroids; i++ )
    {
// #pragma HLS unroll factor=8
#pragma HLS PIPELINE II = 1
        sort_tmpStream.read_nb(sort_tmp);
        outL2dis.read_nb(outSelectedL2dis_tmp);
        if(sort_tmp >= needed_id)
        {
            local_oputCentroids_id.write_nb(i);
            outSelectedL2dis.write_nb(outSelectedL2dis_tmp);
            out_cent_idex += 1;
            if(out_cent_idex >= nprobe) break;
        }
    }
}

/*
 * write topK vectors_id & vectors distance to fpga memory
*/
static void write_topK_vectors_id_distance(int* oputCentroids_id, int* out_topk_dis, hls::stream<int>& outSelectedL2dis, hls::stream<int>& local_oputCentroids_id, unsigned int nprobe)
{
    int local_oputCentroids_id_tmp, local_outSelectedL2dis_tmp;

    for(int i = 0; i < nprobe; i++)
    {
#pragma HLS PIPELINE II = 1
        local_oputCentroids_id.read_nb(local_oputCentroids_id_tmp);
        oputCentroids_id[i] = local_oputCentroids_id_tmp;
        outSelectedL2dis.read_nb(local_outSelectedL2dis_tmp);
        out_topk_dis[i] = local_outSelectedL2dis_tmp;
    }
}

extern "C" 
{

void search_topK_vec_top(TYPE* mem_xqVector, TYPE* mem_CentroidsVector, int* oputCentroids_id, unsigned int numCentroids, unsigned int dim, unsigned int nprobe, int* out_topk_dis)
{
#pragma HLS INTERFACE m_axi port = mem_xqVector offset = slave bundle = gmem0 max_read_burst_length = 32 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = mem_CentroidsVector offset = slave bundle = gmem1 latency=100 max_read_burst_length = 32 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = oputCentroids_id offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = out_topk_dis offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = mem_xqVector
#pragma HLS STABLE variable = mem_xqVector
#pragma HLS INTERFACE s_axilite port = mem_CentroidsVector
#pragma HLS STABLE variable = mem_CentroidsVector
#pragma HLS INTERFACE s_axilite port = oputCentroids_id
#pragma HLS STABLE variable = oputCentroids_id
#pragma HLS INTERFACE s_axilite port = out_topk_dis
#pragma HLS STABLE variable = out_topk_dis
#pragma HLS INTERFACE s_axilite port = numCentroids
#pragma HLS STABLE variable = numCentroids
#pragma HLS INTERFACE s_axilite port = dim
#pragma HLS STABLE variable = dim
#pragma HLS INTERFACE s_axilite port = nprobe
#pragma HLS STABLE variable = nprobe
#pragma HLS INTERFACE ap_ctrl_chain port = return

#pragma HLS cache port=mem_xqVector lines=32 depth=128 
#pragma HLS cache port=mem_CentroidsVector lines=32 depth=128 

    static hls::stream<int> local_centroidsVector("local_centroidsVector");
    static hls::stream<int> outL2dis_1("outL2dis_1");
    static hls::stream<int> outL2dis_2("outL2dis_2");
    static hls::stream<int> sort_tmpStream("sort_tmpStream");
    static hls::stream<int> local_oputCentroids_id("local_oputCentroids_id");
    static hls::stream<int> outL2dis("outL2dis");
    static hls::stream<int> outSelectedL2dis("outSelectedL2dis");

    int local_xqVector[MAX_VECTOR_DIM];

#pragma HLS dataflow off
    read_xq_vector(mem_xqVector, local_xqVector, dim);

#pragma HLS dataflow
    read_base_vector(mem_CentroidsVector, local_centroidsVector, numCentroids, dim);
    compute_l2(local_xqVector, local_centroidsVector, numCentroids, dim, outL2dis_1, outL2dis_2);
    parallel_sort(outL2dis_1, outL2dis_2, sort_tmpStream, outL2dis, numCentroids);
    select_topK(sort_tmpStream, local_oputCentroids_id, outL2dis, outSelectedL2dis, numCentroids, nprobe);
    write_topK_vectors_id_distance(oputCentroids_id, out_topk_dis, outSelectedL2dis, local_oputCentroids_id, nprobe);
}
}

