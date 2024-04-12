#include <ap_int.h>
#include <stdint.h>
#include <hls_math.h>
#include <hls_stream.h>

#define MAX_NPROBE 4096
#define MAX_TOPK 100
#define MAX_VEC_NUM MAX_NPROBE * MAX_TOPK

/*
 * read vector distance from fpga memory 
*/
static void read_vector_dis(int* inL2DisList, hls::stream<int>& outL2dis_1, hls::stream<int>& outL2dis_2, unsigned int num_vecs)
{
    for(int i = 0; i < num_vecs; i++)
    {
#pragma HLS PIPELINE II = 1
        outL2dis_1.write_nb(inL2DisList[i]);
        outL2dis_2.write_nb(inL2DisList[i]);
    }
}

/*
 * parallel sorting
*/
static void parallel_sort(hls::stream<int>& outL2dis_1, hls::stream<int>& outL2dis_2, hls::stream<int>& sort_tmpStream, unsigned int num_vecs)
{
    int sort_tmp;
    int outL2dis_tmp_1, outL2dis_tmp_2;

    // parallel sorting
    for(int i = 0; i < num_vecs; i++)
    {
#pragma HLS PIPELINE II = 1
        sort_tmp = 0;
        outL2dis_1.read_nb(outL2dis_tmp_1);
        for(int j = 0; j < num_vecs; j++)
        {
#pragma HLS PIPELINE II = 1
            outL2dis_2.read_nb(outL2dis_tmp_2);
            if(outL2dis_tmp_1 < outL2dis_tmp_2) sort_tmp += 1;
        }
        sort_tmpStream.write_nb(sort_tmp);
    }
}

/*
 * select top k
*/
static void select_topK(hls::stream<int>& sort_tmpStream, hls::stream<int>& local_topk_vector_id, unsigned int num_vecs, unsigned int topk)
{
    unsigned int needed_id = num_vecs - topk;
    unsigned int out_cent_idex = 0;
    int sort_tmp;

    for(int i = 0; i < num_vecs; i++ )
    {
#pragma HLS PIPELINE II = 1
        sort_tmpStream.read_nb(sort_tmp);
        if(sort_tmp >= needed_id)
        {
            local_topk_vector_id.write_nb(i);
            out_cent_idex += 1;
            if(out_cent_idex >= topk) break;
        }
    }
}

/*
 * write topk vector id to fpga memory
*/
static void write_topk_vector_id(int* ouTopkVecId, hls::stream<int>& local_topk_vector_id, unsigned int topk)
{
    int local_topk_vector_id_tmp;

    for(int i = 0; i < topk; i++ )
    {
#pragma HLS PIPELINE II = 1
        local_topk_vector_id.read_nb(local_topk_vector_id_tmp);
        ouTopkVecId[i] = local_topk_vector_id_tmp;
    }
}

extern "C" 
{

void distribute_topK_top(int* inL2DisList, int* ouTopkVecId, unsigned int nprobe, unsigned int topk)
{
#pragma HLS INTERFACE m_axi port = inL2DisList offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = ouTopkVecId offset = slave bundle = gmem0
#pragma HLS INTERFACE s_axilite port = inL2DisList
#pragma HLS INTERFACE s_axilite port = ouTopkVecId
#pragma HLS INTERFACE s_axilite port = nprobe
#pragma HLS INTERFACE s_axilite port = topk
#pragma HLS INTERFACE ap_ctrl_chain port = return

#pragma HLS cache port=inL2DisList lines=64 depth=128 

    static hls::stream<int> outL2dis_1("outL2dis_1");
    static hls::stream<int> outL2dis_2("outL2dis_2");
    static hls::stream<int> sort_tmpStream("sort_tmpStream");
    static hls::stream<int> local_topk_vector_id("local_topk_vector_id");

    int num_vecs = nprobe * topk;

#pragma HLS dataflow
    read_vector_dis(inL2DisList, outL2dis_1, outL2dis_2, num_vecs);
    parallel_sort(outL2dis_1, outL2dis_2, sort_tmpStream, num_vecs);
    select_topK(sort_tmpStream, local_topk_vector_id, num_vecs, topk);
    write_topk_vector_id(ouTopkVecId, local_topk_vector_id, topk);
}
}
