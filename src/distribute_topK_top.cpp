#include <ap_int.h>
#include <stdint.h>
#include <hls_math.h>
#include <hls_stream.h>

#define MAX_NPROBE 4096
#define MAX_TOPK 100
#define MAX_VEC_NUM MAX_NPROBE * MAX_TOPK

extern "C" 
{

void distribute_topK_top(int* inL2DisList, int* ouTopkVecId, unsigned int nprobe, unsigned int topk)
{
#pragma HLS INTERFACE m_axi port = inL2DisList offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = ouTopkVecId offset = slave bundle = gmem1
#pragma HLS INTERFACE s_axilite port = inL2DisList
#pragma HLS INTERFACE s_axilite port = ouTopkVecId
#pragma HLS INTERFACE s_axilite port = nprobe
#pragma HLS INTERFACE s_axilite port = topk
#pragma HLS INTERFACE ap_ctrl_chain port = return

// #pragma HLS cache port=inL2DisList lines=8 depth=32 

    int sort_tmp[MAX_VEC_NUM];
// #pragma HLS bind_storage variable=sort_tmp impl=lutram
#pragma HLS ARRAY_RESHAPE variable=sort_tmp type=block factor=2 dim=1

    int num_vecs = nprobe * topk;

// #pragma HLS pipeline II=1

    // 并行全比排序，从多个聚类中心的topk中选出最终的topk个vectors
    for(int i = 0; i < num_vecs; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min = MAX_VEC_NUM max = MAX_VEC_NUM
// #pragma HLS pipeline off
#pragma HLS unroll factor=2
        for(int j = 0; j < num_vecs; j++)
        {
#pragma HLS LOOP_TRIPCOUNT min = MAX_VEC_NUM max = MAX_VEC_NUM
// #pragma HLS pipeline off
#pragma HLS unroll factor=2

            if(inL2DisList[i] < inL2DisList[j]) sort_tmp[i] += 1;
        }
    }

    unsigned int needed_id = 0;
    unsigned int out_cent_idex = 0;
    needed_id = num_vecs - topk;
    for(int i = 0; i < num_vecs; i++ )
    {
#pragma HLS LOOP_TRIPCOUNT min = MAX_VEC_NUM max = MAX_VEC_NUM
// #pragma HLS pipeline off
#pragma HLS unroll factor=2

        if(sort_tmp[i] >= needed_id)
        {
            ouTopkVecId[out_cent_idex] = i;
            out_cent_idex += 1;
            if(out_cent_idex >= topk) break;
        }
    }
}

}
