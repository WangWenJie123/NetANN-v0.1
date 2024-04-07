#ifndef __MY_KERNEL_MANAGERS__
#define __MY_KERNEL_MANAGERS__
#include <string>
#include <vector>
// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_queue.h"
#include "rapidcsv.h"
#include <atomic>

#define MAX_SEARCHTOPK_VECS_NUM 31440

enum class searchTopK_KernelManagerState {
    IDLE, EXCUTING
};

class searchTopK_KernelManager
{
private:
    xrt::queue kernel_queue;

    xrt::kernel kernel;
    xrt::run kernel_run;
    xrt::bo search_topK_vec_XqVector;
    int* search_topK_vec_XqVector_map;

    // PingPong Buffer For Cluster Vectors Input
    xrt::bo search_topK_vec_xbVector_ping;
    xrt::bo search_topK_vec_xbVector_pong;
    std::atomic<int> search_topK_vec_xbVector_ping_num;
    std::atomic<int> search_topK_vec_xbVector_pong_num;
    int* search_topK_vec_xbVector_ping_map;
    int* search_topK_vec_xbVector_pong_map;
    xrt::queue search_topK_vec_xbVector_ping_queue;
    xrt::queue search_topK_vec_xbVector_pong_queue;

    // PingPong Buffer For ID Result
    xrt::bo search_topK_vec_out_topK_Vector_ping;
    xrt::bo search_topK_vec_out_topK_Vector_pong;
    int* search_topK_vec_out_topK_Vector_ping_map;
    int* search_topK_vec_out_topK_Vector_pong_map;
    xrt::queue search_topK_vec_out_topK_Vector_ping_queue;
    xrt::queue search_topK_vec_out_topK_Vector_pong_queue;

    // PingPong Buffer For Dis Result
    xrt::bo search_topK_vec_out_topK_dis_ping;
    xrt::bo search_topK_vec_out_topK_dis_pong;
    int* search_topK_vec_out_topK_dis_ping_map;
    int* search_topK_vec_out_topK_dis_pong_map;
    xrt::queue search_topK_vec_out_topK_dis_ping_queue;
    xrt::queue search_topK_vec_out_topK_dis_pong_queue;

    // Dataset Info
    int TOPK;
    int VECTOR_DIM;
    std::string dataset_name;
    int xb_vector_features_fd;
    int* cluster_nav_data;
    int* cluster_size_data;

    // Task Info
    std::vector<int> cluster_ids;
    // 似乎通过携带原始索引来确定写入位置的方式比较好
    std::vector<int> original_indexs;
    int next_task_ind = -1;
    int curr_ping_ind = -1;
    int curr_pong_ind = -1;

    // Self State

    // Manager State Code
    std::atomic<int> manager_state_code = static_cast<int>(searchTopK_KernelManagerState::IDLE); // Enum?

    // Xq
    std::atomic<bool> xq_ready = false;
    std::atomic<bool> xq_reading = false;

    // Xb
    std::atomic<bool> xb_ping_ready = false;
    std::atomic<bool> xb_ping_reading = false;

    std::atomic<bool> xb_pong_ready = false;
    std::atomic<bool> xb_pong_reading = false;

    // Result ID
    std::atomic<bool> result_id_ping_ready = false;
    std::atomic<bool> result_id_ping_writing = false;

    std::atomic<bool> result_id_pong_ready = false;
    std::atomic<bool> result_id_pong_writing = false;

    // Result Dis
    std::atomic<bool> result_dis_ping_ready = false;
    std::atomic<bool> result_dis_ping_writing = false;

    std::atomic<bool> result_dis_pong_ready = false;
    std::atomic<bool> result_dis_pong_writing = false;

    // Last Task, Set in Store Stage
    std::atomic<bool> last_task_signal = false;

    // Distribute TopK Data
    int * distribute_topK_ids;
    // 似乎通过携带原始索引来确定写入位置的方式比较好
    // 详见Task Info部分 original_indexs, 直接写入offset: originial_index * TOPK即可
    int * distribute_topK_dis;

    // Kernel
    std::atomic<bool> kernel_running = false;
    std::atomic<bool> kernel_running_ping = false;
    bool last_kernel_submit_ping = false;

    void _Init_States();
    void _Set_Manager_State(searchTopK_KernelManagerState new_state);
    searchTopK_KernelManagerState _Get_Manager_State();
    void _Read_Xb(int* fpga_mem, int* ssd_mmap, const std::vector<int> invlist_item);
    void inline _Read_Xb_reorg(int *fpga_mem, int xb_fd, int vector_num, int nav_point);

public:
    // searchTopK_KernelManager() = delete;
    searchTopK_KernelManager(std::string id, int topk, int vector_dim, std::string dataset_name, int xb_vector_features_fd, int* cluster_nav_data, int* distribute_topK_ids, int* distribute_topK_dis, int* cluster_size_data, xrt::device& fpga_device, xrt::uuid& kernel_uuid);

    void Init(const std::vector<int>& tasks, const std::vector<int>& original_indexs, int* xq_vec_int); // 控制线程调用, 设置当前kernel的任务
    void Next(); // 控制线程调用，推进任务执行
    bool isEnd();
};

#endif