#include "kernel_manager.h"
// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_queue.h"
#include "rapidcsv.h"
#include <unistd.h>
#include <omp.h>
#include <atomic>
#include <cstring>

searchTopK_KernelManager::searchTopK_KernelManager(std::string id, int topk, int vector_dim, std::string dataset_name, int xb_vector_features_fd, int* cluster_nav_data, int* distribute_topK_ids, int* distribute_topK_dis, rapidcsv::Document& invlist_index_csv, xrt::device& fpga_device, xrt::uuid& kernel_uuid) : TOPK(topk), VECTOR_DIM(vector_dim), invlist_index_csv(invlist_index_csv), dataset_name(dataset_name), xb_vector_features_fd(xb_vector_features_fd), distribute_topK_ids(distribute_topK_ids), distribute_topK_dis(distribute_topK_dis), cluster_nav_data(cluster_nav_data)
{

    xrt::bo::flags p2p_flag = xrt::bo::flags::p2p;

    std::string kernel_name = std::string("search_topK_vec_top") + ":{" + "search_topK_vec_top_" + id + "}";
    kernel = xrt::kernel(fpga_device, kernel_uuid, kernel_name.c_str());

    search_topK_vec_XqVector = xrt::bo(fpga_device, VECTOR_DIM * sizeof(int), kernel.group_id(0));


    // 每次都要将全量数据传输至FPGA上，FPGA内存有限，会限制并行kernel数量？
    search_topK_vec_xbVector_ping = xrt::bo(fpga_device, MAX_SEARCHTOPK_VECS_NUM * VECTOR_DIM * sizeof(int), p2p_flag, kernel.group_id(1)); // Xb数据读取量大，写入量大，为重点优化目标，故pingpong分开memory bank
    search_topK_vec_xbVector_pong = xrt::bo(fpga_device, MAX_SEARCHTOPK_VECS_NUM * VECTOR_DIM * sizeof(int), p2p_flag, kernel.group_id(2));

    search_topK_vec_out_topK_Vector_ping = xrt::bo(fpga_device, TOPK * sizeof(int), kernel.group_id(0));
    search_topK_vec_out_topK_Vector_pong = xrt::bo(fpga_device, TOPK * sizeof(int), kernel.group_id(0));

    search_topK_vec_out_topK_dis_ping = xrt::bo(fpga_device, TOPK * sizeof(int), kernel.group_id(0));
    search_topK_vec_out_topK_dis_pong = xrt::bo(fpga_device, TOPK * sizeof(int), kernel.group_id(0));

    // 内存映射
    search_topK_vec_XqVector_map = search_topK_vec_XqVector.map<int *>();
    search_topK_vec_xbVector_ping_map = search_topK_vec_xbVector_ping.map<int *>();
    search_topK_vec_xbVector_pong_map = search_topK_vec_xbVector_pong.map<int *>();
    search_topK_vec_out_topK_Vector_ping_map = search_topK_vec_out_topK_Vector_ping.map<int *>();
    search_topK_vec_out_topK_Vector_pong_map = search_topK_vec_out_topK_Vector_pong.map<int *>();
    search_topK_vec_out_topK_dis_ping_map = search_topK_vec_out_topK_dis_ping.map<int *>();
    search_topK_vec_out_topK_dis_pong_map = search_topK_vec_out_topK_dis_pong.map<int *>();
}

void searchTopK_KernelManager::_Init_States()
{
    xq_ready = false;
    xq_reading = false;
    xb_ping_ready = false;
    xb_ping_reading = false;
    xb_pong_ready = false;
    xb_pong_reading = false;
    result_id_ping_ready = false;
    result_id_ping_writing = false;
    result_id_pong_ready = false;
    result_id_pong_writing = false;
    result_dis_ping_ready = false;
    result_dis_ping_writing = false;
    result_dis_pong_ready = false;
    result_dis_pong_writing = false;
    last_task_signal = false;
    kernel_running = false;
    kernel_running_ping = false;
    last_kernel_submit_ping = false;
    next_task_ind = 0;
    curr_ping_ind = -1;
    curr_pong_ind = -1;
}

void searchTopK_KernelManager::_Set_Manager_State(searchTopK_KernelManagerState new_state)
{
    manager_state_code = static_cast<int>(new_state);
}

searchTopK_KernelManagerState searchTopK_KernelManager::_Get_Manager_State()
{
    return static_cast<searchTopK_KernelManagerState>(manager_state_code.load());
}

void searchTopK_KernelManager::Init(const std::vector<int>& tasks, const std::vector<int>& original_indexs, int* xq_vec_int)
{
    _Init_States();
    cluster_ids = tasks;
    this->original_indexs = original_indexs;
    // 数据量较小，同步复制到FPGA内存中
    memcpy((void *)search_topK_vec_XqVector_map, (void *)xq_vec_int, VECTOR_DIM * sizeof(int));
    search_topK_vec_XqVector.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    _Set_Manager_State(searchTopK_KernelManagerState::EXCUTING);
}

void searchTopK_KernelManager::Next()
{
    if (_Get_Manager_State() != searchTopK_KernelManagerState::EXCUTING)
    {
        throw "ERROR! Manager Is Not Excuting!";
    }

    // 判断是否已完成所有计算
    if (last_task_signal)
    {
        last_task_signal = false;
        _Set_Manager_State(searchTopK_KernelManagerState::IDLE);
        return;
    }

    // 缓存当前任务状态，因为后续在本函数中会更改任务状态！
    int next_task_ind_save = next_task_ind;
    int curr_ping_ind_save = curr_ping_ind;
    int curr_pong_ind_save = curr_pong_ind;

    // 检查是否可以读入新的数据Xb？
    // 检查是否可以取走产生的结果Result？
    // 检查是否可以启动计算？

    // Load

    if (next_task_ind < cluster_ids.size())
    {
        // Ping
        if (!xb_ping_ready && !xb_ping_reading)
        {
            xb_ping_reading = true;
            // 分配任务
            curr_ping_ind = next_task_ind++;

            int tmp_curr_ping_ind = curr_ping_ind;
            // 提交复制任务到队列中
            search_topK_vec_xbVector_ping_queue.enqueue([tmp_curr_ping_ind, this] {
                std::vector<int> vec_ids = invlist_index_csv.GetRow<int>(cluster_ids[tmp_curr_ping_ind]);
                search_topK_vec_xbVector_ping_num = vec_ids.size();

                std::cout << "Invlist Vector Num:" << search_topK_vec_xbVector_ping_num << std::endl;

                std::cout << "Before Ping Xb Copy! Task Ind:" << tmp_curr_ping_ind << "\n";
                // _Read_Xb(search_topK_vec_xbVector_ping_map, xb_vector_features_mmap, vec_ids);
                _Read_Xb_reorg(search_topK_vec_xbVector_ping_map, xb_vector_features_fd, search_topK_vec_xbVector_ping_num, cluster_nav_data[cluster_ids[tmp_curr_ping_ind]]);
                std::cout << "Ping Xb Copy Complete! Task Ind:" << tmp_curr_ping_ind << "\n";
                xb_ping_ready = true;
            });
        }
    }
    if (next_task_ind < cluster_ids.size())
    {
        // Pong
        if (!xb_pong_ready && !xb_pong_reading && !result_id_pong_ready)
        {
            xb_pong_reading = true;
            // 分配任务
            curr_pong_ind = next_task_ind++;

            int tmp_curr_pong_ind = curr_pong_ind;

            // 提交复制任务到队列中
            search_topK_vec_xbVector_pong_queue.enqueue([tmp_curr_pong_ind, this] {
                std::vector<int> vec_ids = invlist_index_csv.GetRow<int>(cluster_ids[tmp_curr_pong_ind]);
                search_topK_vec_xbVector_pong_num = vec_ids.size();
                std::cout << "Before Pong Xb Copy! Task Ind:" << tmp_curr_pong_ind << "\n";
                // _Read_Xb(search_topK_vec_xbVector_pong_map, xb_vector_features_mmap, vec_ids);
                _Read_Xb_reorg(search_topK_vec_xbVector_pong_map, xb_vector_features_fd, search_topK_vec_xbVector_pong_num, cluster_nav_data[cluster_ids[tmp_curr_pong_ind]]);
                std::cout << "Pong Xb Copy Complete! Task Ind:" << tmp_curr_pong_ind << "\n";
                xb_pong_ready = true;
            });
        }
    }
    // Compute


    if (kernel_running)
    {
        // kernel正在计算ping或pong的数据, 若另一块缓存中的数据可用, 可提前生成run提交进队列
        if (kernel_running_ping)
        {
            if (last_kernel_submit_ping)
            {
                // 尝试提交pong的数据
                if (xb_pong_ready && xb_pong_reading && !result_dis_pong_writing && !result_dis_pong_ready && !result_id_pong_ready && !result_id_pong_writing)
                {
                    xb_pong_reading = false;
                    result_dis_pong_writing = true;
                    result_id_pong_writing = true;
                    last_kernel_submit_ping = false;
                    kernel_queue.enqueue([this]{
                        kernel_running_ping = false;
                        kernel_running = true;

                        xrt::run run = xrt::run(kernel);
                        run.set_arg(0, search_topK_vec_XqVector);
                        run.set_arg(1, search_topK_vec_xbVector_pong);
                        run.set_arg(2, search_topK_vec_out_topK_Vector_pong);
                        run.set_arg(3, search_topK_vec_xbVector_pong_num);
                        run.set_arg(4, VECTOR_DIM);
                        run.set_arg(5, TOPK);
                        run.set_arg(6, search_topK_vec_out_topK_dis_pong);

                        run.start();
                        run.wait();
                        xb_pong_ready = false;
                        result_id_pong_ready = true;
                        result_dis_pong_ready = true;
                        kernel_running = false;
                    });
                }
            }
        }
        else if (!kernel_running_ping)
        {
            if (!last_kernel_submit_ping)
            {
                // 尝试提交ping的数据
                if (xb_ping_ready && xb_ping_reading && !result_dis_ping_writing && !result_dis_ping_ready && !result_id_ping_ready && !result_id_ping_writing)
                {
                    xb_ping_reading = false;
                    result_dis_ping_writing = true;
                    result_id_ping_writing = true;

                    last_kernel_submit_ping = true;
                    kernel_queue.enqueue([this]{
                        kernel_running_ping = true;
                        kernel_running = true;

                        xrt::run run = xrt::run(kernel);
                        run.set_arg(0, search_topK_vec_XqVector);
                        run.set_arg(1, search_topK_vec_xbVector_ping);
                        run.set_arg(2, search_topK_vec_out_topK_Vector_ping);
                        run.set_arg(3, search_topK_vec_xbVector_ping_num);
                        run.set_arg(4, VECTOR_DIM);
                        run.set_arg(5, TOPK);
                        run.set_arg(6, search_topK_vec_out_topK_dis_ping);

                        run.start();
                        run.wait();
                        xb_ping_ready = false;
                        result_id_ping_ready = true;
                        result_dis_ping_ready = true;
                        kernel_running = false;
                    });
                }
            }
        }
    }
    else
    {
        // kernel是空闲的, 则ping pong哪个数据准备好了用哪个
        if (xb_ping_ready && xb_ping_reading && !result_dis_ping_writing && !result_dis_ping_ready && !result_id_ping_ready && !result_id_ping_writing)
        {
            xb_ping_reading = false;
            result_dis_ping_writing = true;
            result_id_ping_writing = true;
            last_kernel_submit_ping = true;
            kernel_queue.enqueue([this]{
                kernel_running_ping = true;
                kernel_running = true;

                xrt::run run = xrt::run(kernel);
                run.set_arg(0, search_topK_vec_XqVector);
                run.set_arg(1, search_topK_vec_xbVector_ping);
                run.set_arg(2, search_topK_vec_out_topK_Vector_ping);
                run.set_arg(3, search_topK_vec_xbVector_ping_num);
                run.set_arg(4, VECTOR_DIM);
                run.set_arg(5, TOPK);
                run.set_arg(6, search_topK_vec_out_topK_dis_ping);

                run.start();
                run.wait();

                xb_ping_ready = false;
                result_id_ping_ready = true;
                result_dis_ping_ready = true;
                kernel_running = false;
            });
        }
        else if (xb_pong_ready && xb_pong_reading && !result_dis_pong_writing && !result_dis_pong_ready && !result_id_pong_ready && !result_id_pong_writing)
        {
            xb_pong_reading = false;
            result_dis_pong_writing = true;
            result_id_pong_writing = true;
            last_kernel_submit_ping = false;
            kernel_queue.enqueue([this]{

                xrt::run run = xrt::run(kernel);
                run.set_arg(0, search_topK_vec_XqVector);
                run.set_arg(1, search_topK_vec_xbVector_pong);
                run.set_arg(2, search_topK_vec_out_topK_Vector_pong);
                run.set_arg(3, search_topK_vec_xbVector_pong_num);
                run.set_arg(4, VECTOR_DIM);
                run.set_arg(5, TOPK);
                run.set_arg(6, search_topK_vec_out_topK_dis_pong);

                kernel_running_ping = false;
                kernel_running = true;
                run.start();
                run.wait();
                xb_pong_ready = false;
                result_id_pong_ready = true;
                result_dis_pong_ready = true;
                kernel_running = false;
            });
        }
    }
    // Store

    // Ping
    if (curr_ping_ind_save >= 0 && curr_ping_ind_save < cluster_ids.size())
    {
        // ID
        if (result_id_ping_writing && result_id_ping_ready)
        {
            result_id_ping_writing = false;

            // Last Check
            if (curr_ping_ind_save == cluster_ids.size()-1)
            {
                search_topK_vec_out_topK_Vector_ping_queue.enqueue([this, curr_ping_ind_save]{
                    search_topK_vec_out_topK_Vector_ping.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    // 在此处将BO中的内容复制到下一阶段的数据缓存里
                    memcpy(distribute_topK_ids + original_indexs[curr_ping_ind_save] * TOPK, search_topK_vec_out_topK_Vector_ping_map, TOPK * sizeof(int));
                    result_id_ping_ready = false;
                    if (result_dis_ping_ready == false) // Last Check
                        last_task_signal = true;
                });
            }
            else
            {
                search_topK_vec_out_topK_Vector_ping_queue.enqueue([this, curr_ping_ind_save]{
                    search_topK_vec_out_topK_Vector_ping.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    // 在此处将BO中的内容复制到下一阶段的数据缓存里
                    memcpy(distribute_topK_ids + original_indexs[curr_ping_ind_save] * TOPK, search_topK_vec_out_topK_Vector_ping_map, TOPK * sizeof(int));
                    result_id_ping_ready = false;
                });
            }
        }
        // Dis
        if (result_dis_ping_writing && result_dis_ping_ready)
        {
            result_dis_ping_writing = false;

            // Last Check
            if (curr_ping_ind_save == cluster_ids.size()-1)
            {
                search_topK_vec_out_topK_dis_ping_queue.enqueue([this, curr_ping_ind_save]{
                    search_topK_vec_out_topK_dis_ping.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    // 在此处将BO中的内容复制到下一阶段的数据缓存里
                    memcpy(distribute_topK_dis + original_indexs[curr_ping_ind_save] * TOPK, search_topK_vec_out_topK_dis_ping_map, TOPK * sizeof(int));
                    result_dis_ping_ready = false;
                    if (result_id_ping_ready == false) // Last Check
                        last_task_signal = true;
                });
            }
            else
            {
                search_topK_vec_out_topK_dis_ping_queue.enqueue([this, curr_ping_ind_save]{
                    search_topK_vec_out_topK_dis_ping.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    // 在此处将BO中的内容复制到下一阶段的数据缓存里
                    memcpy(distribute_topK_dis + original_indexs[curr_ping_ind_save] * TOPK, search_topK_vec_out_topK_dis_ping_map, TOPK * sizeof(int));
                    result_dis_ping_ready = false;
                });
            }
        }
    }
    // Pong
    if (curr_pong_ind_save >= 0 && curr_pong_ind_save < cluster_ids.size())
    {
        // ID
        if (result_id_pong_writing && result_id_pong_ready)
        {
            result_id_pong_writing = false;
            
            // Last Check
            if (curr_pong_ind_save == cluster_ids.size()-1)
            {
                search_topK_vec_out_topK_Vector_pong_queue.enqueue([this, curr_pong_ind_save]{
                    search_topK_vec_out_topK_Vector_pong.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    // 在此处将BO中的内容复制到下一阶段的数据缓存里
                    memcpy(distribute_topK_ids + original_indexs[curr_pong_ind_save] * TOPK, search_topK_vec_out_topK_Vector_pong_map, TOPK * sizeof(int));
                    result_id_pong_ready = false;
                    if (result_dis_pong_ready == false)
                        last_task_signal = true;
                });
            }
            else
            {
                search_topK_vec_out_topK_Vector_pong_queue.enqueue([this, curr_pong_ind_save]{
                    search_topK_vec_out_topK_Vector_pong.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    // 在此处将BO中的内容复制到下一阶段的数据缓存里
                    memcpy(distribute_topK_ids + original_indexs[curr_pong_ind_save] * TOPK, search_topK_vec_out_topK_Vector_pong_map, TOPK * sizeof(int));
                    result_id_pong_ready = false;
                });
            }
        }
        // Dis
        if (result_dis_pong_writing && result_dis_pong_ready)
        {
            result_dis_pong_writing = false;

            // Last Check
            if (curr_pong_ind_save == cluster_ids.size()-1)
            {
                search_topK_vec_out_topK_dis_pong_queue.enqueue([this, curr_pong_ind_save]{
                    search_topK_vec_out_topK_dis_pong.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    // 在此处将BO中的内容复制到下一阶段的数据缓存里
                    memcpy(distribute_topK_dis + original_indexs[curr_pong_ind_save] * TOPK, search_topK_vec_out_topK_dis_pong_map, TOPK * sizeof(int));
                    result_dis_pong_ready = false;
                    if (result_id_pong_ready == false)
                        last_task_signal = true;
                });
            }
            else
            {
                search_topK_vec_out_topK_dis_pong_queue.enqueue([this, curr_pong_ind_save]{
                    search_topK_vec_out_topK_dis_pong.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    // 在此处将BO中的内容复制到下一阶段的数据缓存里
                    memcpy(distribute_topK_dis + original_indexs[curr_pong_ind_save] * TOPK, search_topK_vec_out_topK_dis_pong_map, TOPK * sizeof(int));
                    result_dis_pong_ready = false;
                });
            }
        }
    }
}

void searchTopK_KernelManager::_Read_Xb(int *fpga_mem, int *ssd_mmap, const std::vector<int> invlist_item)
{
    std::cout << "In Read Xb!\n";

    if (dataset_name == std::string("sift200M"))
    {
#pragma omp parallel for num_threads(64)
        for (int i = 0; i < invlist_item.size(); i++)
        {
            memcpy((void *)(fpga_mem + i * VECTOR_DIM), (void *)(ssd_mmap + invlist_item[i]), VECTOR_DIM * sizeof(int));
        }
    }
    else
    {
#pragma omp parallel for num_threads(64)
        for (int i = 0; i < invlist_item.size(); i++)
        {
            memcpy((void *)(fpga_mem + i * VECTOR_DIM), (void *)(ssd_mmap + invlist_item[i] * VECTOR_DIM), VECTOR_DIM * sizeof(int));
        }
    }
}

void searchTopK_KernelManager::_Read_Xb_reorg(int *fpga_mem, int xb_fd, int vector_num, int nav_point)
{
    std::cout << "In Read Xb!\n";

    pread(xb_fd, fpga_mem, vector_num * VECTOR_DIM * sizeof(int), nav_point * VECTOR_DIM * sizeof(int));
}

bool searchTopK_KernelManager::isEnd()
{
    return _Get_Manager_State() == searchTopK_KernelManagerState::IDLE;
}