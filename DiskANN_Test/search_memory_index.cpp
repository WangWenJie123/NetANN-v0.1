// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <omp.h>
#include <set>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <unistd.h>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "aux_utils.h"
#include "index.h"
#include "memory_mapper.h"
#include "utils.h"

std::string log_path =
    "/home/wwj/Vector_DB_Acceleration/DiskANN_Baseline/datasets/eva_logs/";

// 定义一个cpu occupy的结构体，用来存放CPU的信息
typedef struct CPUPACKED {
  char         name[20];  //定义一个char类型的数组名name有20个元素
  unsigned int user;      //定义一个无符号的int类型的user
  unsigned int nice;      //定义一个无符号的int类型的nice
  unsigned int system;    //定义一个无符号的int类型的system
  unsigned int idle;      //定义一个无符号的int类型的idle
  unsigned int lowait;
  unsigned int irq;
  unsigned int softirq;
} CPU_OCCUPY;

// 该函数，利用上述公式。计算出两段时间的之间的CPU占用率
// 输入为：当前时刻和上一采样时刻cpu的信息
double getCpuUse(CPU_OCCUPY* o, CPU_OCCUPY* n) {
  std::cout << "test" << std::endl;
  unsigned long od, nd;
  od =
      (unsigned long) (o->user + o->nice + o->system + o->idle + o->lowait +
                       o->irq +
                       o->softirq);  //第一次(用户+优先级+系统+空闲)的时间再赋给od
  nd =
      (unsigned long) (n->user + n->nice + n->system + n->idle + n->lowait +
                       n->irq +
                       n->softirq);  //第二次(用户+优先级+系统+空闲)的时间再赋给od
  double sum = nd - od;
  double idle = n->idle - o->idle;
  return (sum - idle) / sum;
}

class CSV_Vector_Search_Perf_logger {
 public:
  CSV_Vector_Search_Perf_logger(const char* logger_file) {
    file = std::ofstream(logger_file, std::ios::app);
    file << "datasets" << ',' << "nprobe" << ',' << "index_type" << ','
         << "processor" << ',' << "search_latency/ms" << ','
         << "99.9_latency/ms" << ',' << "throughput/QPS" << ',' << "cpu_usage/%"
         << std::endl;
  }
  ~CSV_Vector_Search_Perf_logger() {
    file.close();
  }

 public:
  void write_to_csv(const char* datasets, uint64_t nprobe, char* index_type,
                    char* processor, double search_latency, double latency_99,
                    float search_thput, double cpu_usage) {
    file << datasets << ',' << std::to_string(nprobe) << ',' << index_type
         << ',' << processor << ',' << std::to_string(search_latency) << ','
         << std::to_string(latency_99) << ',' << std::to_string(search_thput)
         << ',' << std::to_string(cpu_usage) << std::endl;
  }

 private:
  std::ofstream file;
};

double get_cpu_usage() {
  FILE* fd;        // 定义打开文件的指针
  char buff[256];  // 定义个数组，用来存放从文件中读取CPU的信息
  CPU_OCCUPY cpu_occupy, old_cpu_occupy;
  double     cpu_use = 0;

  fd = fopen("/proc/stat", "r");

  if (fd != NULL) {
    // 读取第一行的信息，cpu整体信息
    fgets(buff, sizeof(buff), fd);
    if (strstr(buff, "cpu") !=
        NULL)  // 返回与"cpu"在buff中的地址，如果没有，返回空指针
    {
      // 从字符串格式化输出
      sscanf(buff, "%s %u %u %u %u %u %u %u", cpu_occupy.name, &cpu_occupy.user,
             &cpu_occupy.nice, &cpu_occupy.system, &cpu_occupy.idle,
             &cpu_occupy.lowait, &cpu_occupy.irq, &cpu_occupy.softirq);
      // cpu的占用率 =
      // （当前时刻的任务占用cpu总时间-前一时刻的任务占用cpu总时间）/ （当前时刻
      // - 前一时刻的总时间）
      cpu_use = getCpuUse(&old_cpu_occupy, &cpu_occupy);
    }
  }

  fclose(fd);
  return cpu_use;
}

template<typename T>
int search_memory_index(int argc, char** argv) {
  T*                query = nullptr;
  unsigned*         gt_ids = nullptr;
  unsigned*         gt_tags = nullptr;
  float*            gt_dists = nullptr;
  size_t            query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  std::vector<_u64> Lvec;

  int arg_no = 2;

  _u64        max_points = (_u64) std::atoi(argv[arg_no++]);
  std::string memory_index_file(argv[arg_no++]);
  bool        dynamic_index = (bool) std::atoi(argv[arg_no++]);
  bool        single_index_file = (bool) std::atoi(argv[arg_no++]);
  //  std::string data_file(argv[arg_no++]);
  std::string query_bin(argv[arg_no++]);
  std::string truthset_bin(argv[arg_no++]);
  _u64        recall_at = std::atoi(argv[arg_no++]);
  std::string result_output_prefix(argv[arg_no++]);
  _u32        num_threads = (_u32) std::atoi(argv[arg_no++]);
  std::string distance_metric(argv[arg_no++]);
  std::string save_log_filename(argv[arg_no++]);
  std::string dataset_name(argv[arg_no++]);

  diskann::Metric m = (distance_metric == "cosine" ? diskann::Metric::COSINE
                                                   : diskann::Metric::L2);

  if (distance_metric != "l2" && m == diskann::Metric::L2) {
    std::cout << "Not processing metric: '" << distance_metric
              << "'. Setting to default (L2)" << std::endl;
  }

  bool calc_recall_flag = false;

  for (int ctr = arg_no; ctr < argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    diskann::cout
        << "No valid Lsearch found. Lsearch must be at least recall_at."
        << std::endl;
    return -1;
  }

  // logger
  std::string                   csv_full_path = log_path + save_log_filename;
  CSV_Vector_Search_Perf_logger eva_logger(csv_full_path.c_str());

  std::cout << "Running search on type " << typeid(T).name()
            << " max_points: " << max_points
            << " index file:" << memory_index_file
            << " is dynamic: " << dynamic_index
            << " is single file: " << single_index_file
            << " query file: " << query_bin
            << " truthset file: " << truthset_bin << " K: " << recall_at
            << " num threads: " << num_threads
            << " save prefix: " << result_output_prefix
            << " similarity metric: "
            << (m == diskann::Metric::COSINE ? "cosine" : "l2")
            << " first L: " << Lvec[0] << std::endl;

  diskann::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                               query_aligned_dim);

  if (file_exists(truthset_bin)) {
    diskann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim,
                           &gt_tags);
    if (gt_num != query_num) {
      diskann::cout
          << "Error. Mismatch in number of queries and ground truth data"
          << std::endl;
    }
    calc_recall_flag = true;
  }

  diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  diskann::cout.precision(2);

  diskann::Index<T, uint32_t> index(m, query_dim, max_points, dynamic_index,
                                    single_index_file, dynamic_index);
  index.load(memory_index_file.c_str());

  tsl::robin_set<uint32_t> active_tags;
  index.get_active_tags(active_tags);

  diskann::cout << "Index loaded" << std::endl;

  diskann::Parameters paras;
  std::string         recall_string = "Recall@" + std::to_string(recall_at);
  diskann::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS "
                << std::setw(18) << "Mean Latency (ms)" << std::setw(15)
                << "99.9 Latency" << std::setw(12) << recall_string
                << std::endl;
  diskann::cout
      << "==============================================================="
         "==============="
      << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());
  std::vector<std::vector<unsigned>> query_result_tags(Lvec.size());

  std::vector<double> latency_stats(query_num, 0);

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];
    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_tags[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);

    double init_cpu_usage = get_cpu_usage();

    auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
    for (int64_t i = 0; i < (int64_t) query_num; i++) {
      auto qs = std::chrono::high_resolution_clock::now();
      if (dynamic_index)
        index.search_with_tags(
            query + i * query_aligned_dim, (uint64_t) recall_at, (_u32) L,
            query_result_ids[test_id].data() + i * recall_at,
            query_result_dists[test_id].data() + i * recall_at);
      else
        index.search(query + i * query_aligned_dim, recall_at, (_u32) L,
                     query_result_ids[test_id].data() + i * recall_at);
      auto qe = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = qe - qs;
      latency_stats[i] = diff.count() * 1000;
    }
    auto e = std::chrono::high_resolution_clock::now();

    double final_cpu_usage = get_cpu_usage() * 100;

    std::chrono::duration<double> diff = e - s;

    float qps = (float) ((double) query_num / (double) diff.count());

    float recall = 0;
    if (calc_recall_flag)

      recall = (float) diskann::calculate_recall(
          (_u32) query_num, gt_tags, gt_dists, (_u32) gt_dim,
          query_result_ids[test_id].data(), (_u32) recall_at, (_u32) recall_at,
          active_tags);

    std::sort(latency_stats.begin(), latency_stats.end());
    double mean_latency = 0;
    for (uint64_t q = 0; q < query_num; q++) {
      mean_latency += latency_stats[q];
    }
    mean_latency /= (double) query_num;

    diskann::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
                  << (float) mean_latency << std::setw(15)
                  << (float) latency_stats[(_u64)(0.999 * (double) query_num)]
                  << std::setw(12) << recall << std::endl;

    eva_logger.write_to_csv(dataset_name.c_str(), L, "DiskANN", "cpu",
                            mean_latency,
                            latency_stats[(_u64)(0.999 * (double) query_num)],
                            qps, abs(final_cpu_usage));
  }

  diskann::cout << "Done searching. Now saving results " << std::endl;
  _u64 test_id = 0;

  // for (auto L : Lvec) {
  //   std::string cur_result_path =
  //       result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
  //   diskann::save_bin<_u32>(cur_result_path,
  //   query_result_ids[test_id].data(),
  //                           query_num, recall_at);
  //   test_id++;
  // }

  diskann::aligned_free(query);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 13) {
    diskann::cout
        << "Usage: " << argv[0]
        << " <index_type (float/int8/uint8)>"
           " <max_points> <memory_index_path>"
           " <dynamic_index (0/1)> <single_file_index (0/1)"
           " (must be same as that given while building the index.)>"
           " <query_file.bin>  <truthset.bin (use 'null' for none)> "
           " <K>  <result_output_prefix> <num_threads> <distance_metric"
           " (cosine/l2) case-sensitive>"
           " <L1>  [L2] etc. See README for more information on parameters. "
        << std::endl;
    exit(-1);
  }

  if (std::string(argv[1]) == std::string("int8"))
    search_memory_index<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    search_memory_index<uint8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("float"))
    search_memory_index<float>(argc, argv);
  else
    diskann::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
