// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <atomic>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <pq_flash_index.h>
#include <set>
#include <string.h>
#include <time.h>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <unistd.h>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP true

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

void print_stats(std::string category, std::vector<float> percentiles,
                 std::vector<float> results) {
  diskann::cout << std::setw(20) << category << ": " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    diskann::cout << std::setw(8) << percentiles[s] << "%";
  }
  diskann::cout << std::endl;
  diskann::cout << std::setw(22) << " " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    diskann::cout << std::setw(9) << results[s];
  }
  diskann::cout << std::endl;
}

template<typename T>
int search_disk_index(int argc, char** argv) {
  // load query bin
  T*                query = nullptr;
  unsigned*         gt_ids = nullptr;
  float*            gt_dists = nullptr;
  uint32_t*         tags = nullptr;
  size_t            query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  std::vector<_u64> Lvec;

  int         index = 2;
  std::string index_prefix_path(argv[index++]);
  std::string warmup_query_file = index_prefix_path + "_sample_data.bin";
  bool        single_file_index = std::atoi(argv[index++]) != 0;
  bool        tags_flag = std::atoi(argv[index++]) != 0;
  _u64        num_nodes_to_cache = std::atoi(argv[index++]);
  _u32        num_threads = std::atoi(argv[index++]);
  _u32        beamwidth = std::atoi(argv[index++]);
  std::string query_bin(argv[index++]);
  std::string truthset_bin(argv[index++]);
  _u64        recall_at = std::atoi(argv[index++]);
  std::string result_output_prefix(argv[index++]);
  std::string dist_metric(argv[index++]);
  std::string save_log_filename(argv[index++]);
  std::string dataset_name(argv[index++]);

  diskann::Metric m =
      dist_metric == "cosine" ? diskann::Metric::COSINE : diskann::Metric::L2;
  if (dist_metric != "l2" && m == diskann::Metric::L2) {
    diskann::cout << "Unknown distance metric: " << dist_metric
                  << ". Using default(L2) instead." << std::endl;
  }

  std::string disk_index_tag_file = index_prefix_path + "_disk.index.tags";

  bool calc_recall_flag = false;

  for (int ctr = index; ctr < argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    diskann::cout
        << "No valid Lsearch found. Lsearch must be at least recall_at"
        << std::endl;
    return -1;
  }

  // logger
  std::string                   csv_full_path = log_path + save_log_filename;
  CSV_Vector_Search_Perf_logger eva_logger(csv_full_path.c_str());

  diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    diskann::cout << "beamwidth to be optimized for each L value" << std::endl;
  else
    diskann::cout << " beamwidth: " << beamwidth << std::endl;

  diskann::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                               query_aligned_dim);

  if (file_exists(truthset_bin)) {
    diskann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim,
                           &tags);
    if (gt_num != query_num) {
      diskann::cout
          << "Error. Mismatch in number of queries and ground truth data"
          << std::endl;
    }
    calc_recall_flag = true;
  }

  std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
  reader.reset(new WindowsAlignedFileReader());
#else
  reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
  reader.reset(new LinuxAlignedFileReader());
//  reader.reset(new diskann::MemAlignedFileReader());
#endif

  std::unique_ptr<diskann::PQFlashIndex<T>> _pFlashIndex(
      new diskann::PQFlashIndex<T>(m, reader, single_file_index,
                                   tags_flag));  // no tags support yet.

  int res = _pFlashIndex->load(index_prefix_path.c_str(), num_threads);
  if (res != 0) {
    return res;
  }

  if (tags_flag) {
    tsl::robin_set<_u32> active_tags;
    _pFlashIndex->get_active_tags(active_tags);

    diskann::cout << "Loaded " << active_tags.size()
                  << " tags from index for recall measurement." << std::endl;
  } else {
    diskann::cout << "Not loading tags since they are disabled." << std::endl;
  }

  // cache bfs levels
  std::vector<uint32_t> node_list;
  diskann::cout << "Caching " << num_nodes_to_cache
                << " BFS nodes around medoid(s)" << std::endl;
  _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
  //_pFlashIndex->generate_cache_list_from_sample_queries(
  //   warmup_query_file, 15, 6, num_nodes_to_cache, num_threads, node_list);
  _pFlashIndex->load_cache_list(node_list);
  node_list.clear();
  node_list.shrink_to_fit();

  omp_set_num_threads(num_threads);

  uint64_t warmup_L;
  warmup_L = 20;
  uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
  T*       warmup = nullptr;
  if (WARMUP) {
    if (file_exists(warmup_query_file)) {
      diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num,
                                   warmup_dim, warmup_aligned_dim);
    } else {
      warmup_num = (std::min)((_u32) 150000, (_u32) 15000 * num_threads);
      warmup_dim = query_dim;
      warmup_aligned_dim = query_aligned_dim;
      diskann::alloc_aligned(((void**) &warmup),
                             warmup_num * warmup_aligned_dim * sizeof(T),
                             8 * sizeof(T));
      std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
      std::random_device              rd;
      std::mt19937                    gen(rd());
      std::uniform_int_distribution<> dis(-128, 127);
      for (uint32_t i = 0; i < warmup_num; i++) {
        for (uint32_t d = 0; d < warmup_dim; d++) {
          warmup[i * warmup_aligned_dim + d] = (T) dis(gen);
        }
      }
    }
    diskann::cout << "Warming up index... " << std::flush;
    std::vector<uint64_t> warmup_result_tags_64(warmup_num, 0);
    std::vector<float>    warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) warmup_num; i++) {
      _pFlashIndex->cached_beam_search_ids(
          warmup + (i * warmup_aligned_dim), 1, warmup_L,
          warmup_result_tags_64.data() + (i * 1),
          warmup_result_dists.data() + (i * 1), (uint64_t) 4);
    }
    diskann::cout << "..done" << std::endl;
  }

  diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  diskann::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth"
                << std::setw(16) << "QPS" << std::setw(16) << "Mean Latency"
                << std::setw(16) << "99.9 Latency" << std::setw(16)
                << "Mean IOs" << std::setw(16) << "CPU (s)";
  if (calc_recall_flag) {
    diskann::cout << std::setw(16) << recall_string << std::endl;
  } else
    diskann::cout << std::endl;
  diskann::cout
      << "==============================================================="
         "==========================================="
      << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());

  uint32_t optimized_beamwidth = 2;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];

    if (beamwidth <= 0) {
      //    diskann::cout<<"Tuning beamwidth.." << std::endl;
      optimized_beamwidth =
          optimize_beamwidth(_pFlashIndex, warmup, warmup_num,
                             warmup_aligned_dim, (_u32) L, optimized_beamwidth);
    } else
      optimized_beamwidth = beamwidth;

    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);
    query_result_tags[test_id].resize(recall_at * query_num);

    diskann::QueryStats* stats = new diskann::QueryStats[query_num];

    std::vector<uint64_t> query_result_tags_64(recall_at * query_num);
    std::vector<uint32_t> query_result_tags_32(recall_at * query_num);

    double init_cpu_usage = get_cpu_usage();

    auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) query_num; i++) {
      _pFlashIndex->cached_beam_search(
          query + (i * query_aligned_dim), (uint64_t) recall_at, (uint64_t) L,
          query_result_tags_32.data() + (i * recall_at),
          query_result_dists[test_id].data() + (i * recall_at),
          (uint64_t) optimized_beamwidth, stats + i);
    }
    auto e = std::chrono::high_resolution_clock::now();

    double final_cpu_usage = get_cpu_usage() * 100;

    std::chrono::duration<double> diff = e - s;
    float                         qps =
        (float) ((1.0 * (double) query_num) / (1.0 * (double) diff.count()));

    diskann::convert_types<uint32_t, uint32_t>(
        query_result_tags_32.data(), query_result_tags[test_id].data(),
        (size_t) query_num, (size_t) recall_at);

    float mean_latency = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    /*    float latency_90 = (float) diskann::get_percentile_stats(
            stats, query_num, 0.900,
            [](const diskann::QueryStats& stats) { return stats.total_us; });

        float latency_95 = (float) diskann::get_percentile_stats(
            stats, query_num, 0.950,
            [](const diskann::QueryStats& stats) { return stats.total_us; });
    */
    float latency_999 = (float) diskann::get_percentile_stats(
        stats, query_num, 0.999f,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    float mean_ios = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.n_ios; });

    float mean_cpuus = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.cpu_us; });
    delete[] stats;

    float recall = 0;
    if (calc_recall_flag) {
      recall = (float) diskann::calculate_recall(
          (_u32) query_num, tags, gt_dists, (_u32) gt_dim,
          query_result_tags[test_id].data(), (_u32) recall_at,
          (_u32) recall_at);
    }

    diskann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth
                  << std::setw(16) << qps << std::setw(16) << mean_latency
                  << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                  << std::setw(16) << mean_cpuus;
    if (calc_recall_flag) {
      diskann::cout << std::setw(16) << recall << std::endl;
    }

    eva_logger.write_to_csv(dataset_name.c_str(), L, "DiskANN", "cpu",
                            mean_latency, latency_999, qps, mean_cpuus);
  }
  std::this_thread::sleep_for(std::chrono::seconds(10));

  diskann::cout << "Done searching. Now saving results " << std::endl;
  _u64 test_id = 0;
  // for (auto L : Lvec) {
  //   std::string cur_result_path =
  //       result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
  //   diskann::save_bin<_u32>(cur_result_path,
  //   query_result_ids[test_id].data(),
  //                           query_num, recall_at);

  //   cur_result_path =
  //       result_output_prefix + "_" + std::to_string(L) + "_tags_uint32.bin";
  //   diskann::save_bin<_u32>(cur_result_path,
  //   query_result_tags[test_id].data(),
  //                           query_num, recall_at);
  //   cur_result_path =
  //       result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
  //   diskann::save_bin<float>(cur_result_path,
  //                            query_result_dists[test_id++].data(), query_num,
  //                            recall_at);
  // }

  diskann::aligned_free(query);
  if (warmup != nullptr)
    diskann::aligned_free(warmup);
  delete[] gt_ids;
  delete[] gt_dists;
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 14) {
    diskann::cout
        << "Usage: " << argv[0]
        << " <index_type (float/int8/uint8)>  <index_prefix_path>"
           " <single_file_index(0/1)> <tags(0/1) "
           " <num_nodes_to_cache>  <num_threads>  <beamwidth (use 0 to "
           "optimize internally)> "
           " <query_file.bin>  <truthset.bin (use \"null\" for none)> "
           " <K>  <result_output_prefix> <similarity (cosine/l2)> "
           " <L1>  [L2] etc.  See README for more information on parameters."
        << std::endl;
    exit(-1);
  }

  diskann::cout << "Attach  debugger and press a key" << std::endl;
  /*  char x;
    std::cin >> x;
    */

  if (std::string(argv[1]) == std::string("float"))
    search_disk_index<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    search_disk_index<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    search_disk_index<uint8_t>(argc, argv);
  else
    diskann::cout << "Unsupported index type. Use float or int8 or uint8"
                  << std::endl;
}
