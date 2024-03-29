#include "csv_log.h"

CSV_Vector_Search_Perf_logger::CSV_Vector_Search_Perf_logger(const char* logger_file)
{
    file = std::ofstream(logger_file, std::ios::app);
    file << "nprobe" << ',' << "datasets" << ',' << "index_type" << ',' << "processor" << ',' << "search_latency/ms" << ',' << "R_1" << ',' << "R_10" << ',' << "R_100" << std::endl;
}

CSV_Vector_Search_Perf_logger::~CSV_Vector_Search_Perf_logger()
{
    file.close();
}

void CSV_Vector_Search_Perf_logger::write_to_csv(int nprobe, const char* datasets, char* index_type, char* processor, double search_latency, double search_thput, float R_1, float R_10, float R_100)
{
    file << std::to_string(nprobe) << ',' << datasets << ',' << index_type << ',' << processor << ',' << std::to_string(search_latency) << ',' << std::to_string(search_thput) << ',' << std::to_string(R_1) << ',' << std::to_string(R_10) << ',' << std::to_string(R_100) << std::endl;
}