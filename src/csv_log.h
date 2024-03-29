#ifndef __CSV_LOG__
#define __CSV_LOG__

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

class CSV_Vector_Search_Perf_logger
{
    public:
        CSV_Vector_Search_Perf_logger(const char* logger_file);
        ~CSV_Vector_Search_Perf_logger();
    
    public:
        void write_to_csv(int nprobe, const char* datasets, char* index_type, char* processor, double search_latency, double search_thput, float R_1, float R_10, float R_100);

    private:
        std::ofstream file;
};

#endif