#ifndef __READ_VECTOR_DATASETS__
#define __READ_VECTOR_DATASETS__

#include <stdio.h>
#include <cstdint>
#include <string>

struct datasetInfo
{
    size_t vec_dim;
    size_t vec_num;
    int *vec_data_int = nullptr;
};

float* load_fvecs_data(const char *fname, size_t *d_out, size_t *n_out);

int* load_ivecs_data(const char *fname, size_t *d_out, size_t *n_out);

uint8_t *load_bvecs_data(const char *fname, size_t *d_out, size_t *n_out);

uint8_t *load_bvecs_base_data(const char *fname, size_t *d_out, size_t *n_out);

datasetInfo load_query_vec_data(const std::string &datasetName, const std::string &datasetPath);

#endif