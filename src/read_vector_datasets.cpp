#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "read_vector_datasets.h"
#include <cstdint>

float* load_fvecs_data(const char *fname, size_t *d_out, size_t *n_out)
{
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d; *n_out = n;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for(size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* load_ivecs_data(const char *fname, size_t *d_out, size_t *n_out)
{
    return (int*)load_fvecs_data(fname, d_out, n_out);
}

uint8_t *load_bvecs_data(const char *fname, size_t *d_out, size_t *n_out)
{
    FILE *f = fopen(fname, "r");

    if (!f)
    {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % (d + 4) == 0 || !"weird file size");
    size_t n = sz / (d + 4);

    *d_out = d;
    *n_out = n;
    uint8_t *x = new uint8_t[n * (d + 4)];
    size_t nr = fread(x, sizeof(uint8_t), n * (d + 4), f);
    assert(nr == n * (d + 4) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 4), d * sizeof(*x));

    fclose(f);
    return x;
}

uint8_t *load_bvecs_base_data(const char *fname, size_t *d_out, size_t *n_out)
{
    FILE *f = fopen(fname, "r");

    if (!f)
    {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % (d + 4) == 0 || !"weird file size");
    size_t n = sz / (d + 4);

    *d_out = d;
    *n_out = n / 5;
    uint8_t *x = new uint8_t[n / 5 * (d + 4)];
    size_t nr = fread(x, sizeof(uint8_t), n / 5 * (d + 4), f);
    assert(nr == n / 5 * (d + 4) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n / 5; i++)
        memmove(x + i * d, x + 1 + i * (d + 4), d * sizeof(*x));

    fclose(f);
    return x;
}

datasetInfo load_query_vec_data(const std::string &datasetName, const std::string &datasetPath)
{
    datasetInfo info;
    int *vec_data_int = nullptr;
    if (datasetName == "sift200M" || datasetName == "sift500M")
    {
        uint8_t *vec_data_bignn = load_bvecs_data(datasetPath.c_str(), &(info.vec_dim), &(info.vec_num));
        vec_data_int = (int *)malloc(sizeof(int) * info.vec_num * info.vec_dim);
        for (size_t i = 0; i < info.vec_num; i++)
        {
            for (size_t j = 0; j < info.vec_dim; j++)
            {
                vec_data_int[i * info.vec_dim + j] = int(vec_data_bignn[i * info.vec_dim + j]);
            }
        }
        delete[] vec_data_bignn;
        vec_data_bignn = nullptr;
    }
    else
    {
        float *vec_data_float = load_fvecs_data(datasetPath.c_str(), &info.vec_dim, &info.vec_num);
        vec_data_int = (int *)malloc(sizeof(int) * info.vec_num * info.vec_dim);
        for (size_t i = 0; i < info.vec_num; i++)
        {
            for (size_t j = 0; j < info.vec_dim; j++)
            {
                vec_data_int[i * info.vec_dim + j] = int(vec_data_float[i * info.vec_dim + j]);
            }
        }
        delete[] vec_data_float;
        vec_data_float = nullptr;
    }
    info.vec_data_int = vec_data_int;
    return info;
}