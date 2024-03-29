#ifndef __READ_VECTOR_DATASETS__
#define __READ_VECTOR_DATASETS__

#include <stdio.h>

float* load_fvecs_data(const char *fname, size_t *d_out, size_t *n_out);

int* load_ivecs_data(const char *fname, size_t *d_out, size_t *n_out);

#endif