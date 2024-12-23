#pragma once
#include "matrix.h"

typedef struct _idx_buffer
{

	size_t n_samples;

	matrix **samples;

} idx_buffer, *p_idx_buffer;

idx_buffer *idx_read(char *path);