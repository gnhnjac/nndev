#pragma once
#include "matrix.h"

typedef struct _sample
{

	matrix *input;
	matrix *output;

} sample, *p_sample;

#define smpl_valid(net, sample) ((sample).input->cols == 1 && (sample).output->cols == 1\
									&& (sample).input->rows == (net).input.nodes->rows\
									&& (sample).output->rows == (net).output.nodes->rows)

void smpl_init(sample *s, float input[], size_t input_sz, float output[], size_t output_sz);
sample *smpl_create(float input[], size_t input_sz, float output[], size_t output_sz);