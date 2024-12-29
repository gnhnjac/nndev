#include <stdlib.h>
#include "sample.h"

// initializes a sample given 2 arrays
void smpl_init(sample *s, float input[], size_t input_sz, float output[], size_t output_sz)
{

	s->input = mat_vector_from_arr(input, input_sz);

	s->output = mat_vector_from_arr(output, output_sz);

}

// creates a sample given 2 arrays
sample *smpl_create(float input[], size_t input_sz, float output[], size_t output_sz)
{

	sample *s = (sample *)malloc(sizeof(sample));

	smpl_init(s,input,input_sz,output,output_sz);

	return s;

}

// creates a sample given 2 matrices
sample *smpl_create_mat(matrix *input, matrix *output)
{

	sample *s = (sample *)malloc(sizeof(sample));

	s->input = mat_copy(input);

	s->output = mat_copy(output);

	return s;

}

// frees the sample's internal structure and the sample itself
void smpl_free(sample *smpl)
{

	if (!smpl)
		return;

	mat_free(smpl->input);
	mat_free(smpl->output);

	free(smpl);

}