#include <stdlib.h>
#include "sample.h"

void smpl_init(sample *s, float input[], size_t input_sz, float output[], size_t output_sz)
{

	s->input = mat_vector_from_arr(input, input_sz);

	s->output = mat_vector_from_arr(output, output_sz);

}

sample *smpl_create(float input[], size_t input_sz, float output[], size_t output_sz)
{

	sample *s = (sample *)malloc(sizeof(sample));

	smpl_init(s,input,input_sz,output,output_sz);

	return s;

}