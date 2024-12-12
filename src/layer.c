#include "layer.h"
#include <stdio.h>

// propagates the previous layer through the current layer, returns the current layer
layer *layer_propagate(layer *cur, const layer *prev)
{

	mat_mul(cur->nodes, cur->weights, prev->nodes);
	mat_dadd(cur->nodes,cur->biases);

	if (cur->activation)
		mat_apply_func(cur->nodes, cur->activation);

	return cur;

}

void layer_print(const layer *lay)
{

	printf("nodes:\n");
	mat_print(lay->nodes);

	printf("biases:\n");
	mat_print(lay->biases);

	printf("weights:\n");
	mat_print(lay->weights);

}