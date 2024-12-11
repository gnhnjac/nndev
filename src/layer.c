#include "layer.h"

// propagates the previous layer through the current layer, returns the current layer
layer *layer_propagate(layer *cur, layer *prev)
{

	mat_mul(cur->nodes, cur->weights, prev->nodes);
	if (cur->activation)
		mat_apply_func(cur->nodes, cur->activation);

	return cur;

}