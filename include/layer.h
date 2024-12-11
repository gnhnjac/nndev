#pragma once

#include "matrix.h"

typedef struct _layer
{

	// vector of layer nodes
	matrix *nodes;

	// matrix of weights connecting previous layer to current layer
	matrix *weights;

	// layer activation function
	float (*activation)(float);

} layer, *p_layer;

layer *layer_propagate(layer *cur, layer *prev);