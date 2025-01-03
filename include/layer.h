#pragma once

#include "matrix.h"

typedef struct _layer
{

	// vector of layer nodes
	matrix *nodes;

	// matrix of weights connecting previous layer to current layer
	matrix *weights;

	// matrix of biases for the current layer
	matrix *biases;

	// layer activation function
	float (*activation)(float);

} layer, *p_layer;

layer *layer_propagate(layer *cur, const layer *prev);
matrix *layer_backpropagate(layer *cur, const layer *prev, matrix *layer_cost, float lr);
void layer_internal_free(layer *lay);
void layer_print(const layer *lay);