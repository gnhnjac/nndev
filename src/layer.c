#include "layer.h"
#include "activations.h"
#include <stdio.h>

// propagates the previous layer through the current layer, returns the current layer
layer *layer_propagate(layer *cur, const layer *prev)
{

	mat_mul(cur->nodes, cur->weights, prev->nodes);
	mat_dadd(cur->nodes,cur->biases);

	if (cur->activation == softmaxf)
		softmaxf(cur->nodes);
	else
		mat_apply_func(cur->nodes, cur->activation);

	return cur;

}

// backward propagates the error and adjusts the weights and biases accordingly.
// returns the prev layer's cost
matrix *layer_backpropagate(layer *cur, const layer *prev, matrix *layer_cost, float lr)
{

	// derivative of activation relative to current layer
	if (cur->activation == softmaxf)
		d_softmax(cur->nodes);
	else
		mat_apply_func(cur->nodes, d_act(cur->activation));

	mat_dhad(layer_cost, cur->nodes);

	matrix *weights_t = mat_mtrans(cur->weights);

	matrix *new_layer_cost = mat_mmul(weights_t, layer_cost);

	// scale the layer cost down by the learning rate
	mat_smul(layer_cost,lr);

	matrix *activations = prev->nodes;

	matrix *activations_t = mat_mtrans(activations);

	matrix *weights_gradient = mat_mmul(layer_cost, activations_t);

	// subtract the gradient from the current layer's weights
	mat_dsub(cur->weights, weights_gradient);

	// subtract the layer cost from the current layer's biases
	mat_dsub(cur->biases,layer_cost);

	mat_free(weights_t);
	mat_free(activations_t);
	mat_free(weights_gradient);

	return new_layer_cost;

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