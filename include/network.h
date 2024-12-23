#pragma once

#include <stddef.h>
#include "layer.h"
#include "sample.h"

typedef struct _network
{

	layer input;

	size_t hidden_count;
	layer *hidden;

	layer output;

} network, *p_network;

void net_init(network *net, size_t hidden_count, size_t *layer_sizes, 
			  float (*hid_act)(float), float (*out_act)(float), float (*init_func)());

network *net_create(size_t hidden_count, size_t *layer_sizes, 
			  float (*hid_act)(float), float (*out_act)(float), float (*init_func)());

matrix *net_feedforward(network *net);

void net_sub_grad_avg(network *net, size_t samples);

void net_calc_grads(network *net, const matrix *cost_derivative, float lr);

void net_backpropagate(network *net, const matrix *cost_derivative, float lr);

float net_train_stochastic(network *net, const sample *smpl,
	float (*cost)(const matrix *, const matrix *), float lr);

float net_train_batch(network *net, const sample *samples[], size_t sample_sz, 
	float (*cost)(const matrix *, const matrix *), float lr);


void net_print(const network *net);