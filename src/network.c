#include <stdlib.h>
#include "network.h"


// initializes a network with the specified layers, activation and initialization function.
void net_init(network *net, size_t hidden_count, size_t *layer_sizes, 
			  float (*hid_act)(float), float (*out_act)(float), float (*init_func)())
{

	net->input = (layer){

		// vector of layer nodes
		.nodes = mat_create(layer_sizes[0],1),

		// matrix of weights connecting previous layer to current layer
		// for this layer it's 0 since it's the first layer
		.weights = 0,

		// layer activation function, 0 since it's input
		.activation = 0

	};

	net->hidden_count = hidden_count;
	net->hidden = (layer *)calloc(hidden_count, sizeof(layer));

	// initialize the hidden layers
	for (int i = 0; i < hidden_count; i++)
	{

		net->hidden[i] = (layer){

			// vector of layer nodes
			.nodes = mat_create(layer_sizes[i + 1],1),

			// matrix of weights connecting previous layer to current layer
			// needs to have dims of (this layer nodes) x (prev layer nodes)
			.weights = mat_create(layer_sizes[i + 1],layer_sizes[i]),

			// layer activation function
			.activation = hid_act

		};

		// initialize weights according to init function
		mat_set_func(net->hidden[i].weights, init_func);

	}

	net->output = (layer){

		// vector of layer nodes
		.nodes = mat_create(layer_sizes[hidden_count+1],1),

		// matrix of weights connecting previous layer to current layer
		// needs to have dims of (this layer nodes) x (prev layer nodes)
		.weights = mat_create(layer_sizes[hidden_count+1],layer_sizes[hidden_count]),

		// layer activation function
		.activation = out_act

	};

	mat_set_func(net->output.weights, init_func);

}

// creates a network with the specified layers, activation and initialization function.
network *net_create(size_t hidden_count, size_t *layer_sizes, 
			  float (*hid_act)(float), float (*out_act)(float), float (*init_func)())
{

	network *net = (network *)malloc(sizeof(network));

	net_init(net,hidden_count,layer_sizes,hid_act,out_act,init_func);

	return net;

}

// forward propagates the network and returns the resulting output vector
matrix *net_feedforward(network *net)
{

	layer *src = &net->input;

	// propagate the layers forward by feeding the prev layer through the weights and applying the activation
	for (int i = 0; i < net->hidden_count; i++)
		src = layer_propagate(&net->hidden[i], src);

	// propagate through to the output layer
	return layer_propagate(&net->output, src)->nodes;

}