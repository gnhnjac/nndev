#include <stdlib.h>
#include <stdio.h>
#include "network.h"
#include "costs.h"
#include "activations.h"


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

		// matrix of biases for the current layer, 0 since it's the input layer
		.biases = 0,

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

			// matrix of biases for the current layer
			.biases = mat_create(layer_sizes[i + 1],1),

			// layer activation function
			.activation = hid_act

		};

		// initialize weights according to init function
		mat_set_func(net->hidden[i].weights, init_func);

		// initialize biases according to init function
		mat_set_func(net->hidden[i].biases, init_func);

	}

	net->output = (layer){

		// vector of layer nodes
		.nodes = mat_create(layer_sizes[hidden_count+1],1),

		// matrix of weights connecting previous layer to current layer
		// needs to have dims of (this layer nodes) x (prev layer nodes)
		.weights = mat_create(layer_sizes[hidden_count+1],layer_sizes[hidden_count]),

		// matrix of biases for the current layer
		.biases = mat_create(layer_sizes[hidden_count + 1],1),

		// layer activation function
		.activation = out_act

	};

	mat_set_func(net->output.weights, init_func);

	mat_set_func(net->output.biases, init_func);

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
	return mat_copy(layer_propagate(&net->output, src)->nodes);

}

// backward propagates the error and adjusts the weights and biases accordingly.
void net_backpropagate(network *net, const matrix *cost_derivative , float lr)
{

	// derivative of activation relative to output
	mat_apply_func(net->output.nodes, d_act(net->output.activation));

	matrix *layer_cost = mat_dhad(net->output.nodes, cost_derivative);

	matrix *scaled_cost = mat_copy(layer_cost);

	// scale the layer cost down by the learning rate
	mat_smul(scaled_cost,lr);

	matrix *activations = (net->hidden_count > 0) ? net->hidden[net->hidden_count-1].nodes : net->input.nodes;

	matrix *activations_t = mat_dtrans(activations);

	matrix *weights_t = mat_dtrans(net->output.weights);

	matrix *weights_gradient = mat_dmul(scaled_cost, activations_t);

	// subtract the gradient from the current layer's weights
	mat_dsub(net->output.weights, weights_gradient);

	// subtract the layer cost from the current layer's biases
	mat_dsub(net->output.biases,scaled_cost);

	mat_free(scaled_cost);
	//mat_free(cost_derivative);
	mat_free(activations_t);
	mat_free(weights_gradient);

	for(int i = net->hidden_count - 1; i >= 0; i--)
	{

		mat_apply_func(net->hidden[i].nodes,d_act(net->hidden[i].activation));

		activations = (i > 0) ? net->hidden[i-1].nodes : net->input.nodes;
		activations_t = mat_dtrans(activations);

		matrix *new_layer_cost_rhs = mat_dmul(weights_t, layer_cost);
		matrix *new_layer_cost = mat_dhad(net->hidden[i].nodes, new_layer_cost_rhs);

		scaled_cost = mat_copy(new_layer_cost);

		// // scale the layer cost down by the learning rate
		mat_smul(scaled_cost,lr);

		mat_free(weights_t);
		mat_free(new_layer_cost_rhs);

		weights_t = mat_dtrans(net->hidden[i].weights);

		weights_gradient = mat_dmul(scaled_cost, activations_t);

		mat_dsub(net->hidden[i].weights, weights_gradient);

		mat_dsub(net->hidden[i].biases, scaled_cost);

		mat_free(scaled_cost);
		mat_free(layer_cost);
		mat_free(weights_gradient);
		mat_free(activations_t);

		layer_cost = new_layer_cost;

	}

	mat_free(weights_t);
	mat_free(layer_cost);

}

void net_print(const network *net)
{

	printf("input layer:\n");
	layer_print(&net->input);

	printf("hidden layers:\n");
	for (int i = 0; i < net->hidden_count; i++)
	{

		printf("hidden layer %d:\n",i+1);
		layer_print(&net->hidden[i]);

	}

	printf("output layer:\n");
	layer_print(&net->output);

}