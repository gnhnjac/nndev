#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
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

	matrix *layer_cost = mat_copy(cost_derivative);

	matrix *new_layer_cost = layer_backpropagate(&net->output,
		(net->hidden_count > 0) ? &net->hidden[net->hidden_count-1] : &net->input,
		layer_cost,
		lr);

	mat_free(layer_cost);

	layer_cost = new_layer_cost;

	for(int i = net->hidden_count - 1; i >= 0; i--)
	{

		new_layer_cost = layer_backpropagate(&net->hidden[i],
			(i > 0) ? &net->hidden[i-1] : &net->input,
			layer_cost,
			lr);

		mat_free(layer_cost);

		layer_cost = new_layer_cost;

	}

	mat_free(layer_cost);

}

// trains the network on a single sample
// returns the training cost of the sample
float net_train_stochastic(network *net, const sample *smpl,
	float (*cost)(const matrix *, const matrix *, int), float lr)
{

	matrix *cost_derivative = mat_create(net->output.nodes->rows, 1);

	assert(smpl_valid(*net,*smpl));

	mat_dcopy(net->input.nodes,smpl->input);

	matrix *prediction = net_feedforward(net);

	float training_cost = cost(smpl->output,prediction,1);

	d_cost(cost)(cost_derivative,smpl->output,prediction,1);

	mat_free(prediction);

	net_backpropagate(net, cost_derivative, lr);

	mat_free(cost_derivative);

	return training_cost;

}

// trains the network given a set of samples
// returns the training cost of the network
float net_train_batch(network *net, const sample *samples[], size_t sample_sz, 
	float (*cost)(const matrix *, const matrix *, int), float lr)
{

	float training_cost = 0;

	matrix *average_cost_derivative = mat_create(net->output.nodes->rows, 1);

	matrix *sample_cost_derivative = mat_copy(average_cost_derivative);

	for (int i = 0; i < sample_sz; i++)
	{

		assert(smpl_valid(*net,*samples[i]));

		mat_dcopy(net->input.nodes,samples[i]->input);

		matrix *prediction = net_feedforward(net);

		training_cost += cost(samples[i]->output,prediction,sample_sz);

		d_cost(cost)(sample_cost_derivative,samples[i]->output,prediction,sample_sz);

		mat_dadd(average_cost_derivative,sample_cost_derivative);

		mat_free(prediction);

	}

	net_backpropagate(net, average_cost_derivative, lr);

	mat_free(average_cost_derivative);
	mat_free(sample_cost_derivative);

	return training_cost;

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