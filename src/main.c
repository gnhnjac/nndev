#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.h"
#include "activations.h"
#include "costs.h"

// returns a float number from -1 to 1
float random_float()
{
    return ((float)(rand()) / (float)(RAND_MAX))*2 - 1;
}

int main()
{

	srand(time(NULL)); 

	size_t layer_arch[] = {2,2,1};

	network *net = net_create(1, layer_arch, 
			  sigmoidf, sigmoidf, random_float);

	matrix *expected = mat_create(1,1);

	matrix *cost_derivative = mat_create(1,1);

	float cost = 0;

	for (int i = 0; i < 10000; i++)
	{

		mat_set(cost_derivative,0);

		int choice = rand()%4;

		if (choice == 0)
		{

			net->input.nodes->data[0] = 0;
			net->input.nodes->data[1] = 1;
			mat_set(expected,1);

		} else if (choice == 1)
		{

			net->input.nodes->data[1] = 0;
			net->input.nodes->data[0] = 1;
			mat_set(expected,1);

		} else if (choice == 2)
		{

			net->input.nodes->data[0] = 1;
			net->input.nodes->data[1] = 1;
			mat_set(expected,0);

		} else if (choice == 3)
		{

			net->input.nodes->data[0] = 0;
			net->input.nodes->data[1] = 0;
			mat_set(expected,0);

		}

		matrix *predicted = net_feedforward(net);

		cost += mse(expected,predicted,10000);

		d_mse(cost_derivative,expected,predicted,1);

		net_backpropagate(net, cost_derivative, 1);

		mat_free(predicted);

	}

	printf("cost: %f\n",cost);


	net->input.nodes->data[0] = 1;
	net->input.nodes->data[1] = 1;

	mat_print(net_feedforward(net));

	net->input.nodes->data[0] = 0;
	net->input.nodes->data[1] = 0;

	mat_print(net_feedforward(net));

	net->input.nodes->data[0] = 0;
	net->input.nodes->data[1] = 1;

	mat_print(net_feedforward(net));

	net->input.nodes->data[0] = 1;
	net->input.nodes->data[1] = 0;

	mat_print(net_feedforward(net));

	net->input.nodes->data[0] = 0.5;
	net->input.nodes->data[1] = 0.5;

	mat_print(net_feedforward(net));

	//net_print(net);



	return 0;

}