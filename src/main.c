#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.h"
#include "activations.h"
#include "costs.h"
#include "sample.h"

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

	float azero[] = {0};
	float aone[] = {1};
	float azerozero[] = {0, 0};
	float aoneone[] = {1, 1};
	float azeroone[] = {0, 1};
	float aonezero[] = {1, 0};

	sample *zero = smpl_create(azerozero,2,azero,1);
	sample *one = smpl_create(aoneone,2,azero,1);
	sample *zeroone = smpl_create(azeroone,2,aone,1);
	sample *onezero = smpl_create(aonezero,2,aone,1);

	sample *samples[4] = {zero,one,zeroone,onezero};

	for (int i = 0; i < 10000; i++)
	{

		int choice = rand()%4;

		printf("training cost: %f\n", net_train_stochastic(net, samples[choice], mse, 1));

	}

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