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

	size_t layer_arch[] = {2,2,2};

	network *net = net_create(1, layer_arch, 
			  sigmoidf, sigmoidf, random_float);

	matrix *ones = mat_create(2,1);
	mat_set(ones,1);

	mat_print(net_feedforward(net));

	printf("mean squared error: %f",mse(ones,net_feedforward(net)));

	return 0;

}