#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.h"
#include "activations.h"
#include "costs.h"
#include "sample.h"
#include "idx.h"

#define BATCH_SIZE 10

#define TRAINING_ITERS 100000

// returns a float number from -1 to 1
float random_float()
{
    return ((float)(rand()) / (float)(RAND_MAX))*2 - 1;
}

int main()
{

	srand(time(NULL));

	idx_buffer *training_images = idx_read("../MNIST_ORG/train-images.idx3-ubyte");

	idx_buffer *training_labels = idx_read("../MNIST_ORG/train-labels.idx1-ubyte");

	idx_buffer *testing_images = idx_read("../MNIST_ORG/t10k-images.idx3-ubyte");

	idx_buffer *testing_labels = idx_read("../MNIST_ORG/t10k-labels.idx1-ubyte");

	sample *samples[training_images->n_samples];

	sample *testing_samples[testing_images->n_samples];

	for (int i = 0; i < training_images->n_samples; i++)
	{

		float input[28*28] = {0};

		float output[10] = {0};

		for (int j = 0; j < 28*28;j++)
		{

			// normalize values from 0-255 to 0-1
			input[j] = training_images->samples[i]->data[j]/255.0;

		}

		// activate only the corresponding digit
		output[(int)training_labels->samples[i]->data[0]] = 1;

		samples[i] = smpl_create(input,28*28, output, 10);

	}

	for (int i = 0; i < testing_images->n_samples; i++)
	{

		float input[28*28] = {0};

		float output[10] = {0};

		for (int j = 0; j < 28*28;j++)
		{

			// normalize values from 0-255 to 0-1
			input[j] = testing_images->samples[i]->data[j]/255.0;

		}

		// activate only the corresponding digit
		output[(int)testing_labels->samples[i]->data[0]] = 1;

		testing_samples[i] = smpl_create(input,28*28, output, 10);

	}

	size_t layer_arch[] = {28*28,128,10};

	network *net = net_create(1, layer_arch, 
			  sigmoidf, softmaxf, random_float);

	sample *sample_batch[BATCH_SIZE];

	float training_cost_avg = 0;

	for (int i = 0; i < TRAINING_ITERS; i++)
	{

		int choice = rand()%training_images->n_samples;

		// for (int j = 0; j < BATCH_SIZE; j++)
		// {

		// 	int choice = rand()%training_images->n_samples;

		// 	sample_batch[j] = samples[choice];

		// }

		training_cost_avg += net_train_stochastic(net, samples[choice], bce, 0.1);

		// training_cost_avg += net_train_batch(net, sample_batch, BATCH_SIZE, mse, 0.1);

		if (i % (TRAINING_ITERS / 100) == 0)
		{
			printf("training cost %f\n",training_cost_avg/(TRAINING_ITERS/100));
			training_cost_avg = 0;
		}

	}

	printf("training done, testing...");

	int correct = 0;

	for (int i = 0; i < testing_images->n_samples; i++)
	{

		int correct_label = testing_labels->samples[i]->data[0];

		mat_dcopy(net->input.nodes,testing_samples[i]->input);

		matrix *predicted = net_feedforward(net);

		int max_label = 0;
		float max_value = 0;

		for (int j = 0; j < 10; j++)
		{

			if (predicted->data[j] > max_value)
			{
				max_value = predicted->data[j];
				max_label = j;
			}

		}

		if (max_label == correct_label)
			correct++;

	}

	printf("network testing success: %f%%",correct*100.0/(testing_labels->n_samples));

	//net_print(net);



	return 0;

}