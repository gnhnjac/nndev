#include <math.h>
#include "costs.h"

// returns the squared mean error of the predicted vector relative to the expected vector
// (mse) = 1/n * sum of ((expected - predicted)^2)
float mse(const matrix *expected, const matrix *predicted)
{

	matrix *error = mat_create(expected->rows,1);

	mat_sub(error,expected,predicted);

	matrix *error_transposed = mat_mtrans(error);

	matrix *mse_value_mat = mat_mmul(error_transposed,error);

	float mse_value = mse_value_mat->data[0] / error->rows;

	mat_free(error);
	mat_free(error_transposed);
	mat_free(mse_value_mat);

	return mse_value;

}

float bce(const matrix *expected, const matrix *predicted)
{

	//  mean(-y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred))

	float sum = 0;

	for(int i = 0; i < expected->rows; i++)
	{

		float y_true = expected->data[i];
		float y_pred = predicted->data[i];

		if (y_pred < 0.0001)
			y_pred = 0.0001;
		else if (y_pred > (1 - 0.0001))
			y_pred = 1 - 0.0001;

		sum += -y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred);

	}

	sum /= expected->rows;

	return sum;

}

// returns a function pointer to the derivative of the cost function
void (*d_cost(float (*cost)(const matrix *, const matrix *)))(matrix *, const matrix *, const matrix *)
{

    if (cost == mse)
        return &d_mse;
    else if(cost == bce)
    	return &d_bce;

    return 0;

}

// returns the derivative of the mean squared error
// (mse)` = -2/n * (expected - predicted))
void d_mse(matrix *dst, const matrix *expected, const matrix *predicted)
{

	mat_sub(dst,predicted,expected);

	mat_smul(dst,2.0/predicted->rows);

}

void d_bce(matrix *dst, const matrix *expected, const matrix *predicted)
{
	
	// ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / len(y_true)

	for (int i = 0; i < expected->rows; i++)
	{

		float y_true = expected->data[i];
		float y_pred = predicted->data[i];

		if (y_pred < 0.0001)
			y_pred = 0.0001;
		else if (y_pred > (1 - 0.0001))
			y_pred = 1 - 0.0001;

		dst->data[i] = ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / expected->rows;

	}

}