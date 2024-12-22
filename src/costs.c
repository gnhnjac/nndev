#include "costs.h"

// returns the squared mean error of the predicted vector relative to the expected vector
// (mse) = 1/n * sum of ((expected - predicted)^2)
float mse(const matrix *expected, const matrix *predicted, int samples)
{

	matrix *error = mat_create(expected->rows,1);

	mat_sub(error,expected,predicted);

	matrix *error_transposed = mat_mtrans(error);

	matrix *mse_value_mat = mat_mmul(error_transposed,error);

	float mse_value = mse_value_mat->data[0] / samples;

	mat_free(error);
	mat_free(error_transposed);
	mat_free(mse_value_mat);

	return mse_value;

}

// returns a function pointer to the derivative of the cost function
void (*d_cost(float (*cost)(const matrix *, const matrix *, int)))(matrix *, const matrix *, const matrix *, int)
{

    if (cost == mse)
        return &d_mse;

    return 0;

}

// returns the derivative of the mean squared error
// (mse)` = -2/n * (expected - predicted))
void d_mse(matrix *dst, const matrix *expected, const matrix *predicted, int samples)
{

	//matrix *error = mat_create(expected->rows,1);

	// mat_sub(dst,expected,predicted);

	mat_sub(dst,predicted,expected);

	mat_smul(dst,2/samples);

	//return error;

}