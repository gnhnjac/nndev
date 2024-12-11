#include "costs.h"

// returns the squared mean error of the predicted vector relative to the expected vector
float mse(matrix *expected, matrix *predicted)
{

	mat_dsub(expected,predicted);

	matrix *expected_transposed = mat_dtrans(expected);

	matrix *mse_value_mat = mat_dmul(expected_transposed,expected);

	float mse_value = mse_value_mat->data[0] / expected->rows;

	mat_free(expected_transposed);
	mat_free(mse_value_mat);

	return mse_value;

}