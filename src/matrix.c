#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stddef.h>
#include "matrix.h"

// initializes a matrix with the requested dimensions
void mat_init(matrix *mat, size_t rows, size_t cols)
{

	mat->rows = rows;
	mat->cols = cols;
	mat->data = (float *)calloc(rows * cols,sizeof(float));

}

matrix *mat_vector_from_arr(float arr[], size_t sz)
{

	matrix *vector = mat_create(sz,1);

	for (int i = 0; i < sz; i++)
		vector->data[i] = arr[i];

	return vector;

}

// creates and initializes a matrix with the requested dimensions
matrix *mat_create(size_t rows, size_t cols)
{

	matrix *mat = (matrix *)malloc(sizeof(matrix));

	mat_init(mat,rows,cols);

	return mat;

}

// returns a copy of a matrix
matrix *mat_copy(const matrix *mat)
{

	matrix *cpy = mat_create(mat->rows,mat->cols);

	for (int i = 0; i < mat->rows*mat->cols; i++)
		cpy->data[i] = mat->data[i];

	return cpy;

}

// copies src to dst
void mat_dcopy(matrix *dst, const matrix *src)
{

	assert(mat_deq(*src,*dst));

	for (int i = 0; i < src->rows*src->cols; i++)
		dst->data[i] = src->data[i];
	
}

void mat_load(matrix *mat, float data[])
{

	for (int i = 0; i < mat->rows*mat->cols; i++)
		mat->data[i] = data[i];

}

// returns an identity matrix of dimension: dim x dim and value: val
matrix *mat_ident(size_t dim, float val)
{

	matrix *ident = mat_create(dim,dim);

	for (int i = 0; i < dim; i++)
		ident->data[i*dim + i] = val;

	return ident;

}

// returns an identitized version of a vector
matrix *mat_identitize(matrix *vect)
{

	assert(vect->cols == 1); // make sure it's a vector

	matrix *ident = mat_create(vect->rows,vect->rows);

	for(int i = 0; i < vect->rows; i++)
		ident->data[i*vect->rows + i] = vect->data[i];

	return ident;

}

// adds two matrices together, all dimensions have to be equal
void mat_add(matrix *dst, const matrix *mat1, const matrix *mat2)
{

	assert(mat_deq3(*dst,*mat1,*mat2));

	for (int i = 0; i < mat1->rows*mat1->cols; i++)
		dst->data[i] = mat1->data[i] + mat2->data[i];
}

// subtracts two matrices, all dimensions have to be equal
void mat_sub(matrix *dst, const matrix *mat1, const matrix *mat2)
{

	assert(mat_deq3(*dst,*mat1,*mat2));

	for (int i = 0; i < mat1->rows*mat1->cols; i++)
		dst->data[i] = mat1->data[i] - mat2->data[i];

}

// directly adds elements of 2nd matrix to 1st matrix
void mat_dadd(matrix *mat1, const matrix *mat2)
{
	mat_add(mat1,mat1,mat2);
}

// directly subtracts elements of 2nd matrix from 1st matrix
void mat_dsub(matrix *mat1, const matrix *mat2)
{
	mat_sub(mat1,mat1,mat2);
}

// multiplies a matrix by a scalar amount
void mat_smul(matrix *mat, float scalar)
{

	for (int i = 0; i < mat->rows*mat->cols; i++)
		mat->data[i] *= scalar;

}

// transforms a matrix into the destination
void mat_trans(matrix *dst, const matrix *mat)
{

	assert(mat_teq(*dst,*mat));

	size_t idx = 0;

	for (int i = 0; i < mat->rows; i++)
	{

		for (int j = 0; j < mat->cols; j++)
		{

			dst->data[mat_at(*dst,j,i)] = mat->data[idx++];

		}

	}

}

// transpose a matrix and return the transposed malloced matrix
matrix *mat_mtrans(const matrix *mat)
{

	matrix *dst = mat_create(mat->cols,mat->rows);

	mat_trans(dst,mat);

	return dst;

}

// multiplies two matrices together, 1st's cols have to equal 2nd's rows
// the destination matrix's dimensions have to be {1st rows, 2nd cols}
void mat_mul(matrix *dst, const matrix *mat1, const matrix *mat2)
{

	assert(mat_meq3(*dst, *mat1, *mat2));

	size_t idx = 0;

	for (int i = 0; i < mat1->rows; i++)
	{
		for (int j = 0; j < mat2->cols; j++)
		{

			size_t mat1_idx = i * mat1->cols;
			size_t mat2_idx = j;

			float sum = 0;

			for (int n = 0; n < mat1->cols; n++)
			{
				sum += mat1->data[mat1_idx]*mat2->data[mat2_idx];

				mat1_idx++;
				mat2_idx += mat2->cols;

			}

			dst->data[idx++] = sum;

		}
	}

}

// multiplies 2 matrices together and returns the resulting malloced matrix
matrix *mat_mmul(const matrix *mat1, const matrix *mat2)
{

	assert(mat_meq(*mat1, *mat2));

	matrix *mat = mat_create(mat1->rows, mat2->cols);

	mat_mul(mat,mat1,mat2);

	return mat;

}

// multiplies 2 matrices elementwise and puts it in the 3rd matrix
void mat_had(matrix *dst, const matrix *mat1, const matrix *mat2)
{

	assert(mat_deq(*mat1,*mat2));

	for (int i = 0; i < mat1->rows*mat1->cols; i++)
	{

		dst->data[i] = mat1->data[i] * mat2->data[i];

	}

}

// multiplies 2 matrices together
// returns the resulting hadamard product as an allocated matrix
matrix *mat_mhad(const matrix *mat1, const matrix *mat2)
{

	matrix *mat = mat_create(mat1->rows,mat2->cols);

	mat_had(mat,mat1,mat2);

	return mat;

}

// directly multiplies 2 matrices together hadamardwise and puts the result in the 1st
void mat_dhad(matrix *mat1, const matrix *mat2)
{
	mat_had(mat1,mat1,mat2);
}

// sets a matrix's elements to the specified value
void mat_set(matrix *mat, float val)
{

	for (int i = 0; i < mat->rows*mat->cols; i++)
		mat->data[i] = val;

}

// sets a matrix's elements according to the given function
void mat_set_func(matrix *mat, float (*func)())
{

	for (int i = 0; i < mat->rows*mat->cols; i++)
		mat->data[i] = func();

}

// sets a matrix's elements according to the given function which accepts each element
void mat_apply_func(matrix *mat, float (*func)(float))
{

	if (!func)
		return;

	for (int i = 0; i < mat->rows*mat->cols; i++)
		mat->data[i] = func(mat->data[i]);

}

// prints a matrix
void mat_print(const matrix *mat)
{

	if (!mat)
		return;

	printf("%ldx%ld:\n",mat->rows,mat->cols);

	size_t idx = 0;

	for (int i = 0; i < mat->rows; i++)
	{

		for (int j = 0; j < mat->cols; j++)
		{

			printf("%f ",mat->data[idx++]);

		}

		putchar('\n');

	}

}

// free a matrix that was allocated dynamically including the matrix itself
void mat_free(matrix *mat)
{

	if (!mat)
		return;

	free(mat->data);
	free(mat);

}