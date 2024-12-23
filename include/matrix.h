#pragma once
#include <stddef.h>

typedef struct _matrix
{

	size_t rows;
	size_t cols;
	float *data;

} matrix, *p_matrix;

// returns whether the 2 matrices have equal dimensions.
#define mat_deq(m1, m2) ((m1).rows == (m2).rows && (m1).cols == (m2).cols)

// returns whether the 3 matrices have equal dimensions.
#define mat_deq3(m1, m2, m3) ((m1).rows == (m2).rows && (m1).cols == (m2).cols && (m1).rows == (m3).rows && (m1).cols == (m3).cols)

// returns whether the 2 matrices are valid for matrix multiplication
#define mat_meq(m1,m2) ((m1).cols == (m2).rows)

// returns whether the 3 matrices are valid for matrix multiplication
#define mat_meq3(dst,m1,m2) ((dst).rows == (m1).rows && (dst).cols == (m2).cols && (m1).cols == (m2).rows)

// returns whether the 2 matrices are valid for transposition
#define mat_teq(m1,m2) ((m1).rows == (m2).cols && (m1).cols == (m2).rows)

// returns the 1 dimensional index for the specified row and column in the array
#define mat_at(mat,row,col) (row * (mat).cols + col)

void mat_init(matrix *mat, size_t rows, size_t cols);
matrix *mat_vector_from_arr(float arr[], size_t sz);
matrix *mat_create(size_t rows, size_t cols);
matrix *mat_copy(const matrix *mat);
void mat_dcopy(matrix *dst, const matrix *src);
void mat_load(matrix *mat, float data[]);
matrix *mat_ident(size_t dim, float val);
matrix *mat_identitize(matrix *vect);
void mat_add(matrix *dst, const matrix *mat1, const matrix *mat2);
void mat_sub(matrix *dst, const matrix *mat1, const matrix *mat2);
void mat_dadd(matrix *mat1, const matrix *mat2);
void mat_dsub(matrix *mat1, const matrix *mat2);
void mat_smul(matrix *mat, float scalar);
void mat_trans(matrix *dst, const matrix *mat);
matrix *mat_mtrans(const matrix *mat);
void mat_mul(matrix *dst, const matrix *mat1, const matrix *mat2);
matrix *mat_mmul(const matrix *mat1, const matrix *mat2);
void mat_had(matrix *dst, const matrix *mat1, const matrix *mat2);
matrix *mat_mhad(const matrix *mat1, const matrix *mat2);
void mat_dhad(matrix *mat1, const matrix *mat2);
void mat_set(matrix *mat, float val);
void mat_set_func(matrix *mat, float (*func)());
void mat_apply_func(matrix *mat, float (*func)(float));
void mat_print(const matrix *mat);
void mat_free(matrix *mat);