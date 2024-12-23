#pragma once
#include "matrix.h"

float mse(const matrix *expected, const matrix *predicted);
float bce(const matrix *expected, const matrix *predicted);
void (*d_cost(float (*cost)(const matrix *, const matrix *)))(matrix *, const matrix *, const matrix *);
void d_mse(matrix *dst, const matrix *expected, const matrix *predicted);
void d_bce(matrix *dst, const matrix *expected, const matrix *predicted);