#pragma once
#include "matrix.h"

float mse(const matrix *expected, const matrix *predicted, int samples);
void (*d_cost(float (*cost)(const matrix *, const matrix *, int)))(matrix *, const matrix *, const matrix *, int);
void d_mse(matrix *dst, const matrix *expected, const matrix *predicted, int samples);