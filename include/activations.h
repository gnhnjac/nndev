#pragma once
#include "matrix.h"

float sigmoidf(float f);
float reluf(float f);
void softmaxf(matrix *vect);
float (*d_act(float (*act)(float)))(float);
float d_sigmoidf(float f);
float d_reluf(float f);
void d_softmax(matrix *vect);