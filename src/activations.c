#include "activations.h"
#include <math.h>

float sigmoidf(float f) 
{
    return (1 / (1 + exp(-f)));
}

float reluf(float f)
{
    return (f > 0) ? f : 0;
}

void softmaxf(matrix *vect)
{

    float max = vect->data[0];

    for(int i = 0; i < vect->rows; i++)
    {
        if (vect->data[i] > max)
            max = vect->data[i];
    }

    float sum = 0;

    for (int i = 0; i < vect->rows; i++) {
        vect->data[i] = exp(vect->data[i] - max);
        sum += vect->data[i];
    }

    mat_smul(vect,1.0/sum);

}

// returns a function pointer to the derivative of the activation function
float (*d_act(float (*act)(float)))(float)
{

    if (act == sigmoidf)
        return &d_sigmoidf;
    else if(act == reluf)
        return &d_reluf;
    return 0;

}

float d_sigmoidf(float f)
{
    return f * (1 - f);

}

float d_reluf(float f)
{

    return (f > 0) ? 1 : 0;

}

void d_softmax(matrix *vect)
{

    matrix *vect_ident = mat_identitize(vect);

    matrix *vect_t = mat_mtrans(vect);

    matrix *vect_dot = mat_mmul(vect,vect_t);

    mat_dsub(vect_ident,vect_dot);

    mat_free(vect_t);
    mat_free(vect_dot);

    for(int i = 0; i < vect->rows; i++)
        vect->data[i] = vect_ident->data[i*vect->rows+i];

}