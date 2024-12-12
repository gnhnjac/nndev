#include "activations.h"
#include <math.h>

float sigmoidf(float f) 
{
    return (1 / (1 + exp(-f)));
}

float reluf(float f)
{
    return fmax(0,f);
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

    return (f >= 0) ? 1 : 0;

}