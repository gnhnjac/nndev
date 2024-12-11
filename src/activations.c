#include "activations.h"
#include <math.h>

float sigmoidf(float f) 
{
    return (1 / (1 + pow(EULER_NUMBER_F, -f)));
}