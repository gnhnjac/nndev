#pragma once

float sigmoidf(float f);
float reluf(float f);
float (*d_act(float (*act)(float)))(float);
float d_sigmoidf(float f);
float d_reluf(float f);