#pragma once

#include"include/matrix.hpp"

using namespace vbfod_gpu;

Matrix<float> *temp_local;
Matrix<float> *temp;

bool AllClose(float * x, float * y, int size, float eps);