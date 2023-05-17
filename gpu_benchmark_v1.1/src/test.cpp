
// System includes
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include"iostream"

using namespace std;

bool AllClose(float * x, float * y, int size, float eps)
{

	printf("Comparing GPU and Host CPU results...\n\n");

	for (int i = 0; i < size; i++) {
		float err = fabs(x[i] - y[i]);
		if (err > eps) {
			cout << "x[" << i << "]=" << x[i] << ", y[" << i << "]=" << y[i] << std::endl;

			cout << "error" <<endl;

			return false;
		}
	}

	printf("success...\n\n");
	return true;
}