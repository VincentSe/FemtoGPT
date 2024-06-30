#include <stdexcept>
#include <cmath>
#include <iostream>

#include "ParamFunc.h"

void squareGrad(const double* x, /*out*/double* g)
{
	// Gradient of x[0]^2 + x[1]^2
	g[0] = 2.0 * x[0];
	g[1] = 2.0 * x[1];
}

// Just to test Adam, there is a closed formula for affine regression.
// Gradient of mean square error.
void AffineRegressionGrad(const double* x, /*out*/double* g)
{
	double a = x[0];
	double b = x[1];
	std::vector<double> abs{ 0.0, 1.0, 2.0, 3.0, 4.0 };
	std::vector<double> ord{ 7.3256989, 9.8042683, 12.2828377, 14.7614071, 17.2399765 };
	// true a and b are 2.4785694 and 7.3256989.
	g[0] = 0.0;
	for (unsigned int i = 0; i < 5; i++)
		g[0] += (a * abs[i] + b - ord[i]) * abs[i];
	g[0] *= (2.0 / abs.size());

	g[1] = 0.0;
	for (unsigned int i = 0; i < 5; i++)
		g[1] += a * abs[i] - ord[i];
	g[1] *= (2.0 / abs.size());
	g[1] += 2.0 * b;
}

pf::AdamOptimizer::AdamOptimizer(unsigned int inputDim,
	double stepSize,
	std::function<double& (unsigned int)> GetParam,
	std::function<double* ()> grad)
: fInputDim(inputDim),
fStepSize(stepSize),
fGrad(grad),
fGetParam(GetParam),
mean(inputDim, 0.0),
var(inputDim, 0.0)
{
}

void pf::AdamOptimizer::Step()
{
	double* g = fGrad();
	for (unsigned int i = 0; i < fInputDim; i++)
	{
		mean[i] = beta1 * mean[i] + (1 - beta1) * g[i];
		var[i] = beta2 * var[i] + (1 - beta2) * g[i] * g[i];
		fGetParam(i) -= fStepSize * mean[i] / ((1 - beta1t) * (std::sqrt(var[i] / (1-beta2t)) + epsilon));
	}
	beta1t *= beta1;
	beta2t *= beta2;
}

void TestAdam()
{
	std::vector<double> x(2, 0.0);
	std::vector<double> g(2, 0.0);
	pf::AdamOptimizer optim(2,
		0.05,
		[&x](unsigned int i) -> double& { return x[i]; },
		[&x, &g]() {AffineRegressionGrad(x.data(), g.data()); return g.data(); });
	for (unsigned int i = 0; i < 100; i++)
	{
		optim.Step();
		std::cout << "a: " << x[0] << " b: " << x[1] << std::endl;
	}
}
