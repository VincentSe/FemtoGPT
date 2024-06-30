#include <numeric>
#include <cmath>
#include <chrono>
#include <iostream>
#include <algorithm>

#include "ParamFunc.h"

using namespace pf;

class ScopedClock
{
public:
    ScopedClock()
        : fBegin(std::chrono::steady_clock::now()),
        fName("Unnamed clock")
    {}
    ScopedClock(const std::string& name)
        : fBegin(std::chrono::steady_clock::now()),
        fName(name)
    {
    }
    ~ScopedClock()
    {
        auto end = std::chrono::steady_clock::now();
        std::cout << fName << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(end - fBegin) << std::endl;
    }
    std::chrono::steady_clock::time_point fBegin;
    std::string fName;
};

void pf::AddScaledVector(const double* b,
    const double* e,
    double scale,
    /*out*/double* outVect)
{
    while (b != e)
    {
        *outVect += scale * *b;
        b++;
        outVect++;
    }
}

double pf::MaxNorm(const double* b, const double* e)
{
    double norm = 0.0;
    for (const double* d = b; d < e; d++)
        if (norm < std::abs(*d))
            norm = std::abs(*d);
    return norm;
}

unsigned int ParamFunc::GetInputDim() const
{
    return fInputDim;
}

unsigned int ParamFunc::GetOutputDim() const
{
    return fOutputDim;
}

unsigned int ParamFunc::GetParameterCount() const
{
    return fParameterCount;
}

double ParamFunc::GetParameter(unsigned int index) const
{
    return const_cast<ParamFunc*>(this)->GetParameter(index);
}

std::vector<std::pair<unsigned int, double>> pf::SortParameters(const ParamFunc& f)
{
    std::vector<std::pair<unsigned int, double>> params;
    params.reserve(f.GetParameterCount());
    for (unsigned int i = 0; i < f.GetParameterCount(); i++)
        params.emplace_back(i, f.GetParameter(i));
    std::sort(params.begin(), params.end(), [](const auto& a, const auto& b)
        { return std::abs(a.second) < std::abs(b.second); });
    return params;
}

// Slow and approximative implementation by finite differences,
// to test exact and faster overloads.
void pf::DifferentialFinite(const ParamFunc& f, const double* x, double bump,
    /*out*/double* d)
{
    ScopedClock clk("Finite difference gradient");
    std::vector<double> y(f.GetOutputDim(), 0.0);
    f.Apply(x, /*out*/y.data());
    // With respect to x
    std::vector<double> z(f.GetOutputDim(), 0.0);
    const unsigned int gradSize = f.GetInputDim() + f.GetParameterCount();
    std::fill(d, d + gradSize, 0.0);
    for (unsigned int col = 0; col < f.GetInputDim(); col++)
    {
        const double xSave = x[col];
        const_cast<double&>(x[col]) += bump; // const_cast restored 2 lines below
        f.Apply(x, /*out*/z.data());
        const_cast<double&>(x[col]) = xSave;
        for (unsigned int row = 0; row < f.GetOutputDim(); row++)
            d[row * gradSize + col] += (z[row] - y[row]) / bump;
    }
    // With respect to parameters
    for (unsigned int col = 0; col < f.GetParameterCount(); col++)
    {
        const double pSave = f.GetParameter(col);
        const_cast<ParamFunc&>(f).GetParameter(col) = pSave + bump; // const_cast restored 2 lines below
        f.Apply(x, /*out*/z.data());
        const_cast<ParamFunc&>(f).GetParameter(col) = pSave;
        for (unsigned int row = 0; row < f.GetOutputDim(); row++)
            d[row * gradSize + f.GetInputDim() + col] += (z[row] - y[row]) / bump;
    }
}

bool pf::TestDifferential(const ParamFunc& f, const double* x)
{
    const double bump = 1e-4;
    const unsigned int gradSize = f.GetInputDim() + f.GetParameterCount();
    std::vector<double> g(gradSize, 0.0);
    std::vector<double> finiteDiff(f.GetOutputDim() * gradSize, 0.0);
    std::vector<double> base(f.GetOutputDim(), 0.0);
    DifferentialFinite(f, x, bump, /*out*/finiteDiff.data());
    const double* fd = finiteDiff.data();
    std::vector<double> y(f.GetOutputDim(), 0.0);
    for (unsigned int row = 0; row < f.GetOutputDim(); row++, fd += gradSize)
    {
        std::fill(base.begin(), base.end(), 0.0);
        base[row] = 1.0; // extract row out of differentials below
        // forward pass needed before Differential
        // (and Differential can modify f's stored intermediate values)
        f.Apply(x, /*out*/y.data());
        f.Differential(x, base.data(), /*out*/g.data(), /*out*/g.data() + f.GetInputDim());
        for (unsigned int col = 0; col < gradSize; col++)
        {
            const double gError = g[col] - fd[col];
            if (1e-3 < std::fabs(gError)) // that does not compare small gradients well
            {
                return false;
            }
        }
    }
    return true;
}

Linear::Linear(unsigned int rowCount, unsigned int columnCount)
{
    fOutputDim = rowCount;
    fInputDim = columnCount;
    fParameterCount = fInputDim * fOutputDim;
    fMatrix.resize(fParameterCount, 0.0);
}

void Linear::Fill(double m, double bias)
{
    for (double& d : fMatrix)
        d = m;
    for (double& d : fBiases)
        d = bias;
}

void Linear::Copy(const pf::Linear& other)
{
    for (unsigned int row = 0; row < other.GetOutputDim(); row++)
        for (unsigned int col = 0; col < other.GetInputDim(); col++)
            GetRow(row)[col] = other.GetRow(row)[col];
}

void Linear::Apply(const double* x, /*out*/double* y) const
{
    auto d = fMatrix.begin();
    for (unsigned int row = 0; row < fOutputDim; row++)
    {
        y[row] = std::inner_product(d, d + fInputDim, x, 0.0);
        if (!fBiases.empty())
            y[row] += fBiases[row];
        d += fInputDim;
    }
}

void Linear::ApplyTransposed(const double* x, /*out*/double* v) const
{
    // Do not add biases, because this transposed multiplication is not
    // the usual Apply. Mostly used for differential.
    auto d = fMatrix.begin();
    std::fill(v, v + fInputDim, 0.0);
    for (unsigned int row = 0; row < fOutputDim; row++)
    for (unsigned int col = 0; col < fInputDim; col++)
    {
        v[col] += x[row] * (*d);
        d++;
    }
}

void Linear::Mult(const pf::Linear& rightMatrix, /*out*/pf::Linear& y, unsigned int colOffset) const
{
    if (fInputDim != rightMatrix.fOutputDim 
        || y.fOutputDim < fOutputDim // allow writing a sub-matrix, using offset
        || y.fInputDim < rightMatrix.fInputDim)
        throw std::invalid_argument("bad matrix sizes");
    const double* rowLeft = GetRow(0);
    double* rowMult = y.GetRow(0);
    for (unsigned int row = 0; row < fOutputDim; row++, rowLeft += fInputDim, rowMult += y.fInputDim)
    for (unsigned int col = 0; col < rightMatrix.fInputDim; col++)
    {
        double& coef = rowMult[col + colOffset];
        coef = 0.0;
        for (unsigned int k = 0; k < fInputDim; k++)
            coef += rowLeft[k] * rightMatrix.GetRow(k)[col];
    }
}

double& Linear::GetParameter(unsigned int index)
{
    if (index < fMatrix.size())
        return fMatrix[index];
    index -= fMatrix.size();
    if (index < fBiases.size())
        return fBiases[index];
    throw std::invalid_argument("bad parameter");
}

void Linear::SetDiagonal(std::vector<double>&& diag)
{
    std::fill(fMatrix.begin(), fMatrix.end(), 0.0);
    for (unsigned int row = 0; row < diag.size(); row++)
        fMatrix[row * fInputDim + row] = diag[row];
}

void Linear::SetBiases(std::vector<double>&& biases)
{
    fBiases = std::move(biases);
    fParameterCount = fMatrix.size() + fBiases.size();
}

const double* Linear::GetRow(unsigned int row) const
{
    return fMatrix.data() + row * fInputDim;
}

double* Linear::GetRow(unsigned int row)
{
    return fMatrix.data() + row * fInputDim;
}

void Linear::Differential(const double* x, const double* comp,
    /*out*/double* g, /*out*/double* diffParams) const
{
    // No need to fill g and diffParams with zeros,
    // they are completely written in this function.

    // Differential with respect to biases
    if (!fBiases.empty())
    {
        const unsigned int matrixSize = fOutputDim * fInputDim;
        for (unsigned int outputIdx = 0; outputIdx < fOutputDim; outputIdx++)
            diffParams[matrixSize + outputIdx] = comp[outputIdx] /* *1.0 */;
    }

    // Differential with respect to fMatrix
    double* d = diffParams;
    for (unsigned int row = 0; row < fOutputDim; row++, d += fInputDim)
    {
        for (unsigned int col = 0; col < fInputDim; col++)
            d[col] = comp[row] * x[col];
    }

    ApplyTransposed(comp, /*out*/g); // the differential of a matrix is itself
}

Transform::Transform(ParamFunc& f, unsigned int numCalls)
: fF(f), 
  fNumCalls(numCalls),
  oDiffParams(f.GetParameterCount(), 0.0)
{
    fInputDim = numCalls * fF.GetInputDim();
    fOutputDim = numCalls * fF.GetOutputDim();
    fParameterCount = fF.GetParameterCount();
}

double& Transform::GetParameter(unsigned int index)
{
    return fF.GetParameter(index);
}

void Transform::Apply(const double* x, /*out*/double* y) const
{
    double* v = y;
    const double* uE = x + fNumCalls * fF.GetInputDim();
    for (const double* u = x; u < uE; u += fF.GetInputDim(), v += fF.GetOutputDim())
        fF.Apply(u, /*out*/v);
}

void Transform::Differential(const double* x, const double* comp,
    /*out*/double* diff, /*out*/double* diffParam) const
{
    std::fill(diffParam, diffParam + fF.GetParameterCount(), 0.0);
    double* convolDiff = diff;
    const double* convolComp = comp;
    const double* xE = x + fInputDim;
    // The differential matrix is diagonal by blocks, loop on the blocks
    for (const double* b = x; b < xE; b += fF.GetInputDim(), convolComp += fF.GetOutputDim(), convolDiff += fF.GetInputDim())
    {
        // TODO fF.Apply(b), as the precondition for Differential.
        // Since the same fF is applied to all inputs b, it cannot store
        // every call during the forward pass.
        fF.Differential(b, convolComp, /*out*/convolDiff, /*out*/oDiffParams.data());
        pf::AddScaledVector(oDiffParams.data(), oDiffParams.data() + oDiffParams.size(), 1.0,
            /*out*/diffParam);
    }
}

CartesianProduct::CartesianProduct(std::vector<std::unique_ptr<ParamFunc>>&& functions,
    bool shareParameters)
: fFunctions(std::move(functions))
{
    fInputDim = 0;
    fParameterCount = 0;
    unsigned int maxParamCount = 0;
    for (auto& f : fFunctions)
    {
        // fInputDim is the max of all input dim,
        // which allows truncatures of arguments.
        if (fInputDim < f->GetInputDim())
            fInputDim = f->GetInputDim();
        fOutputDim += f->GetOutputDim();
        fParameterCount += f->GetParameterCount();
        if (maxParamCount < f->GetParameterCount())
            maxParamCount = f->GetParameterCount();
    }
    if (shareParameters)
        fParameterCount = fFunctions[0]->GetParameterCount();
    oDiff.resize(fInputDim + maxParamCount, 0.0);
}

const std::vector<std::unique_ptr<ParamFunc>>& CartesianProduct::GetFunctions() const
{
    return fFunctions;
}

double& CartesianProduct::GetParameter(unsigned int index)
{
    if (fParameterCount == fFunctions[0]->GetParameterCount()) // share parameters
        return fFunctions[0]->GetParameter(index);
    for (auto& f : fFunctions)
    {
        if (index < f->GetParameterCount())
            return f->GetParameter(index);
        index -= f->GetParameterCount();
    }
    throw std::invalid_argument("bad parameter");
}

void CartesianProduct::Apply(const double* x, /*out*/double* y) const
{
    double* v = y;
    for (auto& f : fFunctions)
    {
        f->Apply(x, /*out*/v);
        v += f->GetOutputDim();
    }
}

void CartesianProduct::Differential(const double* x, const double* comp,
    /*out*/double* diff, /*out*/double* diffParam) const
{
    std::fill(diff, diff + fInputDim, 0.0);
    std::fill(diffParam, diffParam + fParameterCount, 0.0);
    const double* convolComp = comp;
    const bool shareParameters = (fParameterCount == fFunctions[0]->GetParameterCount());
    unsigned int paramOffset = 0;
    for (auto& f : fFunctions)
    {
        const unsigned int dim = f->GetInputDim();
        f->Differential(x, convolComp,
            /*out*/oDiff.data(), /*out*/oDiff.data() + dim);
        pf::AddScaledVector(oDiff.data(), oDiff.data() + dim, 1.0,
            /*out*/diff);
        pf::AddScaledVector(oDiff.data() + dim, oDiff.data() + dim + f->GetParameterCount(), 1.0,
            /*out*/diffParam + paramOffset);
        convolComp += f->GetOutputDim();
        if (!shareParameters)
            paramOffset += f->GetParameterCount();
    }
}

Compose::Compose(ParamFunc& f, ParamFunc& g)
: fF(f), fG(g), oV(fF.GetOutputDim(), 0.0)
{
    fInputDim = fF.GetInputDim();
    fOutputDim = fG.GetOutputDim();
    fParameterCount = fF.GetParameterCount() + fG.GetParameterCount();
}

void Compose::Apply(const double* x, /*out*/double* y) const
{
    fF.Apply(x, /*out*/oV.data());
    fG.Apply(oV.data(), /*out*/y);
}

void Compose::Differential(const double* x, const double* comp,
    /*out*/double* diff, /*out*/double* diffParam) const
{
    // Precondition: forward pass already done, oV filled.
    // Chain rule on fG \circ fF.
    std::vector<double> diffG(fF.GetOutputDim(), 0.0);
    fG.Differential(oV.data(), comp, /*out*/diffG.data(), /*out*/diffParam + fF.GetParameterCount());

    fF.Differential(x, diffG.data(), /*out*/diff, /*out*/diffParam);
}

Softmax::Softmax(ParamFunc& f)
    : fF(f)
{
    fInputDim = f.GetInputDim();
    fOutputDim = f.GetOutputDim();
    fParameterCount = f.GetParameterCount();
}

void Softmax::Apply(const double* x, /*out*/double* y) const
{
    fF.Apply(x, /*out*/y);
    double sumExp = 0.0;
    for (double* d = y; d != y + fOutputDim; d++)
    {
        *d = std::exp(*d);
        sumExp += *d;
    }
    for (double* d = y; d != y + fOutputDim; d++)
        *d /= sumExp;
}

void Softmax::Differential(const double* x, const double* comp, /*out*/double* diff, /*out*/double* diffParam) const
{
    // TODO
}

void pf::InPlaceSoftMax(/*in out*/double* scores, unsigned int size)
{
    // Subtract max score to all scores, to avoid infinite exponentials
    double max = scores[0];
    double* xE = scores + size;
    for (double* d = scores; d < xE; d++)
        if (max < *d)
            max = *d;
    double sumExp = 0.0;
    for (double* d = scores; d < xE; d++)
    {
        *d = std::exp(*d - max);
        sumExp += *d;
    }
    for (double* d = scores; d < xE; d++)
        *d /= sumExp;
}

