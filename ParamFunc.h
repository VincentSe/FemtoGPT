#include <memory>
#include <vector>
#include <functional>

namespace pf
{
    void AddScaledVector(const double* b,
        const double* e,
        double scale,
        /*out*/double* outVect);

    // Maximum of absolute values, also called infinity norm
    double MaxNorm(const double* b, const double* e);

    class ParamFunc
    {
    public:
        virtual ~ParamFunc() {}

        unsigned int GetInputDim() const;
        unsigned int GetOutputDim() const;
        unsigned int GetParameterCount() const;

        // We could think of making argument x non const, to reuse its RAM,
        // but it is usually small and the differential often needs the
        // arguments at all stages of the computation.
        virtual void Apply(const double* x, /*out*/double* y) const = 0;

        // Differential with respect to x and parameters.
        // Returns the Jacobian matrix, with fOutputDim rows and fInputDim + fParameterCount
        // columns, multiplied by line matrix comp (fOutputDim columns). This multiplication
        // optimizes RAM during a chain rule, with a final function that has output dimension 1:
        // a line matrix is backpropagated, we never allocate a full rectangular Jacobian matrix.
        // To see why backpropagation is faster than forward propagation, let's assume
        // the leftmost matrix has size (1,n), followed by k square matrices of size (n,n).
        // The number of coefficient multiplications are
        //   - backpropagation : k n^2
        //   - forward propagation : (k-1) n^3 + n^2
        // A default implementation could be provided by finite difference, but since it does
        // not access any private information of ParamFunc, it is moved outside the class
        // in function DifferentialFinite.
        virtual void Differential(const double* x,
            const double* comp, // multiply the differential by this line matrix, for gradient backpropagation
            /*out*/double* diff,
            /*out*/double* diffParams) const = 0;

        // Those 2 functions also serve as de/serialization, no need to define Save and Load methods.
        virtual double& GetParameter(unsigned int index) = 0;
        double GetParameter(unsigned int index) const;

    protected:
        unsigned int fInputDim = 0;
        unsigned int fOutputDim = 0;
        unsigned int fParameterCount = 0;
    };

    std::vector<std::pair<unsigned int, double>> SortParameters(const ParamFunc& f);
    void DifferentialFinite(const ParamFunc& f,
        const double* x,
        double bump,
        /*out*/double* diff);
    bool TestDifferential(const ParamFunc& f, const double* x);

    class Linear : public ParamFunc
    {
    public:
        Linear(unsigned int rowCount, unsigned int columnCount);

        virtual void Apply(const double* x, /*out*/double* y) const override;
        // This differential matrix is sparse, because the i-th output of
        // this ParamFunc is only sensitive to the i-th line of fMatrix.
        virtual void Differential(const double* x, const double* comp,
            /*out*/double* diff, /*out*/double* diffParams) const override;

        virtual double& GetParameter(unsigned int index) override;

        void SetDiagonal(std::vector<double>&& diag);
        void SetBiases(std::vector<double>&& biases);

        // Postcondition: compute x^t * this, row vector of length fInputDim.
        // It is equivalent to computing this^t * x.
        // Precondition: x's length is fOutputDim.
        void ApplyTransposed(const double* x, /*out*/double* y) const;

        // Matrix multiplication
        void Mult(const pf::Linear& r, /*out*/pf::Linear& y, unsigned int colOffset = 0) const;

        // This is quick because the storage is row by row
        const double* GetRow(unsigned int i) const;
        double* GetRow(unsigned int i);
        void Fill(double m, double bias);
        void Copy(const pf::Linear& other); // other is a smaller matrix, copied into upper-left corner

    private:
        // Stored row by row (fMatrix[1] is the 1-th column of the 0-th row),
        // which is how fMatrix is accessed when we multiply it by a vector,
        // i.e. its Apply function.
        std::vector<double> fMatrix;
        std::vector<double> fBiases;
    };

    class Transform : public ParamFunc
    {
    public:
        // Similar to std::transform, also called map in functional programming.
        Transform(ParamFunc& f, unsigned int numCalls);

        double& GetParameter(unsigned int index) override;

        virtual void Apply(const double* x, /*out*/double* y) const override;
        virtual void Differential(const double* x, const double* comp,
            /*out*/double* diff, /*out*/double* diffParams) const override;

    private:
        ParamFunc& fF;
        unsigned int fNumCalls{};
        mutable std::vector<double> oDiffParams;
    };

    class CartesianProduct : public ParamFunc
    {
        // Separate evaluations of several functions, all with same argument.
        // The functions also share the same parameters.
    public:
        CartesianProduct(std::vector<std::unique_ptr<ParamFunc>>&& functions,
            // 2 functions share a parameter if SetParameter on one function
            // affects Apply on the other function.
            bool shareParameters);

        double& GetParameter(unsigned int index) override;

        virtual void Apply(const double* x, /*out*/double* y) const override;
        virtual void Differential(const double* x, const double* comp,
            /*out*/double* diff, /*out*/double* diffParams) const override;

        const std::vector<std::unique_ptr<ParamFunc>>& GetFunctions() const;

    private:
        std::vector<std::unique_ptr<ParamFunc>> fFunctions;
        mutable std::vector<double> oDiff;
    };

    class Compose : public ParamFunc
    {
    public:
        // g \circ f, first apply f then apply g
        Compose(ParamFunc& f, ParamFunc& g);

        virtual void Apply(const double* x, /*out*/double* y) const override;
        virtual void Differential(const double* x, const double* comp,
            /*out*/double* diff, /*out*/double* diffParams) const override;

    private:
        ParamFunc& fF;
        ParamFunc& fG;
        mutable std::vector<double> oV;
    };

    class Softmax : public ParamFunc
    {
    public:
        Softmax(ParamFunc& f);
        virtual void Apply(const double* x, /*out*/double* y) const override;
        virtual void Differential(const double* x, const double* comp,
            /*out*/double* diff, /*out*/double* diffParams) const override;

    private:
        ParamFunc& fF;
    };

    void InPlaceSoftMax(/*in out*/double* x, unsigned int size);

    class AdamOptimizer
    {
    public:
        AdamOptimizer(unsigned int inputDim,
            double stepSize,
            std::function<double&(unsigned int)> GetParam,
            std::function<double*()> grad);

        void Step();

    private:
        unsigned int fInputDim = 0;
        std::function<double&(unsigned int)> fGetParam;
        std::function<double*()> fGrad;
        double fStepSize = 0.05;
        double beta1 = 0.9;
        double beta2 = 0.999;
	    double epsilon = 1e-8;

        // mean stores an exponential-weight average of all past
        // gradients. 0.9^50 is close to 0.005, so this average
        // considers approximately the 50 last gradients.
        // The average smoothes stochastic gradients (Monte Carlo)
        // and gradient oscillations during the descent
        // (when some dimensions vary rapidly, but not in the
        // direction of a local minimum in all dimensions).
        std::vector<double> mean;
        std::vector<double> var;
        // The sum of weights in the average
        double beta1t = 0.9;
        double beta2t = 0.999;
    };
}
