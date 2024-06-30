// Character-level transformer that generates random texts similar to an input text. This is what happens
// in the training of Chat GPT (where the input text is a large chunk of the internet, and where the
// tokens are words instead of characters).
// https://www.youtube.com/watch?v=kCc8FmEb1nY
// https://github.com/karpathy/nanoGPT
// The generation is one character at a time, using the previous n characters. n is called the context size.
// With n=5 this is similar to continuing sequences of numbers like 2 4 6 8 10 __. But instead of applying
// an a priori knowledge in arithmetic, we apply the patterns learnt in a training text.

// It is important to train the model for all contexts sizes from 0 up to n, because at the start of the
// generation the context is empty. If we try instead to start with a corrupted context (like n spaces,
// which never appears in a real text), the corruption will propagate infinitely, because the context
// rolls as more letters are generated.

// Differences with Karpathy:
// - matrix of keys merged with matrix of queries
// - no random approximation of the Loss, which is evaluated on the full training text at
//   each optimization step.
// - English syntax, shows how to mix AI with usual programming
// - cleanup of tiny Shakespeare (_ instead of --, typos, remove character 3 which was a footer)
// - gradients manually coded, to get a better idea of what autograd does
// - trains on a CPU, does not need a GPU

// Here the training text is a subset of 40k lines of Shakespeare :
// https://www.opensourceshakespeare.org/
// https://shakespeare.mit.edu/
// tiny-shakespeare.txt probably taken from the MIT website, and removed didascaliae.

// Shakespeare sonnet generator :
// https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2762063.pdf

// We start with an n-gram language model, and find it generalizes very badly. Because for n>=6 there are
// not enough n-character substrings in the training text to represent all possible n-character substrings
// of English. In other words, for n>=6 the generator rather copies entire substrings of the training text :
// that's plagiarism not generation. The problem of n-gram is rather clear : it looks for the exact same
// (n-1)-prefix in the training text, instead of searching for similar prefixes. But that notion of
// similar prefixes is very hard to define. For example if the context is "the red bli_", the last space
// indicates we are looking for a word that starts with "bli". So this context is highly correlated to
// the shorter context " bli_", which is easier to find in the training text.

// Instead of defining similar prefixes, we will learn them automatically from the data, using a language model
// called transformer. In the attention is all you need paper
// https://arxiv.org/pdf/1706.03762.pdf
// this corresponds to a decoder-only transformer (in the diagram page 3, we remove the inputs column and only
// implement the outputs column).

// Validation losses for different models:
// - Karpathy's full transformer: 165000 (equivalent of 1.48 in his youtube video cited above, at 1:40:33)
// - single-head attention, 2-layer perceptron, context 128 letters, embedding dimension 64: ????, 53568 parameters
// - single-head attention, 2-layer perceptron, context 64 letters, embedding dimension 16: 183000, 7808 parameters
// - 5-gram with bigram defaulting (plagiarism): 211415
// - single-head attention, context 32 letters, embedding dimension 16: 233000, 2816 parameters
// - 2-layer perceptron all context, context 64 letters, embedding dimension 10, hidden layer 100: 247000, 71000 parameters
// - just English syntax: 290000, 0 parameters
// - random: 463000 = 111457 * ln(64)

#include <string>
#include <set>
#include <map>
#include <random>
#include <cassert>
#include <iostream>
#include <numeric>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>

#include "ParamFunc.h"

extern std::filesystem::path sDatasetDirectory;

// std::random_device is just a wrapper around function rand_s.
// rand_s calls the operating system for cryptographically secure random integers (between 0 to UINT_MAX).
static std::random_device rd;
static std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()

std::set<char> CheckNextChars(char c)
{
    std::stringstream ss;
    std::ifstream f(sDatasetDirectory / "Shakespeare/vocabulary.txt");
    std::set<char> letters;
    std::string line;
    while (std::getline(f, /*out*/line))
    {
        if (2 <= std::count(line.begin(), line.end(), '\''))
            throw "bou";
    }
    f.close();
    return letters;
}

double BigramProbability(const std::map<std::string, unsigned int>& bigramProbas,
    char prefix, char target)
{
    std::string s;
    s += prefix;
    auto b = bigramProbas.lower_bound(s);
    s.back() = s.back() + 1;
    auto e = bigramProbas.lower_bound(s);
    unsigned int totalCount = 0;
    double proba = 0.0;
    for (auto c = b; c != e; c++)
    {
        totalCount += c->second;
        if (c->first.back() == target)
            proba = c->second;
    }
    return proba / totalCount;
}

class ScopedClock
{
public:
    ScopedClock()
    : fBegin(std::chrono::steady_clock::now()),
      fName("Unnamed clock")
    {}
    ScopedClock(const char* name)
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
    const char* fName = "";
};

class NgramModel
{
public:
    // Compute the empirical (in trainingText) probabilities of a character,
    // given an (n-1)-length prefix.
    void Train(const std::string& trainingText,
        unsigned int n) // length of the key string
    {
        fProbas.clear();
        fBigramProbas.clear();
        size_t lastChar = trainingText.length() - (n - 1);
        for (auto i = 0; i < lastChar; i++)
        {
            const std::string k = trainingText.substr(i, n);
            fProbas[k]++;
        }

        // Bigram training
        for (size_t i = 0; i < trainingText.length() - 1; i++)
        {
            const std::string k = trainingText.substr(i, 2);
            fBigramProbas[k]++;
        }

        // Add all possible bigrams from the English dictionary.
        // This is a priori knowledge about English, not information
        // from the training text.
        fBigramProbas.emplace("bw", 1);
        fBigramProbas.emplace("b?", 1);
        fBigramProbas.emplace("dt", 1);
        fBigramProbas.emplace("fn", 1);
        fBigramProbas.emplace("ji", 1);
        fBigramProbas.emplace("rz", 1);
        fBigramProbas.emplace("tb", 1);
        fBigramProbas.emplace("z'", 1);
        fBigramProbas.emplace("z!", 1);
        fBigramProbas.emplace("Aj", 1);
        fBigramProbas.emplace("DR", 1);
        fBigramProbas.emplace("EB", 1);
        fBigramProbas.emplace("E'", 1);
        fBigramProbas.emplace("FE", 1);
        fBigramProbas.emplace("G-", 1);
        fBigramProbas.emplace("IB", 1);
        fBigramProbas.emplace("IP", 1);
        fBigramProbas.emplace("NZ", 1);
        fBigramProbas.emplace("O\n", 1);
        fBigramProbas.emplace("RV", 1);
        fBigramProbas.emplace("Sy", 1);
        fBigramProbas.emplace("SP", 1);
        fBigramProbas.emplace("_D", 1);
        fBigramProbas.emplace("'N", 1);
    }

    unsigned int GetN() const
    {
        return fProbas.empty() ? 0 : static_cast<unsigned int>(fProbas.begin()->first.length());
    }

    std::set<char> GetAlphabet() const
    {
        std::set<char> alphabet;
        for (const auto& [s, _] : fProbas)
            alphabet.insert(s.begin(), s.end());
        return alphabet;
    }

    // Compare different language models by likelihood.
    // The likelihood is the probability that the model would produce the validation text.
    // However this likelihood is so small that it cannot be represented by 64-bit
    // floating-point numbers, because we have 65 characters in the alphabet and
    // the validation text has around 100k characters, so that's around (1/65)^100000,
    // even for the ideal language model.
    // We therefore compute the logarithm of this likelihood, which splits into a sum of logarithms
    // for each character drawn. That makes a negative number representable in 64 bits,
    // and we finally return its opposite to get a positive number : the loss, or surprise
    // to observe validationText given the model NgramProbabilities.
    // The loss has no upper bound, because there is no likelihood lower bound :
    // if NgramProbabilities is a bad model, its likelihood on validationText can be
    // any number towards 0, even 0 itself.
    // On the other end we want to model to accept many different validation texts
    // (the text it could generate), so the propability of each is less than 1, and
    // the loss should not be 0.
    // Might be useful to divide by the length of text, to compare training loss
    // and validation loss. For example the loss of a completely random generator is
    // text.length() * std::log(alphabet.size()), which gives std::log(alphabet.size())
    // if we divide by text.length().
    double Loss(const std::string& text) const
    {
        // Loss of completely random model, for comparison
        // const std::set<char> alphabet = GetAlphabet();
        // const double noiseLoss = text.size() * std::log(alphabet.size()); // 465598

        const size_t n = GetN();
        double loss = 0.0;
        double unexplainedLoss = 0.0;
        for (size_t i = 0; i < text.size(); i++)
        {
            // Example, validationText=="ABCDEFGH", n==5, i==4
            const std::string prefix = (n <= i + 1
                ? text.substr((i + 1) - n, n - 1)
                : "\n" + text.substr(0, i));
            auto b = fProbas.lower_bound(prefix);
            const double bigramProba = BigramProbability(fBigramProbas, prefix.back(), text[i]);
            if (prefix.length() + 1 < n || b->first.rfind(prefix, 0) != 0)
            {
                loss -= std::log(bigramProba);
            }
            else
            {
                // Can still be bigram, with 0.001 chance.
                std::string s = prefix;
                s.back() = s.back() + 1;
                auto e = fProbas.lower_bound(s);
                unsigned int totalCount = 0;
                double proba = 0.0;
                for (auto c = b; c != e; c++)
                {
                    totalCount += c->second;
                    if (c->first.back() == text[i])
                        proba = c->second;
                }
                loss -= std::log(0.001 * bigramProba + 0.999 * proba / totalCount);
            }
            if (!std::isfinite(loss))
            {
                throw "bou";
            }
        }
        return loss;
    }

    // In other words, this function uses the trained n-gram language model to generate a random character.
    char GenerateCharacter(const std::string& prefix) const
    {
        // By counting the substrings of its training text, the n-gram model does not generalize well :
        // it only generates n-substrings of its training text. We add a small probability of truncating
        // the prefix to a single character, to improve generalization (and get a finite log-likelihood
        // on the validation text). This random prefix is a first step towards the transformer model.
        unsigned int totalCount = 0;
        std::map<char, unsigned int> logits;
        auto b = fProbas.lower_bound(prefix);
        auto e = fProbas.end();
        std::uniform_int_distribution<> distribGeneralize(0, 999);
        size_t pLen = prefix.length();
        if (prefix.length() + 1 < GetN() || b->first.rfind(prefix, 0) != 0 || distribGeneralize(gen) == 0)
        {
            // Default to BiGram
            std::string lastCharacter;
            lastCharacter += prefix.back();
            b = fBigramProbas.lower_bound(lastCharacter);
            lastCharacter.back() = lastCharacter.back() + 1;
            e = fBigramProbas.lower_bound(lastCharacter);
            pLen = 1;
        }
        else
        {
            std::string s = prefix;
            s.back() = s.back() + 1;
            e = fProbas.lower_bound(s);
        }
        for (auto c = b; c != e; c++)
        {
            totalCount += c->second;
            logits[c->first[pLen]] += c->second;
        }

        // Draw random logit
        std::uniform_int_distribution<> distrib(0, totalCount - 1);
        unsigned int nextCharacterIndex = distrib(gen);
        for (auto [c, count] : logits)
        {
            if (nextCharacterIndex < count)
            {
                // Postcondition: next character is found
                return c;
            }
            else
                nextCharacterIndex -= count;
        }
        assert(false); // logit not found
        return 0;
    }

    std::string GenerateText() const
    {
        // Character based n-grams make invalid words, word-based n-grams would be better.
        // And we should use forward context to understand sentences like
        // "The bear walks towards the honey pot because it is ______".
        // This unknown next word tells whether "it" refers to "bear" or to "honey".

        const unsigned int n = GetN();
        std::string prefix = "\n";
        std::string s;
        for (int i = 0; i < 1000; i++)
        {
            const char c = GenerateCharacter(prefix);
            s += c;
            prefix += c;
            if (n <= prefix.length())
                prefix = prefix.substr(1, n - 1);
        }
        return s;
    }

private:
    std::map<std::string, unsigned int> fProbas;
    // Bigram is used rarely instead of full n-gram, to improve generalization
    std::map<std::string, unsigned int> fBigramProbas;
};

class AttentionHead : public pf::ParamFunc
{
public:
    // An attention head works on a sequence of characters of size fContextLength,
    // embedded each as tokenEmbeddingDim floating-point numbers. Attention replaces
    // each character by a weighted average of its previous characters, via the
    // triangular matrix oWeights. These weights intend to sum up all past information
    // at each character, for the purpose of predicting the next character.
    // An attention head stores a square matrix H (alias fQueryMatrix) that defines
    // he affinity between two tokens x,y by the inner product <| x, Hy |>.
    // That's arguably one of the simplest learnable affinity formulas.
    // Those affinities go through a softmax to produce oWeights. To reduce the number of
    // parameters in H, we split it as a product of 2 rectangular matrices fKeys
    // and fQueries (which usually degenerates H's rank). Finally the weights are
    // applied to another parameter matrix fValues to produce the output.

    // It receives fContextLength token embeddings (which also encode token positions)
    // and produces logits for the next token.

    AttentionHead(unsigned int contextLength,
        unsigned int tokenEmbeddingDim,
        unsigned int outputDim)
    : fContextLength(contextLength),
      fScale(std::sqrt(static_cast<double>(tokenEmbeddingDim))),
      fQueryMatrix(tokenEmbeddingDim, tokenEmbeddingDim),
      fValueMatrix(outputDim, tokenEmbeddingDim),
      oWeights(contextLength, 0.0),
      oAverageToken(tokenEmbeddingDim, 0.0),
      oQuery(fQueryMatrix.GetOutputDim(), 0.0),
      oQt(fQueryMatrix.GetInputDim(), 0.0),
      oDiffV(fValueMatrix.GetInputDim(), 0.0)
    {
        fInputDim = contextLength * tokenEmbeddingDim;
        fOutputDim = outputDim;
        fParameterCount = fQueryMatrix.GetParameterCount() + fValueMatrix.GetParameterCount();
    }

    unsigned int GetTokenEmbeddingDim() const
    {
        return fQueryMatrix.GetInputDim();
    }

    unsigned int GetContextLength() const
    {
        return fContextLength;
    }

    unsigned int& GetContextLength()
    {
        // Allows to truncate context, for example at the beginning of generation
        return fContextLength;
    }

    const pf::Linear& GetQueryMatrix() const
    {
        return fQueryMatrix;
    }

    const pf::Linear& GetValueMatrix() const
    {
        return fValueMatrix;
    }

    virtual double& GetParameter(unsigned int index) override
    {
        const unsigned int queryParams = fQueryMatrix.GetParameterCount();
        if (index < queryParams)
            return fQueryMatrix.GetParameter(index);
        index -= queryParams;
        return fValueMatrix.GetParameter(index);
    }

    // Softmax of the embedded tokens.
    // Number of multiplications for query and inner products:
    // tokenDim^2 + tokenDim * fContextLength.
    // If we separate with a fKeyMatrix, then
    // query = fKeyMatrix^t * fQueryMatrix * last token
    // which takes 2*d*tokenDim multiplications (right multiplication first),
    // where d is the common output dim of fKeyMatrix and fQueryMatrix.
    // It is faster if d < tokenDim/2, and takes less parameters.
    void Weights(const double* tokensB,
        /*out*/std::vector<double>& weights,
        /*out*/std::vector<double>& query) const
    {
        unsigned int tokenDim = GetTokenEmbeddingDim();
        const double* tokensE = tokensB + fContextLength * tokenDim;
        auto w = weights.begin();
        const double* lastToken = tokensE - tokenDim;
        fQueryMatrix.Apply(lastToken, /*out*/query.data());
        for (const double* token = tokensB; token <= lastToken; token += tokenDim)
        {
            // All these inner products with query could be packed
            // as a matrix multiplication by query.
            *w = std::inner_product(query.begin(), query.end(), token, 0.0) / fScale;
            w++;
        }
        pf::InPlaceSoftMax(weights.data(), fContextLength);
    }

    void AverageToken(const double* tokensB,
        const std::vector<double>& weights,
        /*out*/std::vector<double>& averageToken) const
    {
        const unsigned int tokenDim = GetTokenEmbeddingDim();
        const double* tokensE = tokensB + fContextLength * tokenDim;
        std::fill(averageToken.begin(), averageToken.end(), 0.0);
        auto w = weights.begin();
        for (const double* token = tokensB; token < tokensE; token += tokenDim, w++)
            pf::AddScaledVector(token, token + tokenDim, *w, /*out*/averageToken.data());
    }

    // The logits (aka scores) for each next token to predict.
    // The logits will go through a softmax to be converted into probabilities.
    virtual void Apply(const double* tokens, /*out*/double* pastAverage) const override
    {
        Weights(tokens, /*out*/oWeights, /*out*/oQuery);
        AverageToken(tokens, oWeights, /*out*/oAverageToken);
        fValueMatrix.Apply(oAverageToken.data(), /*out*/pastAverage); // faster than applying fValues on each token before average
        oPastAggregate = pastAverage;
    }

    // Differential of affinity_k with respect to tokens and fQueryMatrix
    void DifferentialAffinity(unsigned int k,
        const double* tokenK,
        const double* lastToken, // i.e. token at position fContextLength - 1
        const double* q, // = fQueryMatrix * lastToken
        double comp,
        /*out*/double* diff,
        /*out*/double* diffQuery) const
    {
        // affinity_k = <| fQueryMatrix * lastToken, token_k |> / fScale
        // d affinity_k = <| d (fQueryMatrix * lastToken), token_k |>
        //                + <| fQueryMatrix * lastToken, d token_k |>
        const unsigned int tokenEmbeddingDim = GetTokenEmbeddingDim();

        // Differential with respect to tokenK
        pf::AddScaledVector(q, q + tokenEmbeddingDim, comp, /*out*/diff + (k * tokenEmbeddingDim));

        // Differential with respect to lastToken (factorized to avoid multiplications by fQueryMatrix)
        // fQueryMatrix.ApplyTransposed(tokenK, /*out*/oQt.data()); // fQt could be split by i
        // pf::AddScaledVector(oQt.data(), oQt.data() + tokenDim, comp,
        //    /*out*/diff + ((fContextLength - 1) * tokenDim));
        pf::AddScaledVector(tokenK, tokenK + tokenEmbeddingDim, comp, /*out*/oQt.data());

        // Differential with respect to queries
        for (unsigned int j = 0; j < tokenEmbeddingDim; j++)
        {
            // *lastToken[col] is factorized outside of DifferentialAffinity, to accelerate
            diffQuery[j * tokenEmbeddingDim] += comp * tokenK[j];
        }
    }

    // The gradient of a real function has the same shape as the function's parameters.
    // The output stores the partial derivatives with respect to the corresponding parameters.
    virtual void Differential(const double* tokens, const double* comp,
        /*out*/double* diff, /*out*/double* diffParams) const override
    {
        // Precondition: the forward pass was already done.

        //ScopedClock clk("Attention differential");
        const unsigned int tokenEmbeddingDim = GetTokenEmbeddingDim();
        const double* lastToken = tokens + (fContextLength - 1) * tokenEmbeddingDim;

        std::fill(diff, diff + fInputDim, 0.0);
        std::fill(diffParams, diffParams + fParameterCount, 0.0);

        fValueMatrix.Differential(oAverageToken.data(), comp, 
            /*out*/oDiffV.data(), /*out*/diffParams + fQueryMatrix.GetParameterCount());

        // Now the differential of average token = sum_k weight_k * token_k
        std::fill(oQt.begin(), oQt.end(), 0.0);
        const double averageDiffV = std::inner_product(oDiffV.begin(), oDiffV.begin() + tokenEmbeddingDim, oAverageToken.begin(), 0.0);
        for (unsigned int k = 0; k < fContextLength; k++)
        {
            // Differential with respect to token_k :
            // weight_k * d token_k
            pf::AddScaledVector(oDiffV.data(), oDiffV.data() + tokenEmbeddingDim, oWeights[k],
                /*out*/diff + (k * tokenEmbeddingDim));

            // Differential of softmax :
            // d weight_k = weight_k * d affinity_k - weight_k * \sum_j weight_j * d affinity_j

            // Differential with respect to weights :
            // \sum_k d weight_k * token_k
            // = \sum_k weight_k * token_k * d affinity_k
            //   - \sum_k weight_k * (\sum_j weight_j * d affinity_j) * token_k
            // = \sum_k (token_k - averageToken) * weight_k * d affinity_k
            const double* token_k = tokens + k * tokenEmbeddingDim;
            const double x = std::inner_product(oDiffV.begin(), oDiffV.begin() + tokenEmbeddingDim, token_k, 0.0) - averageDiffV;
            DifferentialAffinity(k, token_k, lastToken, oQuery.data(),
                x * oWeights[k] / fScale,
                /*out*/diff, /*out*/diffParams);
        }

        // Add differential with respect to last token, that was factorized in DifferentialAffinity:
        // oQt * fQueryMatrix
        double* diffLast = diff + ((fContextLength - 1) * tokenEmbeddingDim);
        const double* q = fQueryMatrix.GetRow(0);
        for (unsigned int j = 0; j < tokenEmbeddingDim; j++)
        {
            const double diffQuery = diffParams[j * tokenEmbeddingDim];
            for (unsigned int col = 0; col < tokenEmbeddingDim; col++)
            {
                diffParams[j * tokenEmbeddingDim + col] = diffQuery * lastToken[col];
                diffLast[col] += oQt[j] * (*q);
                q++;
            }
        }
    }

private:
    unsigned int fContextLength = 0;
    // Scaled attention, prevents the softmax from becoming true max
    // (which differential is 0, because its value is 1 on the maximum
    // coordinate and 0 on all other coordinates). In other words,
    // true max would force a predicted character (probability 1),
    // and the parameters would have no (differential) effect on that
    // prediction.
    double fScale = 1.0;
    // For the moment the alphabet is small (65 symbols) and we take it
    // as the token embedding dimension. So we accept a full square
    // fQueries matrix of size 65*65=4225. If the embedding dimension
    // grows, we will reduce the attention's dimension by squashing
    // fQueries into a rectangular matrix and adding another rectangular
    // matrix fKeys, the product of which will be the square attention
    // matrix (with degenerate rank).
    pf::Linear fQueryMatrix;
    // fValueMatrix reduces the number of parameters when there are
    // n heads of attention. The n fValueMatrix have
    // n * tokDim * tokDim / n = tokDim * tokDim
    // parameters. And they reduce the input of the TLP to tokDim,
    // parameters, so its first matrix has tokDim * 4 * tokDim
    // parameters. Without fValueMatrix, the TLP would have a
    // first matrix with n * tokDim * 4 * tokDim parameters.
    pf::Linear fValueMatrix;

    mutable std::vector<double> oWeights;
    mutable std::vector<double> oAverageToken;
    mutable std::vector<double> oQuery;
    mutable std::vector<double> oQt;
    mutable std::vector<double> oDiffV;
    mutable double* oPastAggregate = nullptr;
};

class TLP : public pf::ParamFunc
{
public:
    // 2-layer perceptron, i.e. fully connected neural network
    // with only one hidden layer. By the universal approximation theorem,
    // those can approximate any function, but at the cost of a huge
    // number of neurons in the hidden layer.

    TLP(unsigned int inputDim,
        unsigned int hiddenLayerDim,
        unsigned int outputDim,
        bool biases)
        : fU(hiddenLayerDim, inputDim),
        fV(outputDim, hiddenLayerDim),
        oU(fU.GetOutputDim(), 0.0)
    {
        fInputDim = inputDim;
        fOutputDim = outputDim;
        fParameterCount = fU.GetParameterCount() + fV.GetParameterCount();
        for (unsigned int i = 0; i < GetParameterCount(); i++)
        {
            const double epsilon = (i % 2 == 0 ? 1.0 : -1.0);
            GetParameter(i) = epsilon * (i * 2.0 / GetParameterCount() - 1.0);
        }
        if (biases)
            SetBiases(std::vector<double>(hiddenLayerDim, 0.0), std::vector<double>(outputDim, 0.0));
    }

    virtual double& GetParameter(unsigned int index) override
    {
        const unsigned int queryParams = fU.GetParameterCount();
        if (index < queryParams)
            return fU.GetParameter(index);
        index -= queryParams;
        const unsigned int valueParams = fV.GetParameterCount();
        if (index < valueParams)
            return fV.GetParameter(index);
        throw std::invalid_argument("bad parameter");
    }

    void SetU(unsigned int index, double d)
    {
        fU.GetParameter(index) = d;
    }

    void SetV(unsigned int index, double d)
    {
        fV.GetParameter(index) = d;
    }

    const pf::Linear& GetV() const
    {
        return fV;
    }

    virtual void Apply(const double* x, double* y) const override
    {
        // const_cast<TLP*>(this)->CenterBias();

        fU.Apply(x, /*out*/oU.data());
        for (double& d : oU)
            if (d < 0.0)
                d = 0.0; // ReLU
        fV.Apply(oU.data(), /*out*/y);
    }

    virtual void Differential(const double* x, const double* comp,
        /*out*/double* diff, /*out*/double* diffParams) const override
    {
        //ScopedClock clk("TLP differential");
        std::fill(diff, diff + fInputDim, 0.0);
        std::fill(diffParams, diffParams + GetParameterCount(), 0.0);

        // RELU optimization: recode part of fV's Differential here
        // to skip the coefficients that are blocked by the RELU.
        // Differential with respect to V's parameters
        const unsigned int VmatrixSize = fV.GetOutputDim() * fV.GetInputDim();
        double* diffVParams = diffParams + fU.GetParameterCount();
        for (unsigned int outputIdx = 0; outputIdx < fOutputDim; outputIdx++)
        {
            const double coef = comp[outputIdx];
            // d biasVrow
            if (VmatrixSize < fV.GetParameterCount()) // Postcondition: fV has bias
                diffVParams[VmatrixSize + outputIdx] = coef /* *1.0 */;
            // dVrow * sigma(u)
            for (unsigned int i = 0; i < fU.GetOutputDim(); i++)
                diffVParams[outputIdx * fV.GetInputDim() + i] = coef * oU[i];
        }

        // Write dLoss / dU in oU
        for (unsigned int outputIdx = 0; outputIdx < fU.GetOutputDim(); outputIdx++)
        {
            double& o = oU[outputIdx];
            if (o == 0.0)
                continue; // This node is blocked by the RELU, so it does not affect the loss.
            o = 0.0;
            for (unsigned int row = 0; row < fOutputDim; row++)
                o += comp[row] * fV.GetRow(row)[outputIdx];
        }

        fU.Differential(x, oU.data(), /*out*/diff, /*out*/diffParams);
    }

    // fV's bias is sent into a softmax, so we can subtract its mean
    void CenterBias()
    {
        const unsigned int vSize = fV.GetInputDim() * fV.GetOutputDim();
        double mean = 0.0;
        for (unsigned int i = 0; i < fV.GetOutputDim(); i++)
            mean += fV.GetParameter(vSize + i);
        mean /= fV.GetOutputDim();
        for (unsigned int i = 0; i < fV.GetOutputDim(); i++)
            fV.GetParameter(vSize + i) = fV.GetParameter(vSize + i) - mean;
    }

private:
    void SetBiases(std::vector<double>&& Ubias, std::vector<double>&& Vbias)
    {
        fU.SetBiases(std::move(Ubias));
        fV.SetBiases(std::move(Vbias));
        fParameterCount = fU.GetParameterCount() + fV.GetParameterCount();
    }

    pf::Linear fU;
    pf::Linear fV;

    mutable std::vector<double> oU;
};

class AttentionStack : public pf::ParamFunc
{
public:
    AttentionStack(unsigned int contextLength,
        unsigned int tokenEmbeddingDim,
        unsigned int attentionHeads,
        unsigned int alphabetSize)
    : fFirstAttention(BuildAttentionHeads(contextLength, tokenEmbeddingDim, attentionHeads), false),
      fTLP(tokenEmbeddingDim, 4*tokenEmbeddingDim, alphabetSize, true),
      oPastAverage(tokenEmbeddingDim, 0.0),
      oLogits(alphabetSize, 0.0),
      oConvolDiff(fTLP.GetInputDim(), 0.0)
    {
        fInputDim = contextLength * tokenEmbeddingDim;
        fOutputDim = alphabetSize;
        fParameterCount = fFirstAttention.GetParameterCount() + fTLP.GetParameterCount();
    }

    AttentionStack(const AttentionStack& other) = delete;
    AttentionStack& operator=(const AttentionStack& other) = delete;
    AttentionStack(AttentionStack&& other) = default;
    AttentionStack& operator=(AttentionStack&& other) = default;

    std::vector<std::unique_ptr<ParamFunc>> BuildAttentionHeads(unsigned int contextLength,
        unsigned int tokenEmbeddingDim,
        unsigned int attentionHeads)
    {
        std::vector<std::unique_ptr<ParamFunc>> ret;
        ret.reserve(contextLength);
        fContextLengths.resize(attentionHeads, nullptr);
        for (unsigned int i = 0; i < attentionHeads; i++)
        {
            auto head = std::make_unique<AttentionHead>(contextLength, tokenEmbeddingDim, tokenEmbeddingDim / attentionHeads);
            fContextLengths[i] = &head->GetContextLength();
            ret.emplace_back(std::move(head));
        }
        return ret;
    }

    const pf::CartesianProduct& GetHeads() const
    {
        return fFirstAttention;
    }

    const TLP& GetTLP() const
    {
        return fTLP;
    }

    double& GetParameter(unsigned int index) override
    {
        const unsigned int fstAttParams = fFirstAttention.GetParameterCount();
        if (index < fstAttParams)
            return fFirstAttention.GetParameter(index);
        index -= fstAttParams;
        const unsigned int tlpParams = fTLP.GetParameterCount();
        if (index < tlpParams)
            return fTLP.GetParameter(index);
        throw std::invalid_argument("bad parameter");
    }

    void SetContextLength(unsigned int contextLen)
    {
        // Allows to truncate context, for example at the beginning of generation
        for (unsigned int* p : fContextLengths)
            *p = contextLen;
    }

    std::vector<double>& GetLogits() const
    {
        // the final logits, output of this AttentionStack
        return oLogits;
    }

    void Apply(const double* tokens, /*out*/double* logitsAllTokens) const override
    {
        // TODO layer norm tokens here (if exp args are too big)
        fFirstAttention.Apply(tokens, /*out*/oPastAverage.data());
        // TODO layer norm a here (if exp args are too big)
        fTLP.Apply(oPastAverage.data(), /*out*/logitsAllTokens);
    }

    virtual void Differential(const double* tokens, const double* comp,
        /*out*/double* diff, /*out*/double* diffParams) const override
    {
        // Precondition: the forward pass was already done (for example in DifferentialLoss).

        // Backpropagation of the gradient
        fTLP.Differential(oPastAverage.data(), comp,
            /*out*/oConvolDiff.data(),
            /*out*/diffParams + fFirstAttention.GetParameterCount());
        fFirstAttention.Differential(tokens, oConvolDiff.data(),
            /*out*/diff, /*out*/diffParams);
    }

private:
    std::vector<unsigned int*> fContextLengths;
    pf::CartesianProduct fFirstAttention;
    // there should be a single linear convolution at the end for tokDim -> alphabetSize.
    // For now we take alphabetSize directly as the output dim of fTLP.
    TLP fTLP;

    mutable std::vector<double> oPastAverage;
    mutable std::vector<double> oLogits;
    mutable std::vector<double> oConvolDiff;
};

// Compare std::tolower(word) with lowerWord which has no caps.
int StrcmpLower(const char* wordB, const char* wordE, bool incrLastLetter,
    const char* lowerWord)
{
    while (1)
    {
        if (wordB == wordE)
        {
            // Postcondition: word is finished
            if (*lowerWord == 0)
                return 0; // word == lowerWord
            else
                return -1; // word < lowerWord
        }
        if (*lowerWord == 0)
            return 1; // lowerWord < word
        char c = std::tolower(*wordB);
        if (incrLastLetter && wordB == wordE - 1)
            c++;
        if (c < *lowerWord)
            return -1; // word < lowerWord
        else if (*lowerWord < c)
            return 1; // lowerWord < word
        wordB++;
        lowerWord++;
    }
}

std::string Alphabet(const std::string& text)
{
    std::set<char> s(text.begin(), text.end());
    return std::string(s.begin(), s.end());
}

unsigned int GetCharacterIndex(const std::string& alphabet, char c)
{
    auto it = std::lower_bound(alphabet.begin(), alphabet.end(), c);
    return (it != alphabet.end() && *it == c) ? it - alphabet.begin() : alphabet.size();
}

// Get a valid prefix by going back to the first letter of a word
// sumExp == 0, context was ' in bigger text hat, man! 'tis not so
// TODO call this function in AllowWord
void MoveToStartOfWord(const std::string& text, /*out*/unsigned int& i)
{
    // Same as AllowWord. A word may contain apostrophes.
    char c = std::tolower(text[i]);
    if (('a' <= c && c <= 'z') || c == '-' || c == '\'')
    {
        // i is probably inside a word, except if start or end of quotation.
        while (0 < i && text[i - 1] != ' ' && text[i - 1] != '\n' && text[i - 1] != ',' && text[i - 1] != ';'
            && text[i - 1] != '_') // dash
            i--;
    }
}

class CharGenerator
{
    // Embed prefix to continuous space, call prediction network to produce logits,
    // filter logits by English rules, and softmax to get probabilities for character.

public:
    CharGenerator(unsigned int contextLength,
        unsigned int characterEmbeddingDim,
        unsigned int attentionHeads,
        unsigned int threadCount,
        const std::string& alphabet)
    : fAlphabet(Alphabet(alphabet)),
      fCharacterEmbeddings(static_cast<unsigned int>(fAlphabet.size()), characterEmbeddingDim), // GetRow(i) gives the i-th embedding
      fPositionEmbeddings(contextLength, characterEmbeddingDim),
      fComputeLogits(contextLength, characterEmbeddingDim, attentionHeads, fAlphabet.size()),
      fComment("// comment this character generator here"),
      oEmbeddings(fComputeLogits.GetInputDim(), 0.0),
      oDiffLoss(1, GetParameterCount()),
      oDiffOneSample(1, GetParameterCount())
    {
        std::ifstream f(sDatasetDirectory / "Shakespeare/vocabulary.txt");
        std::string word;
        while (std::getline(f, /*out*/word))
            fVocabulary.push_back(word);

        std::mt19937 mt; // Mersenne twister pseudo-random generator, with a fixed default (not random) seed.
        std::uniform_real_distribution<> distrib(-1.0, 1.0);
        for (unsigned int i = 0; i < GetParameterCount(); i++)
            GetParameter(i) = distrib(mt);

        if (0 < threadCount)
            fChildren.reserve(threadCount - 1);
        for (unsigned int t = 1; t < threadCount; t++)
            fChildren.emplace_back(contextLength, characterEmbeddingDim, attentionHeads, 0, alphabet);
    }

    unsigned int GetParameterCount() const
    {
        return fCharacterEmbeddings.GetParameterCount()
            + fPositionEmbeddings.GetParameterCount() + fComputeLogits.GetParameterCount();
    }

    unsigned int GetHeadCount() const
    {
        return fComputeLogits.GetHeads().GetFunctions().size();
    }

    const std::vector<CharGenerator>& GetChildren() const
    {
        return fChildren;
    }

    double& GetParameter(unsigned int index)
    {
        const unsigned int charParams = fCharacterEmbeddings.GetParameterCount();
        if (index < charParams)
            return fCharacterEmbeddings.GetParameter(index);
        index -= charParams;
        const unsigned int posParams = fPositionEmbeddings.GetParameterCount();
        if (index < posParams)
            return fPositionEmbeddings.GetParameter(index);
        index -= posParams;
        return fComputeLogits.GetParameter(index);
    }

    double GetParameter(unsigned int index) const
    {
        return const_cast<CharGenerator*>(this)->GetParameter(index);
    }

    void SyncChildren()
    {
        for (unsigned int i = 0; i < GetParameterCount(); i++)
            for (CharGenerator& child : fChildren)
                child.GetParameter(i) = GetParameter(i); // the parameters could be shared instead of copied
    }

    const pf::Linear& GetCharacterEmbeddings() const
    {
        return fCharacterEmbeddings;
    }

    const pf::Linear& GetPositionEmbeddings() const
    {
        return fPositionEmbeddings;
    }

    const AttentionStack& GetAttentionStack() const
    {
        return fComputeLogits;
    }

    unsigned int GetContextLength() const
    {
        return fComputeLogits.GetInputDim() / GetTokenEmbeddingDim();
    }

    unsigned int GetTokenEmbeddingDim() const
    {
        return fCharacterEmbeddings.GetInputDim(); // column count
    }

    // The characters generated by this CharGenerator
    const std::string& GetAlphabet() const
    {
        return fAlphabet;
    }

    std::string GetComment() const
    {
        return fComment;
    }

    void SetComment(const std::string& comment)
    {
        fComment = comment;
    }

    pf::Linear& GetDiffLoss() const
    {
        return oDiffLoss;
    }

    pf::Linear& GetDiffLossOneSample() const
    {
        return oDiffOneSample;
    }

    const std::vector<double>& NextCharProbabilities(const char* prefixB, const char* prefixE) const
    {
        unsigned int contextLen = prefixE - prefixB;
        if (GetContextLength() < contextLen)
        {
            // Postcondition: prefix too long, truncate it.
            contextLen = GetContextLength();
            prefixB = prefixE - contextLen;
        }
        Embed(prefixB, prefixE, /*out*/oEmbeddings);
        std::vector<double>& logits = fComputeLogits.GetLogits();
        const_cast<AttentionStack&>(fComputeLogits).SetContextLength(contextLen); // const_cast restored 2 lines below
        fComputeLogits.Apply(oEmbeddings.data(), /*out*/logits.data());
        const_cast<AttentionStack&>(fComputeLogits).SetContextLength(GetContextLength());
        EnglishSyntax(prefixB, prefixE, /*out*/logits.data());

        pf::InPlaceSoftMax(/*in out*/logits.data(), logits.size());
        return logits;
    }

    // Differential of the negative log-likelihood of this sample,
    // which is predicting the character at contextE, given
    // the context [contextB, contextE[.
    // Would be more consistent to define differential of
    // NextCharProbabilities, but there is a little optimization
    // to start with the loss instead.
    void DifferentialLossOneSample(const char* contextB,
        const char* contextE,
        /*out*/pf::Linear& g) const
    {
        // ScopedClock clk("DifferentialLoss sample clock");
        unsigned int contextLen = contextE - contextB;
        if (GetContextLength() < contextLen)
        {
            // Postcondition: prefix too long, truncate it.
            contextLen = GetContextLength();
            contextB = contextE - contextLen;
        }

        // Forward pass to compute the loss and intermediate values needed by chain rule.
        // The time of this forward pass is negligible, because in stochastic gradient
        // we evaluate it on a small portion of the samples.
        NextCharProbabilities(contextB, contextE);

        // Now backpropagation of the loss' gradient (chain rule)

        // d(log(sumExp)) = (d sumExp) / sumExp = \sum_{logit} exp(logit) (d logit) / sumExp
        // In other words, diffLoss is the vector of probabilities,
        // minus 1 on the probability of the observed value.
        // An increase of each bad logit increases the loss,
        // but an increase of the observed logit decreases the loss.
        std::vector<double>& logits = fComputeLogits.GetLogits();
        logits[GetCharacterIndex(fAlphabet, *contextE)] -= 1.0;

        // Between fLoss and fComputeLogits there should be the differential of EnglishSyntax.
        // EnglishSyntax is an affine function, so its differential is itself, a square
        // diagonal matrix with only 0's and 1's. Its multiplication on fLoss.GetDiffLossWrtLogits()
        // just sets some components of fLoss.GetDiffLossWrtLogits() to 0.
        // This is already done, because fLoss.GetDiffLossWrtLogits() is the vector of probabilities,
        // minus 1 on the proba of the next observed character : the probabilities of the characters
        // blocked by EnglishSyntax are already 0. 

        const_cast<AttentionStack&>(fComputeLogits).SetContextLength(contextLen); // const_cast restored 2 lines below
        double* gRow = g.GetRow(0);
        fComputeLogits.Differential(oEmbeddings.data(), logits.data(),
            /*out*/gRow + fCharacterEmbeddings.GetParameterCount(), // fills contextLen * tokenDim numbers
            /*out*/gRow + fCharacterEmbeddings.GetParameterCount() + fPositionEmbeddings.GetParameterCount());
        const_cast<AttentionStack&>(fComputeLogits).SetContextLength(GetContextLength());
        // Postcondition: the last fPositionEmbeddings.GetParameterCount() + fComputeLogits.GetParameterCount()
        // components of g are correct.

        // Finish g with respect to fCharacterEmbeddings.GetParameterCount().
        const unsigned int tokDim = GetTokenEmbeddingDim();
        const double* posEmbd = gRow + fCharacterEmbeddings.GetParameterCount();
        for (unsigned int c = 0; c < contextLen; c++, posEmbd += tokDim)
        {
            pf::AddScaledVector(posEmbd, posEmbd + tokDim, 1.0,
                /*out*/gRow + (GetCharacterIndex(fAlphabet, contextB[c]) * tokDim));
        }
    }


private:
    // Convert textual context into floating-point numbers
    void Embed(const char* contextB, const char* contextE,
        /*out*/std::vector<double>& embeddings) const
    {
        double* outEmbed = embeddings.data();
        const unsigned int tokenDim = GetTokenEmbeddingDim();
        const unsigned int contextLen = contextE - contextB;
        for (unsigned int i = 0; i < contextLen; i++, outEmbed += tokenDim)
        {
            const double* embedding = fCharacterEmbeddings.GetRow(GetCharacterIndex(fAlphabet, contextB[i]));
            std::copy(embedding, embedding + tokenDim, /*out*/outEmbed);
            if (i < fPositionEmbeddings.GetOutputDim())
            {
                const double* posEmbedding = fPositionEmbeddings.GetRow(i);
                pf::AddScaledVector(posEmbedding, posEmbedding + tokenDim, 1.0, /*out*/outEmbed);
            }
        }
    }

    void PrintParameters()
    {
        std::cout << "Character embeddings:" << std::endl;
        unsigned int param = 0;
        for (size_t i = 0; i < fAlphabet.size(); i++)
        {
            std::cout << fAlphabet[i] << ": ";
            for (unsigned int j = 0; j < GetTokenEmbeddingDim(); j++)
            {
                std::cout << fCharacterEmbeddings.GetParameter(param) << " ";
                param++;
            }
            std::cout << std::endl;
        }
    }

    std::vector<std::string>::const_iterator FindWordPrefix(const char* wordPrefixB, const char* wordPrefixE,
        bool incrLastLetter) const // to find end of range
    {
        auto b = fVocabulary.begin();
        auto e = fVocabulary.end();
        while (b != e)
        {
            auto mid = b + (e - b) / 2; // if range [b,e[ is a singleton, mid == b
            const int cmp = StrcmpLower(wordPrefixB, wordPrefixE, incrLastLetter, mid->c_str());
            if (0 == cmp)
            {
                return mid;
            }
            else if (cmp < 0)
            {
                // Postcondition: wordPrefix < mid
                e = mid;
            }
            else
            {
                // Postcondition: mid < wordPrefix
                if (mid == b)
                    return e;
                b = mid;
            }
        }
        // Postcondition: b == e, so it does not matter which we return
        return b;
    }

    // Find next characters allowed from wordPrefix and fVocabulary
    void AllowWordPrefix(const char* wordPrefixB, const char* wordPrefixE,
        /*out*/bool* allowedNextChars) const
    {
        const bool allCaps = std::all_of(wordPrefixB, wordPrefixE,
            [](char c) { return std::isupper(c) || c == '-'; });
        auto it = FindWordPrefix(wordPrefixB, wordPrefixE, false);
        // The prefix is just 1 letter for each word start in the text,
        // faster to compute another dichotomy than compare with every
        // word starting with a given letter.
        const char charE = wordPrefixE[-1];
        auto it2 = FindWordPrefix(wordPrefixB, wordPrefixE, true);
        const size_t wordPrefixSize = wordPrefixE - wordPrefixB;
        for (auto vocabWord = it; vocabWord != it2; vocabWord++)
        {
            if (wordPrefixSize == vocabWord->size()) // end of word
            {
                allowedNextChars['\n'] = true;
                allowedNextChars[' '] = true;
                allowedNextChars['.'] = true;
                allowedNextChars['?'] = true;
                allowedNextChars[','] = true;
                allowedNextChars[';'] = true;
                allowedNextChars['!'] = true;
                allowedNextChars[':'] = true;
                allowedNextChars['_'] = true; // dash
                allowedNextChars['\''] = true;
            }
            else
            {
                // Postcondition: wordPrefixSize < vocabWord->size()
                if (allCaps || charE == '-')
                    allowedNextChars[std::toupper((*vocabWord)[wordPrefixSize])] = true;
                if (!allCaps || wordPrefixE - wordPrefixB == 1)
                    allowedNextChars[std::tolower((*vocabWord)[wordPrefixSize])] = true;
            }
        }
    }

    void AllowWord(const char* prefixB, const char* prefixE, /*out*/bool* allowedNextChars) const
    {
        // Capital letters start each verse line, not prose.
        // This rule could be coded here.
        const char lastToken = prefixE[-1];
        const char* const wordPrefixE = prefixE;
        const char* space = wordPrefixE - 1;
        while (prefixB <= space && *space != ' ' && *space != '\n' && *space != ',' && *space != ';'
            && *space != '_') // dash
            space--;
        const char* wordPrefixB = space + 1;
        AllowWordPrefix(wordPrefixB, wordPrefixE, /*out*/allowedNextChars);
        if ((*space == '\n' || *space == ' ' || *space == ',' || *space == '_') // dash
            && space[1] == '\'')
        {
            // Allow start of quote, for example "\n'fore" can either mean "afore"
            // or quote starting with "fore".
            wordPrefixB = space + 2;
            if (wordPrefixB < wordPrefixE)
                AllowWordPrefix(wordPrefixB, wordPrefixE, /*out*/allowedNextChars);
        }
        if (lastToken == '-')
            allowedNextChars['\n'] = true; // allow to break hyphens, like pack-saddle
    }

    // If the syntax rules are too laxist (possibly no rules at all),
    // the generator will produce corrupt English, and then possibly freeze.
    // If the syntax rules are too strict, the loss is infinite,
    // the gradient descent does not converge at all.
    void EnglishSyntax(const char* prefixB, const char* prefixE, double* logits) const
    {
        const char lastToken = prefixE[-1];
        const char prevToken = (2 <= prefixE - prefixB) ? prefixE[-2] : '\0';
        bool allowedNextChars[256]{};
        if (lastToken == '\n')
        {
            allowedNextChars['\''] = true;
            if (prevToken != '\n') // 2 new lines max
                allowedNextChars['\n'] = true;
            allowedNextChars['_'] = true; // dash
            for (unsigned char c = 'a'; c <= 'z'; c++)
                allowedNextChars[c] = true;
            for (unsigned char c = 'A'; c <= 'Z'; c++)
                allowedNextChars[c] = true;
        }
        else if (lastToken == ' ')
        {
            for (unsigned char c = 'a'; c <= 'z'; c++)
                allowedNextChars[c] = true;
            for (unsigned char c = 'A'; c <= 'Z'; c++)
                allowedNextChars[c] = true;
            allowedNextChars['\''] = true; // words can start with apostrophe
            allowedNextChars['&'] = true; // etc
        }
        else if (lastToken == '?' || lastToken == '.' || lastToken == ':'
            || lastToken == '!' || lastToken == ',' || lastToken == ';')
        {
            allowedNextChars['\n'] = true;
            allowedNextChars[' '] = true;
            allowedNextChars['\''] = true;
            allowedNextChars['_'] = true; // dash
        }
        else if (lastToken == '&') // et cetera
        {
            allowedNextChars['c'] = true;
            allowedNextChars['C'] = true;
        }
        else if (prevToken == '&' && std::tolower(lastToken) == 'c') // et cetera
        {
            allowedNextChars[':'] = true;
            allowedNextChars['.'] = true;
        }
        else if (lastToken == '\'') // Start of quote or end of quote or inside word
        {
            // This is too complicated, we could replace ' by " around quotes in the training text
            bool startOfQuote = false;
            bool endOfQuote = false;
            bool inWord = false; // those booleans are not exclusive, meant as possibilities
            if (prevToken == '?' || prevToken == '!' || prevToken == '.'
                || prevToken == ',' || prevToken == ';' || prevToken == ':' || prevToken == '\0')
            {
                endOfQuote = true;
                inWord = true; // ,'tis
            }
            if (prevToken == ' ')
            {
                startOfQuote = true;
                inWord = true;
                endOfQuote = true;
            }
            if (prevToken == '\n' || prevToken == '_' || prevToken == '\0')
            {
                // For example "\n'fore" can either mean "afore"
                // or quote starting with "fore". 
                startOfQuote = true;
                inWord = true;
            }
            if (('a' <= prevToken && prevToken <= 'z') || ('A' <= prevToken && prevToken <= 'Z'))
            {
                endOfQuote = true;
                inWord = true;
            }
            if (startOfQuote)
            {
                for (unsigned char c = 'a'; c <= 'z'; c++)
                    allowedNextChars[c] = true;
                for (unsigned char c = 'A'; c <= 'Z'; c++)
                    allowedNextChars[c] = true;
            }
            if (inWord)
            {
                // There can be 2 ' in a word, like dow'r'd.
                AllowWord(prefixB, prefixE, allowedNextChars);
            }
            if (endOfQuote)
            {
                allowedNextChars[' '] = true;
                allowedNextChars['\n'] = true;
                allowedNextChars['?'] = true;
                allowedNextChars['!'] = true;
                allowedNextChars[';'] = true;
                allowedNextChars[':'] = true;
                allowedNextChars['_'] = true; // dash
            }
        }
        else if (('a' <= lastToken && lastToken <= 'z')
                || ('A' <= lastToken && lastToken <= 'Z')
                || lastToken == '-') // do not take ', it is handled just above
        {
            AllowWord(prefixB, prefixE, /*out*/allowedNextChars);
        }
        else
            return; // all characters allowed

        // This is an affine function on the logits, with a
        // square diagonal matrix of 0's and 1's.
        for (size_t i = 0; i < fAlphabet.size(); i++)
            if (!allowedNextChars[fAlphabet[i]])
                logits[i] = -1e16;
    }

    std::string fAlphabet;
    std::vector<std::string> fVocabulary;
    pf::Linear fCharacterEmbeddings;
    pf::Linear fPositionEmbeddings; // possibly empty
    // Compute logits for next character, from prefix.
    // The logits go through an English filter, then softmax to get probabilities.
    AttentionStack fComputeLogits;
    std::vector<CharGenerator> fChildren; // for multi-thread
    std::string fComment;

    // Output vectors, to avoid many malloc and free during the gradient descent
    mutable std::vector<double> oEmbeddings;
    mutable pf::Linear oDiffLoss;
    mutable pf::Linear oDiffOneSample;
};

// For the softmaxes to remain diffuse (and not usual maxes)
// we must keep parameters small. They are measured here.
double MaxAbsParam(const CharGenerator& charGen)
{
    double norm = 0.0;
    for (unsigned int i = 0; i < charGen.GetParameterCount(); i++)
    {
        const double p = charGen.GetParameter(i);
        if (std::abs(norm) < std::abs(p))
            norm = p;
    }
    return norm;
}

// Compute the negative log-likelihood of this sample,
// which is predicting the character at contextE, given
// the context [contextB, contextE[.
double LossOneSample(const CharGenerator& charGen,
    const char* contextB,
    const char* contextE)
{
    const std::vector<double>& probas = charGen.NextCharProbabilities(contextB, contextE);
    if (probas[GetCharacterIndex(charGen.GetAlphabet(), *contextE)] == 0.0)
    {
        if (contextB < contextE - 100)
            contextB = contextE - 100;
        std::string context(contextB, contextE);
        throw std::invalid_argument(std::string("Next character impossible ") + *contextE + " in context " + context);
    }
    const double loss = -std::log(probas[GetCharacterIndex(charGen.GetAlphabet(), *contextE)]);
    if (std::isnan(loss) || !std::isfinite(loss))
    {
        std::string context(contextB, contextE);
        throw std::invalid_argument("bad loss, context: " + context);
    }
    return loss;
}

// The loss of char sequence ]textB, textE] is -log of
// the probability that this model generates this sequence,
// under the singleton initial context, the character at textB.
// During training, Loss is evaluated on the full training text.
// After training, when a local minimum is found, Loss is also evaluated
// on the validation text, to compare different models.
// Karpathy randomizes this loss function : at each optimization step,
// randomly sample an index i in ]textB, textE], and compute -log of the
// probability of ]textB+i, textB+i+GetContextLength()], given the initial
// character at textB+i. This improves the beginning of generated texts,
// because it trains the model with more samples of truncated contexts.
// And it allows multiple layers of attention.
double Loss(const CharGenerator& charGen, const char* textB, const char* textE)
{
    ScopedClock clk("Loss clock");
    const unsigned int threadCount = charGen.GetChildren().size() + 1;
    const unsigned int remainder = (textE - textB) % threadCount;
    const unsigned int textThreadLen = ((textE - textB) / threadCount)
        + (remainder ? 1 : 0); // do not distribute the remainder, it is small

    // Those 3 captures could be taken as arguments instead
    auto iter = [textB, textE, textThreadLen](unsigned int threadIdx, const CharGenerator* gen) -> double
        {
            const char* contextE = textB + threadIdx * textThreadLen + 1;
            const char* contextEE = contextE + textThreadLen;
            if (textE < contextEE)
                contextEE = textE + 1;
            double loss = 0.0;
            while (contextE < contextEE)
            {
                // Postcondition: contextE is valid, it is the next character to predict.
                // The transformer model assumes the probability conditional to all past characters
                // is equal to the probability conditional to the last GetContextLength() characters.
                // It is a Markov property, with GetContextLength() instead of 1.
                loss += LossOneSample(*gen, textB, contextE);
                contextE++;
            }
            return loss;
        };

    std::vector<std::future<double>> futures;
    auto child = charGen.GetChildren().begin();
    for (unsigned int i = 1; i < threadCount; i++, child++)
    {
        // Could use a recursive function to create the futures and sum them,
        // without a vector of futures.
        futures.emplace_back(std::async(std::launch::async, iter, i, &*child));
    }

    double loss = iter(0, &charGen);
    for (auto& fut : futures)
        loss += fut.get();
    return loss;
}

// Add differential into oDiffLoss (does not reset it to 0)
void DifferentialLoss(const CharGenerator& charGen, 
    const char* textB,
    const char* subtextB,
    const char* subtextE)
{
    ScopedClock clk("DifferentialLoss clock");

    const unsigned int threadCount = charGen.GetChildren().size() + 1;
    const unsigned int remainder = (subtextE - subtextB + 1) % threadCount;
    const unsigned int textThreadLen = ((subtextE - subtextB + 1) / threadCount)
        + (remainder ? 1 : 0); // do not distribute the remainder, it is small

    // Those 4 captures could be taken as arguments instead
    auto iter = [textB, subtextB, subtextE, textThreadLen](unsigned int threadIdx, const CharGenerator* gen) -> void
        {
            const char* contextE = subtextB + threadIdx * textThreadLen;
            const char* contextEE = contextE + textThreadLen;
            if (subtextE < contextEE)
                contextEE = subtextE + 1;
            while (contextE < contextEE)
            {
                double* rowGsample = gen->GetDiffLossOneSample().GetRow(0);
                std::fill(rowGsample, rowGsample + gen->GetDiffLossOneSample().GetInputDim(), 0.0);
                gen->DifferentialLossOneSample(textB, contextE, /*out*/gen->GetDiffLossOneSample());
                pf::AddScaledVector(rowGsample, rowGsample + gen->GetParameterCount(), 1.0,
                    /*out*/gen->GetDiffLoss().GetRow(0));
                contextE++;
            }
        };

    std::vector<std::future<void>> futures;
    auto child = charGen.GetChildren().begin();
    for (unsigned int i = 1; i < threadCount; i++, child++)
    {
        // Could use a recursive function to create the futures and sum them,
        // without a vector of futures.
        std::fill(child->GetDiffLoss().GetRow(0), child->GetDiffLoss().GetRow(0) + child->GetParameterCount(), 0.0);
        futures.emplace_back(std::async(std::launch::async, iter, i, &*child));
    }
    iter(0, &charGen);
    futures.clear(); // wait for all threads
    for (const CharGenerator& child : charGen.GetChildren())
        pf::AddScaledVector(child.GetDiffLoss().GetRow(0), child.GetDiffLoss().GetRow(0) + child.GetParameterCount(), 1.0,
            /*out*/charGen.GetDiffLoss().GetRow(0));
}

char GenerateCharacter(const CharGenerator& charGen, const char* prefixB, const char* prefixE) 
{
    const std::vector<double>& probas = charGen.NextCharProbabilities(prefixB, prefixE);
    const std::string& alphabet = charGen.GetAlphabet();

    // Randomly draw next character
    std::uniform_real_distribution<> distrib(0, 1);
    double nextCharacterIndex = distrib(gen);
    unsigned int i = 0;
    for (const double& d : probas)
    {
        if (nextCharacterIndex < d)
            return alphabet[i];
        nextCharacterIndex -= d;
        i++;
    }
    throw std::invalid_argument("Cannot generate character");
}

std::string GenerateText(const CharGenerator& charGen)
{
    std::string s(2000, '\n');
    char* const prefixB = s.data();
    char* prefixE = prefixB + 1; // initialize prefix as "\n"
    while (*prefixE)
    {
        *prefixE = '\0'; // prettier in the debugger
        *prefixE = GenerateCharacter(charGen, prefixB, prefixE);
        prefixE++;
    }
    return s;
}

void TestGradientLoss(const CharGenerator& gen, const char* context)
{
    const unsigned int contextLen = gen.GetContextLength();
    const double loss = Loss(gen, context, context + contextLen);
    const double lossBis = Loss(gen, context, context + contextLen);
    if (loss != lossBis)
        throw std::invalid_argument("Loss is not idempotent");
    const unsigned int paramCount = gen.GetParameterCount();
    auto& oDiffLoss = gen.GetDiffLoss();
    std::fill(oDiffLoss.GetRow(0), oDiffLoss.GetRow(0) + paramCount, 0.0);
    DifferentialLoss(gen, context, context + 1, context + contextLen);
    std::vector<double> oDiffLossBis(paramCount, 0.0);
    for (unsigned int i = 0; i < paramCount; i++)
        oDiffLossBis[i] = oDiffLoss.GetRow(0)[i];
    std::fill(oDiffLoss.GetRow(0), oDiffLoss.GetRow(0) + paramCount, 0.0);
    DifferentialLoss(gen, context, context + 1, context + contextLen);
    for (unsigned int i = 0; i < paramCount; i++)
        if (oDiffLoss.GetRow(0)[i] != oDiffLossBis[i])
            throw std::invalid_argument("DiffLoss is not idempotent");
    const double* diffLoss = oDiffLoss.GetRow(0);
    const double bump = 1e-5;
    std::vector<double> finiteDiff(paramCount, 0.0);
    for (unsigned int i = 0; i < paramCount; i++)
    {
        // This test could fail faster by exiting at the first different coordinate.
        // However we execpt this test to pass most of the time, which computes
        // all coordinates. And also we can select the biggest error.
        const double param = gen.GetParameter(i);
        const_cast<CharGenerator&>(gen).GetParameter(i) = param + bump; // const_cast restored 2 lines below
        const_cast<CharGenerator&>(gen).SyncChildren();
        const double lossBump = Loss(gen, context, context + contextLen);
        const_cast<CharGenerator&>(gen).GetParameter(i) = param; // restore param
        const_cast<CharGenerator&>(gen).SyncChildren();
        finiteDiff[i] = (lossBump - loss) / bump;
    }
    double gError = 0.0;
    unsigned int biggestErrorIdx = 0;
    for (unsigned int i = 0; i < paramCount; i++)
    {
        if (gError < std::fabs(diffLoss[i] - finiteDiff[i]))
        {
            gError = std::fabs(diffLoss[i] - finiteDiff[i]);
            biggestErrorIdx = i;
        }
    }
    if (1e-3 < std::fabs(gError)) // that does not compare small gradients well
        throw std::invalid_argument("TestGradientLoss failed: finiteDiff " + std::to_string(finiteDiff[biggestErrorIdx])
            + " diff " + std::to_string(diffLoss[biggestErrorIdx]));
}

void TestCurrentGradient(const CharGenerator& charGen,
    const std::string& trainingText)
{
    const unsigned int paramCount = charGen.GetParameterCount();
    double* diffLoss = charGen.GetDiffLoss().GetRow(0);
    std::vector<double> gradCopy(diffLoss, diffLoss + paramCount);
    std::fill(diffLoss, diffLoss + paramCount, 0.0);
    DifferentialLoss(charGen, trainingText.c_str(), trainingText.c_str() + 1, &trainingText.back());
    double maxDiff = 0.0;
    unsigned int maxIdx;
    for (unsigned int i = 0; i < paramCount; i++)
        if (maxDiff < std::abs(gradCopy[i] - diffLoss[i]))
        {
            maxDiff = std::abs(gradCopy[i] - diffLoss[i]);
            maxIdx = i;
        }

    if (0.001 < maxDiff)
    {
        std::stringstream ss;
        ss << "bad batch gradient " << gradCopy[maxIdx] << " " << diffLoss[maxIdx];
        throw std::invalid_argument(ss.str());
    }
}

void Save(const CharGenerator& g, std::ofstream&& f)
{
    if (!f.good())
        return;
    f << "Comment: " << g.GetComment() << std::endl;
    f << "ContextLength: " << g.GetContextLength() << std::endl;
    f << "CharEmbedDim: " << g.GetTokenEmbeddingDim() << std::endl;
    f << "AttentionHeads: " << g.GetHeadCount() << std::endl;
    // This saves parameters with precision 1e-6.
    // It is enough because training finds a local
    // minimum (small gradient), so changing a
    // parameter by 1e-6 should have no effect.
    f << "Parameters: ";
    for (unsigned int i = 0; i < g.GetParameterCount(); i++)
        f << g.GetParameter(i) << " ";
    f.close();
}

CharGenerator Load(std::ifstream&& f,
    const std::string& alphabet) // should be stored in text file too
{
    unsigned int contextLength = 64;
    unsigned int tokenEmbeddingDim = 32;
    unsigned int attentionHeads = 1;
    unsigned int threadCount = 6;

    if (!f.good())
        return CharGenerator(contextLength, tokenEmbeddingDim, attentionHeads, threadCount, alphabet);

    std::string comment;
    std::string params;
    std::string l;
    while (std::getline(f, l))
    {
        std::istringstream ss(l);
        std::string arg;
        ss >> arg;
        if (arg == "Comment:")
        {
            std::getline(ss, comment);
            if (comment[0] == ' ')
                comment.erase(comment.begin());
        }
        else if (arg == "Parameters:")
            std::getline(ss, params);
        else if (arg == "ContextLength:")
            ss >> contextLength;
        else if (arg == "CharEmbedDim:")
            ss >> tokenEmbeddingDim;
        else if (arg == "AttentionHeads:")
            ss >> attentionHeads;
        else if (arg == "ThreadCount:")
            ss >> threadCount;
        else
            throw std::invalid_argument("Unknown argument: " + arg);
    }

    CharGenerator transformer(contextLength, tokenEmbeddingDim, attentionHeads, threadCount, alphabet);
    const unsigned int paramCount = transformer.GetParameterCount();
    std::istringstream paramStream(params);
    for (unsigned int i = 0; i < paramCount; i++)
    {
        double p;
        paramStream >> p;
        //if (1.8 < std::abs(p))
        //    p -= (int)p;
        transformer.GetParameter(i) = p;
    }
    transformer.SyncChildren();

    if (comment != "")
        transformer.SetComment(comment);

    f.close();
    return transformer;
}

void TrainCharGen(CharGenerator& charGen,
    const std::string& trainingText,
    const std::string& validationText)
{
    std::cout << "Begin training of transformer language model" << std::endl;

    std::vector<double> oSavedParameters(charGen.GetParameterCount(), 0.0);

    const unsigned int paramCount = charGen.GetParameterCount();
    const double initValLoss = Loss(charGen, validationText.c_str(), &validationText.back());
    double trainingLoss = Loss(charGen, trainingText.c_str(), &trainingText.back());
    std::cout << "Initial training loss: " << trainingLoss
        << " Initial validation loss: " << initValLoss << std::endl;
    // stepSize is a length in parameter space, because we multiply it
    // to the normalized gradient. Since parameters should be approximately
    // in [-1,1], 0.1 is already a significant stepSize. The linearly
    // approximated loss's decrease is stepSize * gradient norm.
    double stepSize = 0.01;
    const unsigned int gradientSteps = 50000;
    const unsigned int batchSize = trainingText.size() - 1;
    double leastTrainingLoss = trainingLoss;
    std::uniform_int_distribution<> distrib(0,
        static_cast<int>(trainingText.size()) - batchSize - 1); // end included

    // Stochastic gradient : approximate true gradient by the gradient
    // on a subtext. This is consistent with the hope that this character
    // model generalizes to unseen examples, like the validation text.
    const char* gradB = nullptr;
    const char* gradE = nullptr;
    for (unsigned int i = 0; i < gradientSteps; i++)
    {
        ScopedClock clk("Gradient descent step clock");

        for (unsigned int p = 0; p < paramCount; p++)
            oSavedParameters[p] = charGen.GetParameter(p);

        double* diffLoss = charGen.GetDiffLoss().GetRow(0);
        if (gradB == nullptr)
        {
            unsigned int start = distrib(gen);
            MoveToStartOfWord(trainingText, /*out*/start);
            gradB = trainingText.c_str() + start + 1;
            gradE = gradB - 1; // so that gradM == gradB below
            std::fill(diffLoss, diffLoss + paramCount, 0.0);
        }
        if (gradE < &trainingText.back())
        {
            const char* gradM = gradE + 1;
            gradE += batchSize;
            if (&trainingText.back() < gradE)
                gradE = &trainingText.back();
            DifferentialLoss(charGen, trainingText.c_str(), gradM, gradE);
        }
        else if (trainingText.c_str() + 1 < gradB)
        {
            // Postcondition: gradE == &trainingText.back().
            const char* gradM = gradB - 1;
            gradB -= batchSize;
            if (gradB < trainingText.c_str() + 1)
                gradB = trainingText.c_str() + 1;
            DifferentialLoss(charGen, trainingText.c_str(), gradB, gradM);
        }
        const double norm = std::sqrt(std::inner_product(diffLoss, diffLoss + paramCount, diffLoss, 0.0));
        if (std::isnan(norm))
            throw std::invalid_argument("Bad gradient norm"); // adding more context here does not allow fast debugging
        if (norm <= 1e-10)
            throw std::invalid_argument("Zero gradient norm"); // adding more context here does not allow fast debugging
        for (unsigned int p = 0; p < paramCount; p++)
            charGen.GetParameter(p) -= stepSize * diffLoss[p] / norm; // negative for gradient descent
        charGen.SyncChildren();

        // Loss is easier to compute than DifferentialLoss, but it loops on all training points,
        // so its computation time is similar.
        trainingLoss = Loss(charGen, trainingText.c_str(), &trainingText.back());
        std::stringstream iterMsg;
        iterMsg << "MaxAbsParam: " << MaxAbsParam(charGen)
            << " TrainingLoss: " << trainingLoss
            << " StepSize: " << stepSize;
        // When expected is different from actual, the step is not linear,
        // the higher derivatives play too much, reduce step size.
        if (!std::isnan(trainingLoss) && trainingLoss < leastTrainingLoss)
        {
            iterMsg << " ValidationLoss: " << Loss(charGen, validationText.c_str(), &validationText.back())
                << " Progress: " << leastTrainingLoss - trainingLoss;
            Save(charGen, std::ofstream(sDatasetDirectory / "attention.tmp"));
            std::remove(std::filesystem::path(sDatasetDirectory / "attention.txt").string().c_str());
            std::rename(std::filesystem::path(sDatasetDirectory / "attention.tmp").string().c_str(),
                std::filesystem::path(sDatasetDirectory / "attention.txt").string().c_str()); // check how atomic this is
            leastTrainingLoss = trainingLoss;
            stepSize *= 1.1;
            gradB = gradE = nullptr;
        }
        else
        {
            // Restore previous parameters
            for (unsigned int p = 0; p < paramCount; p++)
                charGen.GetParameter(p) = oSavedParameters[p];
            if (trainingText.c_str() + 1 == gradB && gradE == &trainingText.back())
            {
                // TestCurrentGradient(charGen, trainingText);
                // In the direction of its gradient, the loss is a convex parabola
                // (second-order approximation). We just found the other point where
                // the loss has its current level leastTrainingLoss, so its minimum
                // should be approximately halfway. 0.3 should be a little before
                // that maximum, which should give a few iterations with big progress.
                stepSize *= 0.3;
            }
            else // Incomplete gradient, which may have cause the failure
                stepSize *= 0.9;
            iterMsg << " KO";
        }
        std::cout << iterMsg.str() << std::endl;
    }
}

// TODO factorize with previous TrainCharGen
void TrainAdam(CharGenerator& charGen,
    const std::string& trainingText,
    const std::string& validationText)
{
    const unsigned int paramCount = charGen.GetParameterCount();
    const double initValLoss = Loss(charGen, validationText.c_str(), &validationText.back());
    double trainingLoss = Loss(charGen, trainingText.c_str(), &trainingText.back());
    std::cout << "Initial training loss: " << trainingLoss
        << " Initial validation loss: " << initValLoss << std::endl;
    const double stepSize = 0.001;
    pf::AdamOptimizer adam(paramCount, stepSize,
        [&charGen](unsigned int p) -> double& { return charGen.GetParameter(p); },
        [&charGen, &trainingText]() { DifferentialLoss(charGen, trainingText.c_str(), trainingText.c_str() + 1, &trainingText.back());
                        return charGen.GetDiffLoss().GetRow(0); });

    const unsigned int gradientSteps = 50000;
    double prevTrainingLoss = trainingLoss;
    for (unsigned int i = 0; i < gradientSteps; i++)
    {
        ScopedClock clk("Gradient descent step clock");

        adam.Step();
        charGen.SyncChildren();

        trainingLoss = Loss(charGen, trainingText.c_str(), &trainingText.back());
        std::stringstream iterMsg;
        iterMsg << "MaxAbsParam: " << MaxAbsParam(charGen)
            << " TrainingLoss: " << trainingLoss;
        // When expected is different from actual, the step is not linear,
        // the higher derivatives play too much, reduce step size.
        if (!std::isnan(trainingLoss))
        {
            iterMsg << " ValidationLoss: " << Loss(charGen, validationText.c_str(), &validationText.back())
                << " Progress: " << prevTrainingLoss - trainingLoss;
            Save(charGen, std::ofstream(sDatasetDirectory / "attention.tmp"));
            std::remove(std::filesystem::path(sDatasetDirectory / "attention.txt").string().c_str());
            std::rename(std::filesystem::path(sDatasetDirectory / "attention.tmp").string().c_str(),
                std::filesystem::path(sDatasetDirectory / "attention.txt").string().c_str()); // check how atomic this is
            prevTrainingLoss = trainingLoss;
        }
        std::cout << iterMsg.str() << std::endl;
    }
}

CharGenerator IncreaseEmbedding(const CharGenerator& source, unsigned int tokDim)
{
    const unsigned int contextLength = 64;
    const unsigned int attentionHeads = 2;
    const unsigned int threadCount = 1;
    const std::string alphabet = source.GetAlphabet();
    const unsigned int sourceDim = source.GetTokenEmbeddingDim();
    CharGenerator target(contextLength, tokDim, attentionHeads, threadCount, alphabet);

    // Enlarge character and position embeddings with zeros
    unsigned int col;
    pf::Linear& charEmdb = const_cast<pf::Linear&>(target.GetCharacterEmbeddings());
    pf::Linear& posEmdb = const_cast<pf::Linear&>(target.GetPositionEmbeddings());
    charEmdb.Fill(0.0, 0.0);
    charEmdb.Copy(source.GetCharacterEmbeddings());
    posEmdb.Fill(0.0, 0.0);
    posEmdb.Copy(source.GetPositionEmbeddings());

    // Enlarge first query matrix.
    // Affinities are <| c_i, Q * c_last |> / fScale
    // Characters are truncated so Q's last columns and last rows remain random.
    const AttentionStack& sourceAttention = source.GetAttentionStack();
    const AttentionStack& targetAttention = target.GetAttentionStack();
    const AttentionHead& sourceHead = dynamic_cast<const AttentionHead&>(*sourceAttention.GetHeads().GetFunctions()[0]);
    AttentionHead& firstHead = const_cast<AttentionHead&>(dynamic_cast<const AttentionHead&>(*targetAttention.GetHeads().GetFunctions()[0]));
    pf::Linear& firstQuery = const_cast<pf::Linear&>(firstHead.GetQueryMatrix());
    const double sourceScale = std::sqrt(static_cast<double>(sourceDim));
    const double targetScale = std::sqrt(static_cast<double>(tokDim));
    for (unsigned int row = 0; row < sourceDim; row++)
    {
        for (col = 0; col < sourceDim; col++)
            firstQuery.GetRow(row)[col]
                = sourceHead.GetQueryMatrix().GetRow(row)[col] * targetScale / sourceScale;
    }

    // First Value matrix
    // Average character is truncated so V's last columns remain random.
    const_cast<pf::Linear&>(firstHead.GetValueMatrix()).Copy(sourceHead.GetValueMatrix());

    // Set second value matrix to zero
    AttentionHead& secondHead = const_cast<AttentionHead&>(dynamic_cast<const AttentionHead&>(*targetAttention.GetHeads().GetFunctions()[1]));
    for (unsigned int row = 0; row < sourceDim; row++)
    {
        for (col = 0; col < sourceDim; col++)
            const_cast<pf::Linear&>(secondHead.GetValueMatrix()).GetRow(row)[col] = 0.0;
    }

    // TLP's first matrix
    // Average character is truncated so V's last columns remain random.
    // Set lower-left part to 0.
    const TLP& sourceTLP = sourceAttention.GetTLP();
    TLP& targetTLP = const_cast<TLP&>(targetAttention.GetTLP());
    for (unsigned int row = 0; row < 4 * sourceDim; row++)
    {
        for (col = 0; col < sourceDim; col++)
            targetTLP.SetU(row * tokDim + col,
                sourceTLP.ParamFunc::GetParameter(row * sourceDim + col));
    }
    for (unsigned int row = 4 * sourceDim; row < 4 * tokDim; row++)
    {
        for (col = 0; col < sourceDim; col++)
            targetTLP.SetU(row * tokDim + col, 0.0);
    }
    for (unsigned int row = 0; row < 4 * tokDim; row++)
        targetTLP.SetU(4 * tokDim * tokDim + row,
            row < 4 * sourceDim ? sourceTLP.ParamFunc::GetParameter(4 * sourceDim * sourceDim + row) : 0);

    // TLP's second matrix
    for (unsigned int row = 0; row < alphabet.size(); row++)
    {
        for (col = 0; col < 4 * sourceDim; col++)
            targetTLP.SetV(row * 4 * tokDim + col,
                sourceTLP.GetV().ParamFunc::GetParameter(row * 4 * sourceDim + col));
        targetTLP.SetV(alphabet.size() * 4 * tokDim + row,
            row < 4 * sourceDim ? sourceTLP.GetV().ParamFunc::GetParameter(alphabet.size() * 4 * sourceDim + row) : 0);
    }

    return target;
}

void PrettyPrint(const std::string& text)
{
    for (const char c : text)
    {
        if (c == '_')
            std::cout << "—";
        else
            std::cout << c;
    }
    std::cout << std::endl << "———————————————————————————————————————————————————————————————————————————————————————————" 
        << std::endl << std::endl;
}

// This shows how transformers can degenerate into any bigram model.
// However these are not always local minima, because there is a
// best bigram model for the training text. We do have local minima
// when we use transition probabilities that are so peaky that they
// do not account for all cases in the training text: the loss is
// infinite, i.e. the generation probability is 0 (local maximum).
CharGenerator DegenerateTransformer(const std::string& alphabet)
{
    const unsigned int contextLength = alphabet.size(); // could be any length
    const unsigned int attentionHeads = 1;
    const unsigned int threadCount = 1;
    const unsigned int tokDim = alphabet.size(); // avoids final projection matrix
    CharGenerator target(contextLength, tokDim, attentionHeads, threadCount, alphabet);
    pf::Linear& posEmdb = const_cast<pf::Linear&>(target.GetPositionEmbeddings());
    posEmdb.Fill(0.0, 0.0); // disable positions
    pf::Linear& charEmdb = const_cast<pf::Linear&>(target.GetCharacterEmbeddings());
    // set char embeddings to identity matrix
    charEmdb.Fill(0.0, 0.0);
    charEmdb.SetDiagonal(std::vector<double>(tokDim, 1));
    AttentionStack& stack = const_cast<AttentionStack&>(target.GetAttentionStack());
    AttentionHead& head = const_cast<AttentionHead&>(dynamic_cast<const AttentionHead&>(*stack.GetHeads().GetFunctions()[0]));
    pf::Linear& queryMatrix = const_cast<pf::Linear&>(head.GetQueryMatrix());
    // set query matrix to 1000 * identity matrix, to get average token = last token
    queryMatrix.Fill(0.0, 0.0);
    queryMatrix.SetDiagonal(std::vector<double>(tokDim, 1000));
    pf::Linear& valueMatrix = const_cast<pf::Linear&>(head.GetValueMatrix());
    // set value matrix to identity matrix
    valueMatrix.Fill(0.0, 0.0);
    valueMatrix.SetDiagonal(std::vector<double>(tokDim, 1));
    TLP& tlp = const_cast<TLP&>(stack.GetTLP());
    for (unsigned int row = 0; row < 4 * tokDim; row++)
    for (unsigned int col = 0; col < tokDim; col++)
        tlp.SetU(row * tokDim + col, row == col ? 1.0 : 0.0);
    // Now the k-th column of tlp's matrix V gives the probabilities
    // of character after character k (bigram model).
    const_cast<pf::Linear&>(tlp.GetV()).Fill(0.0, 0.0);
    for (unsigned int row = 0; row < tokDim; row++)
        for (unsigned int col = 0; col < tokDim; col++)
            tlp.SetV(row * tokDim * 4 + col, 1.0 / tokDim);
    return target;
}

std::string GenerateSimilarText(const std::string& text)
{
    // split 90% dataset, 10% validation set
    size_t trunc = (text.size() * 9) / 10; // 90% training, 10% validation
    std::set<char> excludedChars = { '?', '\n' }; // a text cannot start with those
    while (excludedChars.contains(text[trunc]))
        trunc++;
    const std::string training = text.substr(0, trunc);
    const std::string valid = text.substr(trunc, text.size() - trunc);

    //NgramModel nm;
    //nm.Train(training, 5);
    //const double ngramLoss = nm.Loss(valid); // 211415

    CharGenerator transformer = Load(std::ifstream(sDatasetDirectory / "attention.txt"), text);
    // CharGenerator transformer = DegenerateTransformer(Alphabet(text));
    // Save(IncreaseEmbedding(transformer, 32), std::ofstream(sDatasetDirectory / "attention2.txt"));
    try {
        //auto it = training.find("The offences we have made you do we'll answer,");
        // TestGradientLoss(transformer, training.c_str());
        PrettyPrint(GenerateText(transformer));
        TrainCharGen(/*out*/transformer, training, valid); }
    catch (const std::exception& e)
    { std::cout << e.what() << std::endl; return ""; }
    std::string gen = GenerateText(transformer);
    return gen;
}
