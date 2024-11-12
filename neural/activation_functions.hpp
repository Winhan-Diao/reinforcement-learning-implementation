#pragma once
#include <memory>
#include <stdexcept>
#include <valarray>
#include <functional>

enum class ActivationFunctions {
    INVALID,
    SIGMOID,
    TANH,
    RELU,
    LEAKYRELU,
    PRRELU,
    SOFTMAX,
    CUBEROOT,
    SGNEXP,
};

struct ActivationFunction {
    virtual std::valarray<double> operator() (const std::valarray<double>& x) = 0;
    virtual std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) = 0;
    // static std::unique_ptr<ActivationFunction> buildActivationFunction(const ActivationFunctions& n);
};

struct Sigmoid: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        return 1 / (1 + std::exp(-x));
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        return y * (1 - y) * usGrad;
    }
};

struct Tanh: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        return std::tanh(x);
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        return (1 - y * y) * usGrad;
    }
};

struct Relu: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        std::valarray<double> r(x.size());
        std::transform(std::cbegin(x), std::cend(x), std::begin(r), [](const double& e) {
            return std::max(0.0, e);
        });
        return r;
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        std::valarray<double> r(y.size());
        std::transform(std::cbegin(y), std::cend(y), std::begin(r), [](const double& e) {
            return e > 0 ? 1.0 : 0.0;
        });
        return r * usGrad;
    }
};

struct LeakyRelu: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        std::valarray<double> r(x.size());
        std::transform(std::cbegin(x), std::cend(x), std::begin(r), [](const double& e) {
            return (e > 0)? e: .02 * e;
        });
        return r;
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        std::valarray<double> r(y.size());
        std::transform(std::cbegin(y), std::cend(y), std::begin(r), [](const double& e) {
            return e > 0 ? 1.0 : 0.02;
        });
        return r * usGrad;
    }
};

struct PrRelu: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        std::valarray<double> r(x.size());
        std::transform(std::cbegin(x), std::cend(x), std::begin(r), [](const double& e) {
            return (e > 0)? e: .2 * e;
        });
        return r;
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        std::valarray<double> r(y.size());
        std::transform(std::cbegin(y), std::cend(y), std::begin(r), [](const double& e) {
            return (e > 0)? 1. : .2;
        });
        return r * usGrad;
    }
};

struct Softmax: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        auto expX = std::exp(x);
        double expSum = expX.sum();
        return std::move(expX) / expSum;
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        double sum = (y * usGrad).sum();
        return -y * (sum - usGrad);
    }
};

struct CubeRoot: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        return std::pow(std::abs(x), 1. / 3) * x.apply([](double v) -> double { return v < 0? -1: 1; });
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        return usGrad / (y * y * 3 + 1e-5);
    }
};

struct SgnExp: ActivationFunction {
    std::valarray<double> operator() (const std::valarray<double>& x) override {
        return (1 - std::exp(-std::abs(x))) * x.apply([](double v) -> double { return v < 0? -1: 1; });
    }
    std::valarray<double> derivative(const std::valarray<double>& y, const std::valarray<double>& usGrad) override {
        return (1 - std::abs(y)) * usGrad;
    }
};

static std::unique_ptr<ActivationFunction> buildActivationFunction(const ActivationFunctions& n) {
    switch (n) {
        case ActivationFunctions::SIGMOID:
            return std::make_unique<Sigmoid>();
        case ActivationFunctions::TANH:
            return std::make_unique<Tanh>();
        case ActivationFunctions::RELU:
            return std::make_unique<Relu>();
        case ActivationFunctions::LEAKYRELU:
            return std::make_unique<LeakyRelu>();
        case ActivationFunctions::PRRELU:
            return std::make_unique<PrRelu>();
        case ActivationFunctions::SOFTMAX:
            return std::make_unique<Softmax>();
        case ActivationFunctions::CUBEROOT:
            return std::make_unique<CubeRoot>();
        case ActivationFunctions::SGNEXP:
            return std::make_unique<SgnExp>();
        case ActivationFunctions::INVALID:
        default:
            throw std::runtime_error{"cannot build ActivationFunction"};
    }
}
