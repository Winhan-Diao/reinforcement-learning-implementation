#pragma once
#include <memory>
#include <stdexcept>
#include <valarray>
#ifndef M_PI
    #define M_PI 3.1415926535897932384626433832795
#endif

enum class LossFunctions {
    MSE,
    CROSS_ENTROPY_LOSS,
    CROSS_ENTROPY_LOSS_A,
    TAN,
};

struct LossFunction {
    virtual std::valarray<double> operator() (const std::valarray<double>& actual, const std::valarray<double>& predicted) = 0;
    // static std::unique_ptr<LossFunction> buildLossFunction(const LossFunctions&);
};

struct MSE: LossFunction {
    std::valarray<double> operator() (const std::valarray<double>& actual, const std::valarray<double>& predicted) override {
        return predicted - actual;
    }
};

struct CrossEntropyLoss: LossFunction {
    std::valarray<double> operator() (const std::valarray<double>& actual, const std::valarray<double>& predicted) override {
        std::valarray<double> loss(actual.size());
        for (ssize_t i = 0; i < actual.size(); ++i) {
            loss[i] = (actual[i] == 0)? (-1. / (predicted[i] - 1.)): (-1. / predicted[i]);  
        }
        return loss;
    }
};

struct CrossEntropyLossAmend: LossFunction {
    std::valarray<double> operator() (const std::valarray<double>& actual, const std::valarray<double>& predicted) override {
        std::valarray<double> loss(actual.size());
        for (ssize_t i = 0; i < actual.size(); ++i) {
            loss[i] = (actual[i] == 0)? (-1. / (predicted[i] - 1.) - 1): (-1. / predicted[i] + 1);  
        }
        return loss;
    }
};

struct Tan: LossFunction {
    static constexpr double tau = M_PI / 2; 
    std::valarray<double> operator() (const std::valarray<double>& actual, const std::valarray<double>& predicted) override {
        std::valarray<double> loss(actual.size());
        for (ssize_t i = 0; i < actual.size(); ++i) {
            loss[i] = (std::pow(std::tan(((actual[i] == 0)? predicted[i]: 1. - predicted[i]) * tau), 2) + 1) * tau * ((actual[i] == 0)? 1: -1);
        }
        return loss;
    }
};

static std::unique_ptr<LossFunction> buildLossFunction(const LossFunctions& n) {
    switch (n) {
        case LossFunctions::CROSS_ENTROPY_LOSS:
            return std::make_unique<CrossEntropyLoss>();
        case LossFunctions::CROSS_ENTROPY_LOSS_A:
            return std::make_unique<CrossEntropyLossAmend>();
        case LossFunctions::MSE:
            return std::make_unique<MSE>();
        case LossFunctions::TAN:
            return std::make_unique<Tan>();
        default:
            throw std::runtime_error{"cannot build LossFunction"};
    }
}
