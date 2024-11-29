#pragma once
#include <type_traits>
#include <vector>
#include <random>
#include <stdexcept>

template <template <class...> class T, class... Args>
size_t chooseAction(const T<double, Args...>& policy, std::mt19937& gen) {
    double acc = 0;
    double d = std::uniform_real_distribution(0., 1.)(gen);
    for (size_t i = 0; i < policy.size(); ++i) {
        acc += policy[i];
        if (d <= acc) {
            return i;
        }
    }
    throw std::runtime_error{"failed to choose a policy"};
}

template <template <class...> class T, class... Args>
size_t chooseAction(const T<double, Args...>& policy, std::mt19937& gen, double normalizeFactor) {
    double acc = 0;
    double d = std::uniform_real_distribution(0., 1.)(gen);
    for (size_t i = 0; i < policy.size(); ++i) {
        acc += (1 - normalizeFactor) * (policy[i]) + normalizeFactor * (1. / policy.size());
        if (d <= acc) {
            return i;
        }
    }
    throw std::runtime_error{"failed to choose a policy"};
}