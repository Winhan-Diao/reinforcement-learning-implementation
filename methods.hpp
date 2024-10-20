#pragma once
#include <vector>
#include <algorithm>
#include <cassert>
#include "playground.hpp"
#include "random_utils.hpp"

inline void foo() {}

struct Step {
    AbstractBlock *state;
    ActionType action;
    double reward;
};

template <class _MultiplyFunctor>
class EGreedyMonteCarlo {
    std::vector<std::vector<double>> qValues;
    std::vector<std::vector<double>> policies;
    std::vector<std::vector<size_t>> ns;
    const Playground& playground;
    size_t episodeLength;
    double discount;
    double epsilon;
    _MultiplyFunctor alphaFunctor;

public:
    template <class T>
    static auto toEGreedy(T&& policy, double eposilon) -> decltype(std::forward<T>(policy)) {
        auto size = policy.size();
        auto iter = std::max_element(policy.begin(), policy.end());
        std::for_each(policy.begin(), policy.end(), [&iter, size, eposilon](auto& v){
            if (&*iter == &v) {
                v = 1 - double(size - 1) / size * eposilon;
            } else {
                v = 1. / size * eposilon;
            }
        });
        return std::forward<T>(policy);
    }
    static std::vector<double> toEGreedy(size_t policyCount, size_t maxargIndex, double eposilon) {
        double star = 1 - double(policyCount - 1) / policyCount * eposilon;
        double other = 1. / policyCount * eposilon;
        std::vector<double> tmp(policyCount, other);
        tmp[maxargIndex] = star;
        return tmp;
    }
    static void toEGreedy(std::vector<double>& policy, size_t policyCount, size_t maxargIndex, double eposilon) {
        assert(policy.size() == policyCount);
        double star = 1 - double(policyCount - 1) / policyCount * eposilon;
        double other = 1. / policyCount * eposilon;
        for (size_t i = 0; i < policyCount; ++i) {
            policy[i] = i == maxargIndex? star: other;
        }
    }
    EGreedyMonteCarlo(const Playground& playground, size_t episodeLength, double discount, double epsilon, _MultiplyFunctor&& alphaFunctor)
        : qValues(playground.height * playground.width, std::vector<double>(size_t(5), double(0)))
        , policies(playground.height * playground.width, this->toEGreedy(std::vector<double>{0, 0, 0, 0, 1}, epsilon))
        , ns(playground.height * playground.width, std::vector<size_t>(5, 0))
        , playground(playground) 
        , episodeLength(episodeLength)
        , discount(discount)
        , epsilon(epsilon)
        , alphaFunctor(std::forward<_MultiplyFunctor>(alphaFunctor))
    {}
    void run() { run(this->epsilon); }
    void run(double epsilon) {
        std::vector<Step> episode = std::vector<Step>();
        double g = 0;

        AbstractBlock *currentState = playground.sample();
        // AbstractBlock *currentState = playground.map[playground.width - 1].get();
        for (size_t i = 0; i < episodeLength; ++i) {
            size_t p = choosePolicy(policies[currentState->getFlatten()], playground.gen);
            auto [nextState, reward] = playground.act(currentState, static_cast<ActionType>(p));
            episode.push_back(Step{currentState, static_cast<ActionType>(p), reward});
            currentState = nextState;
        }
        for (ssize_t i = episodeLength - 1; i >= 0; --i) {
            g = discount * g + episode[i].reward;
            qValues[episode[i].state->getFlatten()][size_t(episode[i].action)] += alphaFunctor(++ns[episode[i].state->getFlatten()][size_t(episode[i].action)]) * (g - qValues[episode[i].state->getFlatten()][size_t(episode[i].action)]);
        }
        for (size_t i = 0; i < qValues.size(); ++i) {
            toEGreedy(policies[i], 5, std::distance(qValues[i].cbegin(), std::max_element(qValues[i].cbegin(), qValues[i].cend())), epsilon);
        }
        return;
    }
    friend int main();
};

inline void showPolicy(const std::vector<std::vector<double>>& policy, const Playground& playground, std::ostream& os = std::cout) {
    for (uint16_t h = 0; h < playground.height; ++h) {
        for (uint16_t w = 0; w < playground.width; ++w) {
            const auto& aPolicy = policy[h * playground.width + w];
            auto argmexPolicy = std::distance(aPolicy.cbegin(), std::max_element(aPolicy.cbegin(), aPolicy.cend()));
            switch (argmexPolicy) {
                case 0:
                    os << "∧";
                    break;
                case 1:
                    os << ">";
                    break;
                case 2:
                    os << "∨";
                    break;
                case 3:
                    os << "<";
                    break;
                case 4:
                    os << "o";
                    break;
                default:
                    throw std::runtime_error{"bad policy"};
            }
            std::cout << " ";
        }
        std::cout << "\r\n";
    }
}