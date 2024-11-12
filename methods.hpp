#pragma once
#include <vector>
#include <algorithm>
#include <cassert>
#include "playground.hpp"
#include "random_utils.hpp"

inline void foo() {}

template <class T>
inline auto toEGreedy(T&& policy, double eposilon) -> decltype(std::forward<T>(policy)) {
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
inline std::vector<double> toEGreedy(size_t policyCount, size_t maxargIndex, double eposilon) {
    double star = 1 - double(policyCount - 1) / policyCount * eposilon;
    double other = 1. / policyCount * eposilon;
    std::vector<double> tmp(policyCount, other);
    tmp[maxargIndex] = star;
    return tmp;
}
inline void toEGreedy(std::vector<double>& policy, size_t policyCount, size_t maxargIndex, double eposilon) {
    assert(policy.size() == policyCount);
    double star = 1 - double(policyCount - 1) / policyCount * eposilon;
    double other = 1. / policyCount * eposilon;
    for (size_t i = 0; i < policyCount; ++i) {
        policy[i] = i == maxargIndex? star: other;
    }
}

struct Step {
    AbstractBlock *state;
    ActionType action;
    double reward;
};

struct Experience {
    AbstractBlock *state;
    ActionType action;
    double reward;
    AbstractBlock *nextState;
    ActionType nextAction;
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
    EGreedyMonteCarlo(const Playground& playground, size_t episodeLength, double discount, double epsilon, _MultiplyFunctor&& alphaFunctor)
        : qValues(playground.height * playground.width, std::vector<double>(size_t(5), double(0)))
        , policies(playground.height * playground.width, toEGreedy(std::vector<double>{0, 0, 0, 0, 1}, epsilon))
        , ns(playground.height * playground.width, std::vector<size_t>(5, 0))
        , playground(playground) 
        , episodeLength(episodeLength)
        , discount(discount)
        , epsilon(epsilon)
        , alphaFunctor(std::forward<_MultiplyFunctor>(alphaFunctor))
    {}
    std::vector<std::vector<double>>& getPolicies() { return const_cast<std::vector<std::vector<double>>&>(const_cast<const EGreedyMonteCarlo<_MultiplyFunctor>&>(*this).getPolicies()); }
    const std::vector<std::vector<double>>& getPolicies() const { return policies; }
    void run() { run(this->epsilon); }
    void run(double epsilon) {
        std::vector<Step> episode = std::vector<Step>();
        double g = 0;

        AbstractBlock *currentState = playground.sample();
        // AbstractBlock *currentState = playground.map[playground.width - 1].get();
        for (size_t i = 0; i < episodeLength; ++i) {
            size_t p = chooseAction(policies[currentState->getFlatten()], playground.gen);
            auto [nextState, reward] = playground.act(currentState, static_cast<ActionType>(p));
            episode.push_back(Step{currentState, static_cast<ActionType>(p), reward});
            currentState = nextState;
        }
        for (ssize_t i = episodeLength - 1; i >= 0; --i) {
            g = discount * g + episode[i].reward;
            qValues[episode[i].state->getFlatten()][size_t(episode[i].action)] += alphaFunctor(ns[episode[i].state->getFlatten()][size_t(episode[i].action)]) * (g - qValues[episode[i].state->getFlatten()][size_t(episode[i].action)]);
        }
        for (size_t i = 0; i < qValues.size(); ++i) {
            toEGreedy(policies[i], 5, std::distance(qValues[i].cbegin(), std::max_element(qValues[i].cbegin(), qValues[i].cend())), epsilon);
        }
        return;
    }
    friend void mcDemo(const std::string&, double, double);
};

template <class _MultiplyFunctor>
class Sarsa {
    std::vector<std::vector<double>> qValues;
    std::vector<std::vector<double>> policies;
    std::vector<std::vector<size_t>> ns;
    const Playground& playground;
    size_t episodeLength;
    double discount;
    double epsilon;
    _MultiplyFunctor alphaFunctor;
public:
    Sarsa(const Playground& playground, size_t episodeLength,  double discount, double epsilon, _MultiplyFunctor&& alphaFunctor)
        : qValues(playground.height * playground.width, std::vector<double>(size_t(5), 0.))
        , policies(playground.height * playground.width, toEGreedy(std::vector<double>{0, 0, 0, 0, 1}, epsilon))
        , ns(playground.height * playground.width, std::vector<size_t>(5, 0))
        , playground(playground) 
        , episodeLength(episodeLength)
        , discount(discount)
        , epsilon(epsilon)
        , alphaFunctor(std::forward<_MultiplyFunctor>(alphaFunctor))
    {}
    std::vector<std::vector<double>>& getPolicies() { return const_cast<std::vector<std::vector<double>>&>(const_cast<const Sarsa<_MultiplyFunctor>&>(*this).getPolicies()); }
    const std::vector<std::vector<double>>& getPolicies() const { return policies; }
    void run() { run(this->epsilon); }
    void run(double epsilon) {
        AbstractBlock *currentState = playground.sample();
        // AbstractBlock *currentState = playground.map[0].get();
        size_t p = chooseAction(policies[currentState->getFlatten()], playground.gen);
        for (size_t i = 0; i < episodeLength; ++i) {
            auto [nextState, reward] = playground.act(currentState, static_cast<ActionType>(p));
            size_t nextP = chooseAction(policies[nextState->getFlatten()], playground.gen);
            qValues[currentState->getFlatten()][p] += alphaFunctor(ns[currentState->getFlatten()][p]) * (reward + discount * qValues[nextState->getFlatten()][nextP] - qValues[currentState->getFlatten()][p]);
            toEGreedy(policies[currentState->getFlatten()], 5, std::distance(qValues[currentState->getFlatten()].cbegin(), std::max_element(qValues[currentState->getFlatten()].cbegin(), qValues[currentState->getFlatten()].cend())), epsilon);
            currentState = nextState;
            p = nextP;
        }
    }
};

template <class _MultiplyFunctor>
class OnlineQLearning {
    std::vector<std::vector<double>> qValues;
    std::vector<std::vector<double>> policies;
    std::vector<std::vector<size_t>> ns;
    const Playground& playground;
    size_t episodeLength;
    double discount;
    double epsilon;
    _MultiplyFunctor alphaFunctor;
public:
    OnlineQLearning(const Playground& playground, size_t episodeLength,  double discount, double epsilon, _MultiplyFunctor&& alphaFunctor)
        : qValues(playground.height * playground.width, std::vector<double>(size_t(5), 0.))
        , policies(playground.height * playground.width, toEGreedy(std::vector<double>{0, 0, 0, 0, 1}, epsilon))
        , ns(playground.height * playground.width, std::vector<size_t>(5, 0))
        , playground(playground) 
        , episodeLength(episodeLength)
        , discount(discount)
        , epsilon(epsilon)
        , alphaFunctor(std::forward<_MultiplyFunctor>(alphaFunctor))
    {}
    std::vector<std::vector<double>>& getPolicies() { return const_cast<std::vector<std::vector<double>>&>(const_cast<const OnlineQLearning<_MultiplyFunctor>&>(*this).getPolicies()); }
    const std::vector<std::vector<double>>& getPolicies() const { return policies; }
    void run() { run(this->epsilon); }
    void run(double epsilon) {
        AbstractBlock *currentState = playground.sample();
        // AbstractBlock *currentState = playground.map[0].get();
        size_t p = chooseAction(policies[currentState->getFlatten()], playground.gen);
        for (size_t i = 0; i < episodeLength; ++i) {
            auto [nextState, reward] = playground.act(currentState, static_cast<ActionType>(p));
            size_t nextP = chooseAction(policies[nextState->getFlatten()], playground.gen);
            qValues[currentState->getFlatten()][p] += alphaFunctor(ns[currentState->getFlatten()][p]) * (reward + discount * *std::max_element(qValues[nextState->getFlatten()].cbegin(), qValues[nextState->getFlatten()].cend()) - qValues[currentState->getFlatten()][p]);
            toEGreedy(policies[currentState->getFlatten()], 5, std::distance(qValues[currentState->getFlatten()].cbegin(), std::max_element(qValues[currentState->getFlatten()].cbegin(), qValues[currentState->getFlatten()].cend())), epsilon);
            currentState = nextState;
            p = nextP;
        }
    }
};

template <class _MultiplyFunctor>
class OfflineQLearning {
    std::vector<std::vector<double>> qValues;
    std::vector<std::vector<double>> behaviorPolicies;
    std::vector<std::vector<double>> targetPolicies;
    std::vector<std::vector<size_t>> ns;
    const Playground& playground;
    size_t episodeLength;
    double discount;
    double epsilon;
    _MultiplyFunctor alphaFunctor;
public:
    OfflineQLearning(const Playground& playground, size_t episodeLength,  double discount, double epsilon, _MultiplyFunctor&& alphaFunctor)
        : qValues(playground.height * playground.width, std::vector<double>(size_t(5), 0.))
        , behaviorPolicies(playground.height * playground.width, std::vector<double>(size_t(5), 1. / 5))
        , targetPolicies(playground.height * playground.width, std::vector<double>(size_t(5), 0.))
        , ns(playground.height * playground.width, std::vector<size_t>(5, 0))
        , playground(playground) 
        , episodeLength(episodeLength)
        , discount(discount)
        , epsilon(epsilon)
        , alphaFunctor(std::forward<_MultiplyFunctor>(alphaFunctor))
    {}
    std::vector<std::vector<double>>& getPolicies() { return const_cast<std::vector<std::vector<double>>&>(const_cast<const OfflineQLearning<_MultiplyFunctor>&>(*this).getPolicies()); }
    const std::vector<std::vector<double>>& getPolicies() const { return targetPolicies; }
    void run() { run(this->epsilon); }
    void run(double epsilon) {
        AbstractBlock *currentState = playground.sample();
        // AbstractBlock *currentState = playground.map[0].get();
        size_t p = chooseAction(behaviorPolicies[currentState->getFlatten()], playground.gen);
        for (size_t i = 0; i < episodeLength; ++i) {
            auto [nextState, reward] = playground.act(currentState, static_cast<ActionType>(p));
            size_t nextP = chooseAction(behaviorPolicies[nextState->getFlatten()], playground.gen);
            qValues[currentState->getFlatten()][p] += alphaFunctor(ns[currentState->getFlatten()][p]) * (reward + discount * *std::max_element(qValues[nextState->getFlatten()].cbegin(), qValues[nextState->getFlatten()].cend()) - qValues[currentState->getFlatten()][p]);
            for (size_t i = 0, argmax = std::distance(qValues[currentState->getFlatten()].cbegin(), std::max_element(qValues[currentState->getFlatten()].cbegin(), qValues[currentState->getFlatten()].cend())); i < 5; ++i) {
                targetPolicies[currentState->getFlatten()][i] = i == argmax? 1.: 0.;
            }
            currentState = nextState;
            p = nextP;
        }
    }
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

