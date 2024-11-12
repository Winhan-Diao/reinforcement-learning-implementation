#pragma once
#include <vector>
#include <algorithm>
#include <cassert>
#include "playground.hpp"
#include "random_utils.hpp"
#include "neural/network.hpp"
#define ACTION_COUNTS 5

inline void foo() {}

template <class T>
inline auto toEGreedy(T&& policy, double epsilon) -> decltype(std::forward<T>(policy)) {
    auto size = policy.size();
    auto iter = std::max_element(policy.begin(), policy.end());
    std::for_each(policy.begin(), policy.end(), [&iter, size, epsilon](auto& v){
        if (&*iter == &v) {
            v = 1 - double(size - 1) / size * epsilon;
        } else {
            v = 1. / size * epsilon;
        }
    });
    return std::forward<T>(policy);
}
inline std::vector<double> toEGreedy(size_t policyCount, size_t maxargIndex, double epsilon) {
    double star = 1 - double(policyCount - 1) / policyCount * epsilon;
    double other = 1. / policyCount * epsilon;
    std::vector<double> tmp(policyCount, other);
    tmp[maxargIndex] = star;
    return tmp;
}
inline void toEGreedy(std::vector<double>& policy, size_t policyCount, size_t maxargIndex, double epsilon) {
    assert(policy.size() == policyCount);
    double star = 1 - double(policyCount - 1) / policyCount * epsilon;
    double other = 1. / policyCount * epsilon;
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
};

class TabularMethod {
protected:
    std::vector<std::vector<double>> qValues;
    std::vector<std::vector<double>> policies;
    const Playground& playground;
    double discount;
public:
    TabularMethod(const Playground& playground, double discount, std::vector<double> policiesActionTemplate = std::vector<double>(sizeof(ACTION_COUNTS)))
        : qValues(playground.width * playground.height, std::vector<double>(size_t(ACTION_COUNTS))) 
        , policies(playground.width * playground.height, policiesActionTemplate)
        , playground(playground)
        , discount(discount)
    {}
    std::vector<std::vector<double>>& getPolicies() { return const_cast<std::vector<std::vector<double>>&>(const_cast<const TabularMethod&>(*this).getPolicies()); }
    const std::vector<std::vector<double>>& getPolicies() const { return policies; }
    virtual void train(double epsilon, double learningRate, size_t length) = 0;
    virtual ~TabularMethod() = default;
};

class MonteCarlo: public TabularMethod {
public:
    MonteCarlo(const Playground& playground, double discount)
        : TabularMethod(playground, discount, toEGreedy(ACTION_COUNTS, 0, 1.))
    {}
    void train(double epsilon, double learningRate, size_t length) override {
        std::vector<Step> episode = std::vector<Step>();
        double g = 0;
        AbstractBlock *currentState = playground.sample();
        for (size_t i = 0; i < length; ++i) {
            size_t p = chooseAction(policies[currentState->getFlatten()], playground.gen);
            auto [nextState, reward] = playground.act(currentState, static_cast<ActionType>(p));
            episode.push_back(Step{currentState, static_cast<ActionType>(p), reward});
            currentState = nextState;
        }
        for (ssize_t i = length - 1; i >= 0; --i) {
            g = discount * g + episode[i].reward;
            qValues[episode[i].state->getFlatten()][size_t(episode[i].action)] += learningRate * (g - qValues[episode[i].state->getFlatten()][size_t(episode[i].action)]);
        }
        for (size_t i = 0; i < qValues.size(); ++i) {
            toEGreedy(policies[i], 5, std::distance(qValues[i].cbegin(), std::max_element(qValues[i].cbegin(), qValues[i].cend())), epsilon);
        }
    }
};

class Sarsa: public TabularMethod {
public:
    Sarsa(const Playground& playground, double discount)
        : TabularMethod(playground, discount, toEGreedy(ACTION_COUNTS, 0, 1.))
    {}
    void train(double epsilon, double learningRate, size_t length) override {
        AbstractBlock *currentState = playground.sample();
        size_t p = chooseAction(policies[currentState->getFlatten()], playground.gen);
        for (size_t i = 0; i < length; ++i) {
            auto [nextState, reward] = playground.act(currentState, static_cast<ActionType>(p));
            size_t nextP = chooseAction(policies[nextState->getFlatten()], playground.gen);
            qValues[currentState->getFlatten()][p] += learningRate * (reward + discount * qValues[nextState->getFlatten()][nextP] - qValues[currentState->getFlatten()][p]);
            toEGreedy(policies[currentState->getFlatten()], 5, std::distance(qValues[currentState->getFlatten()].cbegin(), std::max_element(qValues[currentState->getFlatten()].cbegin(), qValues[currentState->getFlatten()].cend())), epsilon);
            currentState = nextState;
            p = nextP;
        }
    }
};

class QLearning: public TabularMethod {
public:
    QLearning(const Playground& playground, double discount)
        : TabularMethod(playground, discount, toEGreedy(ACTION_COUNTS, 0, 1.))
    {}
    void train(double epsilon, double learningRate, size_t length) override {
        train(epsilon, learningRate, length, policies);
    }
    void train(double epsilon, double learningRate, size_t length, const std::vector<std::vector<double>>& behaviorPolicies) {
         AbstractBlock *currentState = playground.sample();
        size_t p = chooseAction(behaviorPolicies[currentState->getFlatten()], playground.gen);
        for (size_t i = 0; i < length; ++i) {
            auto [nextState, reward] = playground.act(currentState, static_cast<ActionType>(p));
            size_t nextP = chooseAction(behaviorPolicies[nextState->getFlatten()], playground.gen);
            qValues[currentState->getFlatten()][p] += learningRate * (reward + discount * *std::max_element(qValues[nextState->getFlatten()].cbegin(), qValues[nextState->getFlatten()].cend()) - qValues[currentState->getFlatten()][p]);
            toEGreedy(policies[currentState->getFlatten()], 5, std::distance(qValues[currentState->getFlatten()].cbegin(), std::max_element(qValues[currentState->getFlatten()].cbegin(), qValues[currentState->getFlatten()].cend())), epsilon);
            currentState = nextState;
            p = nextP;
        }
   }
   void explore(double epsilon, double learningRate, size_t length) {
        train(epsilon, learningRate, length, std::vector<std::vector<double>>(playground.height * playground.width, std::vector<double>(ACTION_COUNTS, 1. / ACTION_COUNTS)));
   }
};

class DeepQNetwork {
    Network targetNetwork;
    Network qNetwork;
    std::vector<Experience> expPool;
    const Playground& playground;
    double discount;
public:
    DeepQNetwork(const Playground& playground, double discount)
        : targetNetwork(2, ACTION_COUNTS, std::vector<size_t>{128}, std::vector<ActivationFunctions>{ActivationFunctions::LEAKYRELU}, ActivationFunctions::LEAKYRELU, LossFunctions::MSE)
        , qNetwork(targetNetwork)
        , expPool()
        , playground(playground)
        , discount(discount)
    {}
    Network& getTargetNetwork() { return targetNetwork; }
    const Network& getTargetNetwork() const { return targetNetwork; }
    void expReplay(double epsilon, size_t length) {
        expPool.resize(length);
        AbstractBlock *currentState = playground.map[9 * 10 + 8].get();
        for (size_t i = 0; i < length; ++i) {
            std::valarray<double> qValues = targetNetwork.run({double(currentState->getX()), double(currentState->getY())});
            ActionType action = static_cast<ActionType>(std::distance(std::cbegin(qValues), std::max_element(std::cbegin(qValues), std::cend(qValues))));
            ActionType finalAction = static_cast<ActionType>(chooseAction(toEGreedy(ACTION_COUNTS, size_t(action), epsilon), playground.gen));
            auto [nextState, reward] = playground.act(currentState, finalAction);
            expPool[i] = Experience{currentState, finalAction, reward, nextState};
            currentState = nextState;
        }
    }
    void expReplay(size_t length) {
        expPool.resize(length);
        AbstractBlock *currentState = playground.map[9 * 10 + 8].get();
        for (size_t i = 0; i < length; ++i) {
            ActionType action = static_cast<ActionType>(std::uniform_int_distribution(0, ACTION_COUNTS - 1)(playground.gen));
            auto [nextState, reward] = playground.act(currentState, action);
            expPool[i] = Experience{currentState, action, reward, nextState};
            currentState = nextState;
        }
    }
    void train(double learningRate, size_t batchLength) {
        for (size_t i = 0; i < batchLength; ++i) {
            const Experience& exp = expPool[std::uniform_int_distribution(size_t(0), expPool.size() - 1)(playground.gen)];
            std::valarray<double> qValues = qNetwork.run({double(exp.state->getX()), double(exp.state->getY())});
            std::valarray<double> nextQValues = targetNetwork.run({double(exp.nextState->getX()), double(exp.nextState->getY())});
            qValues[size_t(exp.action)] = exp.reward + discount * *std::max_element(std::cbegin(nextQValues), std::cend(nextQValues));
            qNetwork.train({double(exp.state->getX()), double(exp.state->getY())}, qValues, learningRate);
            // qNetwork.train({double(exp.state->getX()), double(exp.state->getY())}, {double(exp.state->getX()), double(exp.state->getY()), std::pow(exp.state->getX(), 2), std::pow(exp.state->getY(), 2), 1.}, learningRate);       //debug
        }
        targetNetwork = qNetwork;
    }
};

inline void showPolicy(const std::vector<std::vector<double>>& policy, const Playground& playground, std::ostream& os = std::cout) {
    for (uint16_t h = 0; h < playground.height; ++h) {
        for (uint16_t w = 0; w < playground.width; ++w) {
            const auto& aPolicy = policy[h * playground.width + w];
            auto argmaxPolicy = std::distance(aPolicy.cbegin(), std::max_element(aPolicy.cbegin(), aPolicy.cend()));
            switch (argmaxPolicy) {
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

inline void showPolicy(Network& qValues, const Playground& playground, std::ostream& os = std::cout) {
    for (uint16_t h = 0; h < playground.height; ++h) {
        for (uint16_t w = 0; w < playground.width; ++w) {
            const auto& aPolicy = qValues.run({double(w), double(h)});
            auto argmaxPolicy = std::distance(std::cbegin(aPolicy), std::max_element(std::cbegin(aPolicy), std::cend(aPolicy)));
            // std::cout << aPolicy[0] << ' ' << aPolicy[1] << ' ' << aPolicy[2] << ' ' << aPolicy[3] << ' ' << aPolicy[4] << "\r\n";      //debug
            switch (argmaxPolicy) {
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

inline void printQValues(Network& qValues, const Playground& playground, std::ostream& os = std::cout) {
    for (uint16_t h = 0; h < playground.height; ++h) {
        for (uint16_t w = 0; w < playground.width; ++w) {
            const auto& aPolicy = qValues.run({double(w), double(h)});
            os << "h: " << h 
                      << "; w: " << w 
                      << " {";
            for (size_t i = 0; i < ACTION_COUNTS; ++i) {
                os << aPolicy[i] << ", ";
            }
            os << '}' << "\r\n";
        }
    }
}
