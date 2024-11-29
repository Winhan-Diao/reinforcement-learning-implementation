#pragma once
#include <vector>
#include <algorithm>
#include <cassert>
#include "playground.hpp"
#include "random_utils.hpp"
#include "neural/network.hpp"
#include "utils.hpp"
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

class ExperienceBuffer {
    std::vector<Experience> pool;
    size_t volume;
    size_t current;
    bool full;
public:
    ExperienceBuffer(size_t volume): pool(volume), volume(volume), current(0), full(false) {}
    const Experience& sample(std::mt19937& gen) const {
        return pool[std::uniform_int_distribution(size_t(0), full? volume: current)(gen)];
    }
    void add(Experience e) {
        if (current < (volume - 1))
            pool[++current] = e;
        else {
            current = 0;
            pool[current] = e;
            full = true;
        }
    }
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
        episode.reserve(length);
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
        : targetNetwork(2, ACTION_COUNTS, std::vector<size_t>{128}, std::vector<ActivationFunctions>{ActivationFunctions::LEAKYRELU}, ActivationFunctions::LEAKYRELU, LossFunctions::MSE, true)
        , qNetwork(targetNetwork)
        , expPool()
        , playground(playground)
        , discount(discount)
    {}
    Network& getTargetNetwork() { return targetNetwork; }
    const Network& getTargetNetwork() const { return targetNetwork; }
    void expReplay(double epsilon, size_t length) {
        expPool.resize(length);
        AbstractBlock *currentState = playground.sample();
        for (size_t i = 0; i < length; ++i) {
            std::valarray<double> qValues = targetNetwork.run({double(currentState->getX()), double(currentState->getY())});
            // printValarray(qValues);     //debug
            ActionType action = static_cast<ActionType>(std::distance(std::cbegin(qValues), std::max_element(std::cbegin(qValues), std::cend(qValues))));
            ActionType finalAction = static_cast<ActionType>(chooseAction(toEGreedy(ACTION_COUNTS, size_t(action), epsilon), playground.gen));
            auto [nextState, reward] = playground.act(currentState, finalAction);
            expPool[i] = Experience{currentState, finalAction, reward, nextState};
            currentState = nextState;
        }
    }
    void expReplay(size_t length) {
        expPool.resize(length);
        AbstractBlock *currentState = playground.sample();
        for (size_t i = 0; i < length; ++i) {
            ActionType action = static_cast<ActionType>(std::uniform_int_distribution(0, ACTION_COUNTS - 1)(playground.gen));
            auto [nextState, reward] = playground.act(currentState, action);
            expPool[i] = Experience{currentState, action, reward, nextState};
            currentState = nextState;
        }
    }
    // Requires new experience replays before training
    void train(double learningRate, size_t batchLength) {
        for (size_t i = 0; i < batchLength; ++i) {
            const Experience& exp = expPool[std::uniform_int_distribution(size_t(0), expPool.size() - 1)(playground.gen)];
            std::valarray<double> qValues = qNetwork.run({double(exp.state->getX()), double(exp.state->getY())});
            std::valarray<double> nextQValues = targetNetwork.run({double(exp.nextState->getX()), double(exp.nextState->getY())});
            // qValues[size_t(exp.action)] = exp.reward + discount * *std::max_element(std::cbegin(nextQValues), std::cend(nextQValues));
            qNetwork.train({double(exp.state->getX()), double(exp.state->getY())}, exp.reward + discount * *std::max_element(std::cbegin(nextQValues), std::cend(nextQValues)), size_t(exp.action), learningRate);
            // qNetwork.train({double(exp.state->getX()), double(exp.state->getY())}, {double(exp.state->getX()), double(exp.state->getY()), std::pow(exp.state->getX(), 2), std::pow(exp.state->getY(), 2), 1.}, learningRate);       //debug
        }
        targetNetwork = qNetwork;
    }
};

class REINFORCE {
    Network policyNetwork;
    const Playground& playground;
    double discount;
public:
    REINFORCE(const Playground& playground, double discount)
        : policyNetwork(2, ACTION_COUNTS, std::vector<size_t>{128}, std::vector<ActivationFunctions>{ActivationFunctions::LEAKYRELU}, ActivationFunctions::SOFTMAX, LossFunctions::POLICY_GRADIENT_LOSS, true)
        , playground(playground)
        , discount(discount)
    {}
    Network& getPolicyNetwork() { return policyNetwork; }
    const Network& getPolicyNetwork() const { return policyNetwork; }    
    void train(double learningRate, size_t length) {
        std::vector<Step> episode;
        episode.reserve(length);
        AbstractBlock *currentState = playground.sample();
        for (size_t i = 0; i < length; ++i) {
            ActionType action = static_cast<ActionType>(chooseAction(policyNetwork.run({double(currentState->getX()), double(currentState->getY())}), playground.gen));
            auto[nextState, reward] = playground.act(currentState, action);
            episode.emplace_back(currentState, action, reward);
            currentState = nextState;
        }
        double meanReward = std::accumulate(std::cbegin(episode), std::cend(episode), 0., [](double acc, const Step& s) -> double { return acc + s.reward; }) / episode.size();
        double stdReward = std::accumulate(std::cbegin(episode), std::cend(episode), 0., [meanReward](double acc, const Step& s) -> double { return acc + std::pow(s.reward - meanReward, 2); });
        std::for_each(std::begin(episode), std::end(episode), [meanReward, stdReward](Step& s) { s.reward = (s.reward - meanReward) / (stdReward + 1e-7); });
        double g = 0;
        for (ssize_t i = length - 1; i >= 0; --i) {
            g = discount * g + episode[i].reward;
            policyNetwork.train({double(currentState->getX()), double(currentState->getY())}, g, size_t(episode[i].action), learningRate);
        }
    }
};

class QActorCritic {
    Network qNetwork;
    Network policyNetwork;
    const Playground& playground;
    double discount;
    ssize_t startIndex;
public:
    QActorCritic(const Playground& playground, double discount, ssize_t startIndex = -1)
        : qNetwork(2, ACTION_COUNTS, std::vector<size_t>{128}, std::vector<ActivationFunctions>{ActivationFunctions::LEAKYRELU}, ActivationFunctions::LEAKYRELU, LossFunctions::MSE, true)
        , policyNetwork(2, ACTION_COUNTS, std::vector<size_t>{128}, {ActivationFunctions::LEAKYRELU}, ActivationFunctions::STABLE_SOFTMAX_V3, LossFunctions::CUSTOM, true)
        , playground(playground) 
        , discount(discount)
        , startIndex(startIndex)
    {
        qNetwork.assignData(policyNetwork);
    }
    Network& getPolicyNetwork() { return policyNetwork; }
    const Network& getPolicyNetwork() const { return policyNetwork; }    
    Network& getQNetwork() { return qNetwork; }
    const Network& getQNetwork() const { return qNetwork; }    
    void train(double qLearningRate, double policyLearningRate, size_t length, double normalize = 0.0) {
        AbstractBlock *currentState = (startIndex == -1)? playground.sample(): playground.map[startIndex].get();
        for (size_t i = 0; i < length; ++i) {
            std::valarray<double> policy = policyNetwork.run({double(currentState->getX()), double(currentState->getY())});
            // toSumOne(std::begin(policy), std::end(policy));
            ActionType action = static_cast<ActionType>(chooseAction(policy, playground.gen));      //(1/3)
            // ActionType action = static_cast<ActionType>(std::distance(std::cbegin(policy), std::max_element(std::cbegin(policy), std::cend(policy))));       //(2/3)
            // ActionType action = static_cast<ActionType>(chooseAction(policy, playground.gen, normalize));      //(3/3)
            auto[nextState, reward] = playground.act(currentState, action);
            std::valarray<double> nextPolicy = policyNetwork.run({double(nextState->getX()), double(nextState->getY())});
            // ActionType nextAction = static_cast<ActionType>(chooseAction(nextPolicy, playground.gen));       //(1/2)
            ActionType nextAction = static_cast<ActionType>(std::distance(std::cbegin(nextPolicy), std::max_element(std::cbegin(nextPolicy), std::cend(nextPolicy))));      //(2/2)
            
            double nextQValue = qNetwork.run({double(nextState->getX()), double(nextState->getY())})[size_t(nextAction)];
            qNetwork.train({double(currentState->getX()), double(currentState->getY())}, reward + discount * nextQValue, size_t(action), qLearningRate);
            std::valarray<double> improvedQValues = qNetwork.run({double(currentState->getX()), double(currentState->getY())});
            policyNetwork.train({double(currentState->getX()), double(currentState->getY())}, -improvedQValues / (policy + 1.e-10) - 0 * (-policy * std::log(policy)).sum(), policyLearningRate);
            currentState = nextState;
        }
    }
};

class AdvantageActorCritic {
    Network vNetwork;
    Network policyNetwork;
    const Playground& playground;
    double discount;
    ssize_t startIndex;
public:
    AdvantageActorCritic(const Playground& playground, double discount, ssize_t startIndex = -1)
        : vNetwork(2, 1, std::vector<size_t>{16}, std::vector<ActivationFunctions>{ActivationFunctions::LEAKYRELU}, ActivationFunctions::LEAKYRELU, LossFunctions::MSE, true)
        , policyNetwork(2, ACTION_COUNTS, std::vector<size_t>{128}, {ActivationFunctions::LEAKYRELU}, ActivationFunctions::LOG_SOFTMAX, LossFunctions::POLICY_GRADIENT_LOSS, true)
        , playground(playground) 
        , discount(discount)
        , startIndex(startIndex)
    {}
    Network& getPolicyNetwork() { return policyNetwork; }
    const Network& getPolicyNetwork() const { return policyNetwork; }    
    Network& getVNetwork() { return vNetwork; }
    const Network& getVNetwork() const { return vNetwork; }    
    void train(double vLearningRate, double policyLearningRate, size_t length) {
        AbstractBlock *currentState = (startIndex == -1)? playground.sample(): playground.map[startIndex].get();
        for (size_t i = 0; i < length; ++i) {
            std::valarray<double> policy = policyNetwork.run({double(currentState->getX()), double(currentState->getY())});
            toSumOne(std::begin(policy), std::end(policy));
            size_t action = chooseAction(policy, playground.gen);     //(1/2)
            // size_t action = std::distance(std::cbegin(policy), std::max_element(std::cbegin(policy), std::cend(policy)));      //(2/2)
            auto[nextState, reward] = playground.act(currentState, static_cast<ActionType>(action));
            double tdTarget = reward + discount * vNetwork.run({double(nextState->getX()), double(nextState->getY())})[0];
            double advantage = tdTarget - vNetwork.run({double(currentState->getX()), double(currentState->getY())})[0];
            
            vNetwork.train({double(currentState->getX()), double(currentState->getY())}, {tdTarget}, vLearningRate);
            policyNetwork.train({double(currentState->getX()), double(currentState->getY())}, advantage, action, policyLearningRate);
            currentState = nextState;
        }
    }
};

class AdvantageActorCriticV3 {
    Network vNetwork;
    Network policyNetwork;
    const Playground& playground;
    double discount;
    ssize_t startIndex;
    std::valarray<std::valarray<double>> trajectoriesAdv;
    std::valarray<std::valarray<std::valarray<double>>> trajectoriesAdvUnsquzeed;
    std::vector<std::vector<size_t>> trajectoriesActions;
    std::valarray<std::valarray<std::valarray<double>>> trajectoriesStates;
public:
    AdvantageActorCriticV3(const Playground& playground, double discount, ssize_t startIndex)
        : vNetwork(2, 1, std::vector<size_t>{128}, std::vector<ActivationFunctions>{ActivationFunctions::LEAKYRELU}, ActivationFunctions::LEAKYRELU, LossFunctions::CUSTOM, true)
        , policyNetwork(2, ACTION_COUNTS, std::vector<size_t>{128}, {ActivationFunctions::LEAKYRELU}, ActivationFunctions::STABLE_SOFTMAX_V3, LossFunctions::POLICY_GRADIENT_LOSS, true)
        , playground(playground) 
        , discount(discount)
        , startIndex(startIndex)
        , trajectoriesAdv()
        , trajectoriesAdvUnsquzeed()
        , trajectoriesActions()
        , trajectoriesStates()
    {}
    Network& getPolicyNetwork() { return policyNetwork; }
    const Network& getPolicyNetwork() const { return policyNetwork; }    
    Network& getVNetwork() { return vNetwork; }
    const Network& getVNetwork() const { return vNetwork; }    
    void generateTrajectories(size_t trajectoryLength, size_t trajectoryCounts) {
        trajectoriesAdv = std::valarray<std::valarray<double>>(std::valarray<double>(trajectoryLength), trajectoryCounts);
        trajectoriesAdvUnsquzeed = std::valarray<std::valarray<std::valarray<double>>>(std::valarray<std::valarray<double>>(trajectoryLength), trajectoryCounts);
        trajectoriesActions = std::vector<std::vector<size_t>>(trajectoryCounts, std::vector<size_t>(trajectoryLength));
        trajectoriesStates = std::valarray<std::valarray<std::valarray<double>>>(std::valarray<std::valarray<double>>(trajectoryLength), trajectoryCounts);
        for (size_t i = 0; i < trajectoryCounts; ++i) {
            AbstractBlock *currentState = (startIndex == -1)? playground.sample(): playground.map[startIndex].get();
            for (size_t j = 0; j < trajectoryLength; ++j) {
                std::valarray<double> policy = policyNetwork.run({double(currentState->getX()), double(currentState->getY())});
                size_t action = chooseAction(policy, playground.gen);
                auto[nextState, reward] = playground.act(currentState, static_cast<ActionType>(action));
                trajectoriesAdv[i][j] = -vNetwork.run({double(currentState->getX()), double(currentState->getY())})[0] + reward + discount * vNetwork.run({double(nextState->getX()), double(nextState->getY())})[0];
                // trajectoriesAdvUnsquzeed[i][j] = std::valarray<double>{trajectoriesAdv[i][j]};
                trajectoriesActions[i][j] = action;
                trajectoriesStates[i][j] = std::valarray<double>{double(currentState->getX()), double(currentState->getY())};
                currentState = nextState;
            }
            double advMean = trajectoriesAdv[i].sum() / trajectoriesAdv[i].size();
            double advStd = std::sqrt( std::pow(trajectoriesAdv[i] - advMean, 2).sum() / trajectoriesAdv[i].size());
            trajectoriesAdv[i] = (trajectoriesAdv[i] - advMean) / (advStd + 1.e-100);
            for (size_t j = 0; j < trajectoryLength; ++j) 
                trajectoriesAdvUnsquzeed[i][j] = std::valarray<double>{trajectoriesAdv[i][j]};
        }
    }
    void train(double vLearningRate, double policyLearningRate) {
        for (size_t i = 0; i < trajectoriesActions.size(); ++i) {
            policyNetwork.batchedTrain(trajectoriesStates[i], trajectoriesAdv[i] / trajectoriesActions.size(), trajectoriesActions[i], policyLearningRate, 6);
            vNetwork.batchedTrain(trajectoriesStates[i], -trajectoriesAdvUnsquzeed[i], vLearningRate, 6);
        }
    }
};

class AdvantageActorCriticV4 {
    Network vNetwork;
    Network policyNetwork;
    const Playground& playground;
    double discount;
    ssize_t startIndex;
    std::valarray<double> trajectoriesAdv;
    std::valarray<std::valarray<double>> trajectoriesAdvUnsquzeed;
    std::vector<size_t> trajectoriesActions;
    std::valarray<std::valarray<double>> trajectoriesStates;
    size_t trajectoryCounts;
public:
    AdvantageActorCriticV4(const Playground& playground, double discount, ssize_t startIndex)
        : vNetwork(2, 1, std::vector<size_t>{128}, std::vector<ActivationFunctions>{ActivationFunctions::LEAKYRELU}, ActivationFunctions::LEAKYRELU, LossFunctions::CUSTOM, true)
        , policyNetwork(2, ACTION_COUNTS, std::vector<size_t>{128}, {ActivationFunctions::LEAKYRELU}, ActivationFunctions::STABLE_SOFTMAX_V3, LossFunctions::POLICY_GRADIENT_LOSS, true)
        , playground(playground) 
        , discount(discount)
        , startIndex(startIndex)
        , trajectoriesAdv()
        , trajectoriesAdvUnsquzeed()
        , trajectoriesActions()
        , trajectoriesStates()
        , trajectoryCounts(0)
    {}
    Network& getPolicyNetwork() { return policyNetwork; }
    const Network& getPolicyNetwork() const { return policyNetwork; }    
    Network& getVNetwork() { return vNetwork; }
    const Network& getVNetwork() const { return vNetwork; }    
    void generateTrajectories(size_t trajectoryLength, size_t trajectoryCounts) {
        this->trajectoryCounts = trajectoryCounts;
        trajectoriesAdv = std::valarray<double>(trajectoryCounts * trajectoryLength);
        trajectoriesAdvUnsquzeed = std::valarray<std::valarray<double>>(trajectoryLength * trajectoryCounts);
        trajectoriesActions = std::vector<size_t>(trajectoryCounts * trajectoryLength);
        trajectoriesStates = std::valarray<std::valarray<double>>(trajectoryLength * trajectoryCounts);
        for (size_t i = 0; i < trajectoryCounts; ++i) {
            AbstractBlock *currentState = (startIndex == -1)? playground.sample(): playground.map[startIndex].get();
            for (size_t j = 0; j < trajectoryLength; ++j) {
                std::valarray<double> policy = policyNetwork.run({double(currentState->getX()), double(currentState->getY())});
                size_t action = chooseAction(policy, playground.gen);
                auto[nextState, reward] = playground.act(currentState, static_cast<ActionType>(action));
                trajectoriesAdv[i * trajectoryLength + j] = -vNetwork.run({double(currentState->getX()), double(currentState->getY())})[0] + reward + discount * vNetwork.run({double(nextState->getX()), double(nextState->getY())})[0];
                trajectoriesAdvUnsquzeed[i * trajectoryLength + j] = std::valarray<double>{trajectoriesAdv[i * trajectoryLength + j]};
                trajectoriesActions[i * trajectoryLength + j] = action;
                trajectoriesStates[i * trajectoryLength + j] = std::valarray<double>{double(currentState->getX()), double(currentState->getY())};
                currentState = nextState;
            }
            double advMean = trajectoriesAdv.sum() / trajectoriesAdv.size();
            double advStd = std::sqrt( std::pow(trajectoriesAdv - advMean, 2).sum() / trajectoriesAdv.size());
            trajectoriesAdv = (trajectoriesAdv - advMean) / (advStd + 1.e-100);
            // for (size_t j = 0; j < trajectoryLength; ++j) 
            //     trajectoriesAdvUnsquzeed[i * trajectoryLength + j] = std::valarray<double>{trajectoriesAdv[i * trajectoryLength + j]};
        }
    }
    void train(double vLearningRate, double policyLearningRate) {
        policyNetwork.batchedTrain(trajectoriesStates, trajectoriesAdv / trajectoryCounts, trajectoriesActions, policyLearningRate, 6);
        vNetwork.batchedTrain(trajectoriesStates, -trajectoriesAdvUnsquzeed, vLearningRate, 6);
    }
};


class[[deprecated]] AdvantageActorCriticV2 {
    Network qNetwork;
    Network policyNetwork;
    const Playground& playground;
    double discount;
public:
    AdvantageActorCriticV2(const Playground& playground, double discount)
        : qNetwork(2, ACTION_COUNTS, std::vector<size_t>{128}, std::vector<ActivationFunctions>{ActivationFunctions::LEAKYRELU}, ActivationFunctions::LEAKYRELU, LossFunctions::MSE, true)
        , policyNetwork(2, ACTION_COUNTS, std::vector<size_t>{128}, {ActivationFunctions::LEAKYRELU}, ActivationFunctions::STABLE_SOFTMAX, LossFunctions::CUSTOM, true)
        , playground(playground) 
        , discount(discount)
    {
        qNetwork.assignData(policyNetwork);
    }
    Network& getPolicyNetwork() { return policyNetwork; }
    const Network& getPolicyNetwork() const { return policyNetwork; }    
    Network& getQNetwork() { return qNetwork; }
    const Network& getQNetwork() const { return qNetwork; }    
    void train(double qLearningRate, double policyLearningRate, size_t length) {
        AbstractBlock *currentState = playground.sample();
        for (size_t i = 0; i < length; ++i) {
            std::valarray<double> policy = policyNetwork.run({double(currentState->getX()), double(currentState->getY())});
            ActionType action = static_cast<ActionType>(chooseAction(policy, playground.gen));      //(1/2)
            // ActionType action = static_cast<ActionType>(std::distance(std::cbegin(policy), std::max_element(std::cbegin(policy), std::cend(policy))));       //(2/2)
            auto[nextState, reward] = playground.act(currentState, action);
            std::valarray<double> nextPolicy = policyNetwork.run({double(nextState->getX()), double(nextState->getY())});
            // ActionType nextAction = static_cast<ActionType>(chooseAction(nextPolicy, playground.gen));       //(1/2)
            ActionType nextAction = static_cast<ActionType>(std::distance(std::cbegin(nextPolicy), std::max_element(std::cbegin(nextPolicy), std::cend(nextPolicy))));      //(2/2)
            
            std::valarray<double> qValues = qNetwork.run({double(currentState->getX()), double(currentState->getY())});
            double nextQValue = qNetwork.run({double(nextState->getX()), double(nextState->getY())})[size_t(nextAction)];
            qNetwork.train({double(currentState->getX()), double(currentState->getY())}, reward + discount * nextQValue, size_t(action), qLearningRate);
            std::valarray<double> improvedQValues = qNetwork.run({double(currentState->getX()), double(currentState->getY())});
            double stateValue = expectation(policy, improvedQValues);        //(1/2)
            // double stateValue = improvedQValues[std::distance(std::cbegin(policy), std::max_element(std::cbegin(policy), std::cend(policy)))];        //(2/2)
            policyNetwork.train({double(currentState->getX()), double(currentState->getY())}, -(improvedQValues - stateValue) / (policy + 1.e-10) - 0.0 * (-policy * std::log(policy)).sum(), policyLearningRate);
            currentState = nextState;
        }
    }
};

class PPOClip {
    Network vNetwork;
    Network policyNetwork;
    Network oldPolicyNetwork;
    const Playground& playground;
    std::vector<Experience> expPool;
    std::valarray<double> advantages;
    std::valarray<double> tdReturns;
    double discount;
    ssize_t startIndex;
public:
    PPOClip(const Playground& playground, double discount, ssize_t startIndex = -1)
        : vNetwork(2, 1, std::vector<size_t>{128}, std::vector<ActivationFunctions>{ActivationFunctions::LEAKYRELU}, ActivationFunctions::LEAKYRELU, LossFunctions::MSE, true)
        , policyNetwork(2, ACTION_COUNTS, std::vector<size_t>{128}, {ActivationFunctions::LEAKYRELU}, ActivationFunctions::LOG_SOFTMAX, LossFunctions::CUSTOM, true)
        , oldPolicyNetwork(policyNetwork)
        , playground(playground) 
        , expPool()
        , advantages()
        , tdReturns()
        , discount(discount)
        , startIndex(startIndex)
    {}
    Network& getPolicyNetwork() { return oldPolicyNetwork; }
    const Network& getPolicyNetwork() const { return oldPolicyNetwork; }    
    Network& getVNetwork() { return vNetwork; }
    const Network& getVNetwork() const { return vNetwork; }    
    void generateExperience(size_t length) {
        expPool.resize(length);
        advantages.resize(length);
        tdReturns.resize(length);
        AbstractBlock *currentState = (startIndex == -1)? playground.sample(): playground.map[startIndex].get();
        for (size_t i = 0; i < length; ++i) {
            std::valarray<double> policy = oldPolicyNetwork.run({double(currentState->getX()), double(currentState->getY())});
            // policy = policy / 2 + .5;
            toSumOne(std::begin(policy), std::end(policy));
            size_t action = chooseAction(policy, playground.gen);
            auto[nextState, reward] = playground.act(currentState, static_cast<ActionType>(action));
            expPool[i] = Experience{currentState, static_cast<ActionType>(action), reward, nextState};
            tdReturns[i] = reward + discount * vNetwork.run({double(nextState->getX()), double(nextState->getY())})[0];
            advantages[i] = tdReturns[i] - vNetwork.run({double(currentState->getX()), double(currentState->getY())})[0];
            currentState = nextState;
        }
        double advMean = advantages.sum() / advantages.size();
        double advStd = std::sqrt( std::pow(advantages - advMean, 2).sum() / advantages.size());
        advantages = (advantages - advMean) / (advStd + 1.e-100);
    }
    void train(double vLearningRate, double policyLearningRate, double epsilon) {
        for (size_t i = 0; i < expPool.size(); ++i) {
            double oldPolicy = oldPolicyNetwork.run({double(expPool[i].state->getX()), double(expPool[i].state->getY())})[size_t(expPool[i].action)];
            double policy = policyNetwork.run({double(expPool[i].state->getX()), double(expPool[i].state->getY())})[size_t(expPool[i].action)];
            double ratio = policy / oldPolicy;
            // double clipRatio = std::clamp(ratio, 1 - epsilon, 1 + epsilon);
            if (advantages[i] > 0 && (ratio < (1 + epsilon)) || advantages[i] < 0 && (ratio > (1 - epsilon)))
                policyNetwork.train({double(expPool[i].state->getX()), double(expPool[i].state->getY())}, -advantages[i] / (oldPolicy + 1.e-100), size_t(expPool[i].action), policyLearningRate);
            // if (((policy / oldPolicy) <= double(1 + epsilon)) && ((policy / oldPolicy) >= double(1 - epsilon))) {
                // policyNetwork.train({double(expPool[i].state->getX()), double(expPool[i].state->getY())}, -advantages[i] / oldPolicy, size_t(expPool[i].action), policyLearningRate);
            // }
            vNetwork.train({double(expPool[i].state->getX()), double(expPool[i].state->getY())}, {tdReturns[i]}, vLearningRate);
        }
        oldPolicyNetwork = policyNetwork;
    }

};

// class DDPG {
//     Network qNetwork;
//     Network targetQNetwork;
//     Network policyNetwork;
//     Network targetPolicyNetwork;
//     ExperienceBuffer expBuffer;
//     const Playground& playground;
//     double discount;
// public:
//     DDPG(const Playground& playground, double discount, size_t expBufferSize)
//         : qNetwork(2, ACTION_COUNTS, std::vector<size_t>{128}, std::vector<ActivationFunctions>{ActivationFunctions::LEAKYRELU}, ActivationFunctions::LEAKYRELU, LossFunctions::MSE, true)
//         , targetQNetwork(qNetwork)
//         , policyNetwork(2, ACTION_COUNTS, std::vector<size_t>{128}, std::vector<ActivationFunctions>{ActivationFunctions::LEAKYRELU}, ActivationFunctions::STABLE_SOFTMAX, LossFunctions::CUSTOM, true)
//         , targetPolicyNetwork(policyNetwork)
//         , expBuffer(expBufferSize)
//         , playground(playground)
//         , discount(discount)
//     {}
//     void train(double qLearningRate, double policyLearningRate, size_t length) {
//         // expBuffer.add();
//     }
// };


inline void showPolicy(const std::vector<std::vector<double>>& policy, const Playground& playground, std::ostream& os = std::cout) {
    for (uint16_t h = 0; h < playground.height; ++h) {
        for (uint16_t w = 0; w < playground.width; ++w) {
            const auto& aPolicy = policy[h * playground.width + w];
            auto argmaxPolicy = std::distance(aPolicy.cbegin(), std::max_element(aPolicy.cbegin(), aPolicy.cend()));
            switch (ActionType(argmaxPolicy)) {
                case ActionType::UP:
                    os << "∧";
                    break;
                case ActionType::RIGHT:
                    os << ">";
                    break;
                case ActionType::DOWN:
                    os << "∨";
                    break;
                case ActionType::LEFT:
                    os << "<";
                    break;
                case ActionType::NONE:
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

inline void showPolicy(Network& policies, const Playground& playground, std::ostream& os = std::cout) {
    for (uint16_t h = 0; h < playground.height; ++h) {
        for (uint16_t w = 0; w < playground.width; ++w) {
            const auto& aPolicy = policies.run({double(w), double(h)});
            auto argmaxPolicy = std::distance(std::cbegin(aPolicy), std::max_element(std::cbegin(aPolicy), std::cend(aPolicy)));
            // std::cout << aPolicy[0] << ' ' << aPolicy[1] << ' ' << aPolicy[2] << ' ' << aPolicy[3] << ' ' << aPolicy[4] << "\r\n";      //debug
            switch (ActionType(argmaxPolicy)) {
                case ActionType::UP:
                    os << "∧";
                    break;
                case ActionType::RIGHT:
                    os << ">";
                    break;
                case ActionType::DOWN:
                    os << "∨";
                    break;
                case ActionType::LEFT:
                    os << "<";
                    break;
                case ActionType::NONE:
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

[[deprecated]]inline void showOneOutputPolicy(Network& policies, const Playground& playground, std::ostream& os = std::cout) {
    for (uint16_t h = 0; h < playground.height; ++h) {
        for (uint16_t w = 0; w < playground.width; ++w) {
            for (int i = 0; i < ACTION_COUNTS; ++i) {

            }
            const auto& aPolicy = policies.run({double(w), double(h)});
            auto argmaxPolicy = std::distance(std::cbegin(aPolicy), std::max_element(std::cbegin(aPolicy), std::cend(aPolicy)));
            // std::cout << aPolicy[0] << ' ' << aPolicy[1] << ' ' << aPolicy[2] << ' ' << aPolicy[3] << ' ' << aPolicy[4] << "\r\n";      //debug
            switch (ActionType(argmaxPolicy)) {
                case ActionType::UP:
                    os << "∧";
                    break;
                case ActionType::RIGHT:
                    os << ">";
                    break;
                case ActionType::DOWN:
                    os << "∨";
                    break;
                case ActionType::LEFT:
                    os << "<";
                    break;
                case ActionType::NONE:
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

inline void printScaledValues(Network& qValues, const Playground& playground, std::ostream& os = std::cout) {
    for (uint16_t h = 0; h < playground.height; ++h) {
        for (uint16_t w = 0; w < playground.width; ++w) {
            auto aPolicy = qValues.run({double(w), double(h)});
            toSumOne(std::begin(aPolicy), std::end(aPolicy));
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

inline void printStateValues(Network& qValues, const Playground& playground, std::ostream& os = std::cout) {
    for (uint16_t h = 0; h < playground.height; ++h) {
        for (uint16_t w = 0; w < playground.width; ++w) {
            const auto& value = qValues.run({double(w), double(h)});
            os << "h: " << h 
                      << "; w: " << w 
                      << " {";
            os << value[0];
            os << '}' << "\r\n";
        }
    }
}

