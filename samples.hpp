#pragma once
#include <fstream>
#include "methods.hpp"

void mcDemo(const std::string& fileName, size_t length = 1'000, double learningRate = .001, size_t exploringEpoch = 50'000, size_t exploitingEpoch = 50'000, double exploringEpsilon = .9, double discount = .5) {
    std::cout << "Running " << "mcDemo" << "\r\n";
    Playground pg1;
    if (std::ifstream is{fileName, std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    std::cout << "Exploring State" << "\r\n";
    MonteCarlo eMC(pg1, discount);
    showPolicy(eMC.getPolicies(), pg1);
    puts("");
    for (int i = 0; i < exploringEpoch; ++i) {
        eMC.train(exploringEpsilon, learningRate, length);
        if (i % (exploringEpoch / 10) == 0) {
            showPolicy(eMC.getPolicies(), pg1);
            puts("");
        }
    }
    std::cout << "Exploiting State" << "\r\n";
    for (int i = 0; i < exploitingEpoch; ++i) {
        eMC.train(std::pow(1 + 1. / exploitingEpoch, - 3 * i) * 0.9, learningRate, length);
        if (i % (exploitingEpoch / 10) == 0) {
            showPolicy(eMC.getPolicies(), pg1);
            puts("");
        }
    }
}

void sarsaDemo(const std::string& fileName, size_t length = 1'000, double learningRate = .001, size_t exploringEpoch = 50'000, size_t exploitingEpoch = 50'000, double exploringEpsilon = .9, double discount = .5) {
    std::cout << "Running " << "sarsaDemo" << "\r\n";
    Playground pg1;
    if (std::ifstream is{fileName, std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    std::cout << "Exploring State" << "\r\n";
    Sarsa sarsa(pg1, discount);
    showPolicy(sarsa.getPolicies(), pg1);
    puts("");
    for (int i = 0; i < exploringEpoch; ++i) {
        sarsa.train(exploringEpsilon, learningRate, length);
        if (i % (exploringEpoch / 10) == 0) {
            showPolicy(sarsa.getPolicies(), pg1);
            puts("");
        }
    }
    std::cout << "Exploiting State" << "\r\n";
    for (int i = 0; i < exploitingEpoch; ++i) {
        sarsa.train(std::pow(1 + 1. / exploitingEpoch, - 3 * i) * exploringEpsilon, learningRate, length);
        if (i % (exploitingEpoch / 10) == 0) {
            showPolicy(sarsa.getPolicies(), pg1);
            puts("");
        }
    }

}

void qLearningDemo(const std::string& fileName, size_t length = 1'000, double learningRate = .1, size_t exploringEpoch = 5'000, size_t exploitingEpoch = 5'000, double exploringEpsilon = .9, double discount = .5) {
    std::cout << "Running " << "qLearningDemo" << "\r\n";
    Playground pg1;
    if (std::ifstream is{fileName, std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    std::cout << "Exploring State" << "\r\n";
    QLearning qLearning(pg1, discount);
    showPolicy(qLearning.getPolicies(), pg1);
    puts("");
    for (int i = 0; i < exploringEpoch; ++i) {
        qLearning.explore(exploringEpsilon, learningRate, length);
        if (i % (exploringEpoch / 10) == 0) {
            showPolicy(qLearning.getPolicies(), pg1);
            puts("");
        }
    }
    std::cout << "Exploiting State" << "\r\n";
    for (int i = 0; i < exploitingEpoch; ++i) {
        qLearning.train(std::pow(1 + 1. / exploitingEpoch, - 3 * i) * exploringEpsilon, learningRate, length);
        if (i % (exploitingEpoch / 10) == 0) {
            showPolicy(qLearning.getPolicies(), pg1);
            puts("");
        }
    }

}

void dqnDemo(const std::string& fileName, size_t expPoolSize = 10'000, size_t batchCounts = 10, size_t batchLength = 500, double learningRate = .000'1, size_t exploringEpoch = 1'000, size_t exploitingEpoch = 1'000, double exploitingEpsilon = .9, double discount = .9) {
    std::cout << "Running " << "dqnDemo" << "\r\n";
    Playground pg1;
    if (std::ifstream is{fileName, std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    std::cout << "Exploring State" << "\r\n";
    DeepQNetwork dqn(pg1, discount);
    showPolicy(dqn.getTargetNetwork(), pg1);
    puts("");
    for (int i = 0; i < exploringEpoch; ++i) {
        dqn.expReplay(expPoolSize);
        for (size_t j = 0; j < batchCounts; ++j)
            dqn.train(learningRate, batchLength);
        if (i % (exploringEpoch / 10) == 0) {
            showPolicy(dqn.getTargetNetwork(), pg1);
            printQValues(dqn.getTargetNetwork(), pg1);
            puts("");
        }
    }
    std::cout << "Exploiting State" << "\r\n";
    for (int i = 0; i < exploitingEpoch; ++i) {
        dqn.expReplay(exploitingEpsilon, expPoolSize);
        for (size_t j = 0; j < batchCounts; ++j)
            dqn.train(learningRate, batchLength);
        if (i % (exploitingEpoch / 10) == 0) {
            showPolicy(dqn.getTargetNetwork(), pg1);
            printQValues(dqn.getTargetNetwork(), pg1);
            puts("");
        }
    }

}