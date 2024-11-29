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
        qLearning.train((- double(i) / exploitingEpoch + 1) * exploringEpsilon, learningRate, length);
        if (i % (exploitingEpoch / 10) == 0) {
            showPolicy(qLearning.getPolicies(), pg1);
            puts("");
        }
    }

}

template <class T>
void tabularNetwork(const std::string& fileName, size_t length = 1'000, double learningRate = .1, size_t exploringEpoch = 5'000, size_t exploitingEpoch = 5'000, double exploringEpsilon = .9, double discount = .5) {
    std::cout << "Running " << "tabularNetwork" << "\r\n";
    Playground pg1;
    if (std::ifstream is{fileName, std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    std::cout << "Exploring State" << "\r\n";
    T method(pg1, discount);
    showPolicy(method.getPolicies(), pg1);
    puts("");
    for (int i = 0; i < exploringEpoch; ++i) {
        method.train(exploringEpsilon, learningRate, length);
        if (i % (exploringEpoch / 10) == 0) {
            showPolicy(method.getPolicies(), pg1);
            puts("");
        }
    }
    std::cout << "Exploiting State" << "\r\n";
    for (int i = 0; i < exploitingEpoch; ++i) {
        method.train((- double(i) / exploitingEpoch + 1) * exploringEpsilon, learningRate, length);
        if (i % (exploitingEpoch / 10) == 0) {
            showPolicy(method.getPolicies(), pg1);
            puts("");
        }
    }

}

void dqnDemo(const std::string& fileName, size_t expPoolSize = 10'000, size_t batchCounts = 1, size_t batchLength = 5'000, double learningRate = .000'1, size_t exploringEpoch = 1'000, size_t exploitingEpoch = 1'000, double exploitingEpsilon = .9, double discount = .9) {
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

void REINFORCEDemo(const std::string& fileName, size_t length = 10'000, double learningRate = .000'001, size_t epoch = 1'000, double discount = .9) {
    std::cout << "Running " << "REINFORCEDemo" << "\r\n";
    Playground pg1;
    if (std::ifstream is{fileName, std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    REINFORCE rf(pg1, discount);
    showPolicy(rf.getPolicyNetwork(), pg1);
    printQValues(rf.getPolicyNetwork(), pg1);
    puts("");
    // std::cout << "Exploring State" << "\r\n";
    for (int i = 0; i < epoch; ++i) {
        rf.train(learningRate, length);
        if (i % (epoch / 100) == 0) {
            showPolicy(rf.getPolicyNetwork(), pg1);
            printQValues(rf.getPolicyNetwork(), pg1);
            puts("");
        }
    }

}

void qacDemo(const std::string& fileName, ssize_t startIndex = -1, size_t length = 1'000, double qLearningRate = .000'1, double policyLearningRate = .000'1, size_t exploringEpoch = 10'000, double discount = .9) {
    std::cout << "Running " << "qacDemo" << "\r\n";
    Playground pg1;
    if (std::ifstream is{fileName, std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    QActorCritic qac(pg1, discount, startIndex);
    showPolicy(qac.getPolicyNetwork(), pg1);
    // printQValues(qac.getQNetwork(), pg1);
    // puts("");
    printQValues(qac.getPolicyNetwork(), pg1);
    puts("");
    std::cout << "Exploring State" << "\r\n";
    for (int i = 0; i < exploringEpoch; ++i) {
        qac.train(qLearningRate, policyLearningRate, length);
        if (i % (exploringEpoch / 100) == 0) {
            showPolicy(qac.getPolicyNetwork(), pg1);
            // printQValues(qac.getQNetwork(), pg1);
            // puts("");
            printQValues(qac.getPolicyNetwork(), pg1);
            puts("");
        }
    }
}

void a2cDemo(const std::string& fileName, ssize_t startIndex = -1, size_t length = 1'000, double vLearningRate = .000'1, double policyLearningRate = .000'01, size_t epoch = 10'000, double discount = .9) {
    std::cout << "Running " << "a2cDemo" << "\r\n";
    Playground pg1;
    if (std::ifstream is{fileName, std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    AdvantageActorCritic a2c(pg1, discount, startIndex);
    showPolicy(a2c.getPolicyNetwork(), pg1);
    printStateValues(a2c.getVNetwork(), pg1);
    puts("");
    printScaledValues(a2c.getPolicyNetwork(), pg1);
    puts("");
    std::cout << "Exploring State" << "\r\n";
    for (int i = 0; i < epoch; ++i) {
        a2c.train(vLearningRate, policyLearningRate, length);
        if (i % (epoch / 100) == 0) {
            showPolicy(a2c.getPolicyNetwork(), pg1);
            printStateValues(a2c.getVNetwork(), pg1);
            puts("");
            printScaledValues(a2c.getPolicyNetwork(), pg1);
            puts("");
        }
    }
}

void a2cV3Demo(const std::string& fileName, ssize_t startIndex = -1, size_t trajectoryLength = 1'000, size_t trajectoryCounts = 10, double vLearningRate = .000'01, double policyLearningRate = .000'01, size_t epoch = 10'000, double discount = .9) {
    std::cout << "Running " << "a2cDemo" << "\r\n";
    Playground pg1;
    if (std::ifstream is{fileName, std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    AdvantageActorCriticV3 a2c(pg1, discount, startIndex);
    showPolicy(a2c.getPolicyNetwork(), pg1);
    printStateValues(a2c.getVNetwork(), pg1);
    puts("");
    printScaledValues(a2c.getPolicyNetwork(), pg1);
    puts("");
    std::cout << "Exploring State" << "\r\n";
    for (int i = 0; i < epoch; ++i) {
        a2c.generateTrajectories(trajectoryLength, trajectoryCounts);
        a2c.train(vLearningRate, policyLearningRate);
        if (i % (epoch / 100) == 0) {
            showPolicy(a2c.getPolicyNetwork(), pg1);
            printStateValues(a2c.getVNetwork(), pg1);
            puts("");
            printScaledValues(a2c.getPolicyNetwork(), pg1);
            puts("");
        }
    }
}

void a2cV4Demo(const std::string& fileName, ssize_t startIndex = -1, size_t trajectoryLength = 500, size_t trajectoryCounts = 10, double vLearningRate = .000'1, double policyLearningRate = .000'1, size_t epoch = 10'000, double discount = .9) {
    std::cout << "Running " << "a2cDemo" << "\r\n";
    Playground pg1;
    if (std::ifstream is{fileName, std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    AdvantageActorCriticV4 a2c(pg1, discount, startIndex);
    showPolicy(a2c.getPolicyNetwork(), pg1);
    printStateValues(a2c.getVNetwork(), pg1);
    puts("");
    printScaledValues(a2c.getPolicyNetwork(), pg1);
    puts("");
    std::cout << "Exploring State" << "\r\n";
    for (int i = 0; i < epoch; ++i) {
        a2c.generateTrajectories(trajectoryLength, trajectoryCounts);
        a2c.train(vLearningRate, policyLearningRate);
        if (i % (epoch / 100) == 0) {
            showPolicy(a2c.getPolicyNetwork(), pg1);
            printStateValues(a2c.getVNetwork(), pg1);
            puts("");
            printScaledValues(a2c.getPolicyNetwork(), pg1);
            puts("");
        }
    }
}

void ppoClipDemo(const std::string& fileName, ssize_t startIndex = -1, size_t length = 1'000, double vLearningRate = .000'1, double policyLearningRate = .000'01, size_t exploringEpoch = 10'000, double epsilon = .2, double discount = .9) {
    std::cout << "Running " << "ppoClipDemo" << "\r\n";
    Playground pg1;
    if (std::ifstream is{fileName, std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    PPOClip ppoc(pg1, discount, startIndex);
    showPolicy(ppoc.getPolicyNetwork(), pg1);
    // printStateValues(ppoc.getVNetwork(), pg1);
    // puts("");
    printScaledValues(ppoc.getPolicyNetwork(), pg1);
    puts("");
    std::cout << "Exploring State" << "\r\n";
    for (int i = 0; i < exploringEpoch; ++i) {
        ppoc.generateExperience(length);
        ppoc.train(vLearningRate, policyLearningRate, epsilon);
        if (i % (exploringEpoch / 100) == 0) {
            showPolicy(ppoc.getPolicyNetwork(), pg1);
            // printStateValues(ppoc.getVNetwork(), pg1);
            // puts("");
            printScaledValues(ppoc.getPolicyNetwork(), pg1);
            puts("");
        }
    }
}
