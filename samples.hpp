#pragma once
#include <fstream>
#include "methods.hpp"

void mcDemo(const std::string& fileName, size_t episodeLength = 10'000, double learningRateFactor = .000'001, double baseCorrection = .000'01, size_t exploringEpisodeCounts = 500'000, size_t exploitingEpisodeCounts = 500'000, double exploringEpsilon = .9) {
    std::cout << "Running " << "mcDemo" << "\r\n";
    Playground pg1;
    if (std::ifstream is{fileName, std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    std::cout << "Exploring State" << "\r\n";
    EGreedyMonteCarlo eMC(pg1, episodeLength, .5, exploringEpsilon, [&](size_t& n) -> double { return learningRateFactor * std::pow(1 + baseCorrection, -static_cast<ssize_t>(n++)); });
    showPolicy(eMC.getPolicies(), pg1);
    puts("");
    for (int i = 0; i < exploringEpisodeCounts; ++i) {
        eMC.run(exploringEpsilon);
        if (i % (exploringEpisodeCounts / 10) == 0) {
            showPolicy(eMC.getPolicies(), pg1);
            puts("");
        }
    }
    std::cout << "Exploiting State" << "\r\n";
    for (int i = 0; i < exploitingEpisodeCounts; ++i) {
        eMC.run(std::pow(1 + 1. / exploitingEpisodeCounts, - 3 * i) * 0.9);
        if (i % (exploitingEpisodeCounts / 10) == 0) {
            showPolicy(eMC.getPolicies(), pg1);
            puts("");
        }
    }
}

void sarsaDemo(const std::string& fileName, size_t episodeLength = 1'000, double learningRateFactor = .000'1, double baseCorrection = .000'01, size_t exploringEpisodeCounts = 50'000, size_t exploitingEpisodeCounts = 50'000, double exploringEpsilon = .9) {
    std::cout << "Running " << "sarsaDemo" << "\r\n";
    Playground pg1;
    if (std::ifstream is{fileName, std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    std::cout << "Exploring State" << "\r\n";
    Sarsa sarsa(pg1, episodeLength, .5, exploringEpsilon, [&](size_t& n) -> double { return learningRateFactor * std::pow(1 + baseCorrection, -static_cast<ssize_t>(n++)); });
    showPolicy(sarsa.getPolicies(), pg1);
    puts("");
    for (int i = 0; i < exploringEpisodeCounts; ++i) {
        sarsa.run(exploringEpsilon);
        if (i % (exploringEpisodeCounts / 10) == 0) {
            showPolicy(sarsa.getPolicies(), pg1);
            puts("");
        }
    }
    std::cout << "Exploiting State" << "\r\n";
    for (int i = 0; i < exploitingEpisodeCounts; ++i) {
        sarsa.run(std::pow(1 + 1. / exploitingEpisodeCounts, - 3 * i) * exploringEpsilon);
        if (i % (exploitingEpisodeCounts / 10) == 0) {
            showPolicy(sarsa.getPolicies(), pg1);
            puts("");
        }
    }

}

void onlineQLearningDemo(const std::string& fileName, size_t episodeLength = 1'000, double learningRateFactor = .001, double baseCorrection = .000'1, size_t exploringEpisodeCounts = 50'000, size_t exploitingEpisodeCounts = 50'000, double exploringEpsilon = .99) {
    std::cout << "Running " << "onlineQLearningDemo" << "\r\n";
    Playground pg1;
    if (std::ifstream is{fileName, std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    std::cout << "Exploring State" << "\r\n";
    OnlineQLearning onlineQ(pg1, episodeLength, .5, exploringEpsilon, [&](size_t& n) -> double { return learningRateFactor * std::pow(1 + baseCorrection, -static_cast<ssize_t>(n++)); });
    showPolicy(onlineQ.getPolicies(), pg1);
    puts("");
    for (int i = 0; i < exploringEpisodeCounts; ++i) {
        onlineQ.run(exploringEpsilon);
        if (i % (exploringEpisodeCounts / 10) == 0) {
            showPolicy(onlineQ.getPolicies(), pg1);
            puts("");
        }
    }
    std::cout << "Exploiting State" << "\r\n";
    for (int i = 0; i < exploitingEpisodeCounts; ++i) {
        onlineQ.run(std::pow(1 + 1. / exploitingEpisodeCounts, - 3 * i) * exploringEpsilon);
        if (i % (exploitingEpisodeCounts / 10) == 0) {
            showPolicy(onlineQ.getPolicies(), pg1);
            puts("");
        }
    }

}

void offlineQLearningDemo(const std::string& fileName, size_t episodeLength = 1'000, double learningRateFactor = .001, double baseCorrection = .000'1, size_t exploringEpisodeCounts = 50'000, size_t exploitingEpisodeCounts = 50'000, double exploringEpsilon = .99) {
    std::cout << "Running " << "offlineQLearningDemo" << "\r\n";
    Playground pg1;
    if (std::ifstream is{fileName, std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    std::cout << "Exploring State" << "\r\n";
    OfflineQLearning offlineQ(pg1, episodeLength, .5, exploringEpsilon, [&](size_t& n) -> double { return learningRateFactor * std::pow(1 + baseCorrection, -static_cast<ssize_t>(n++)); });
    showPolicy(offlineQ.getPolicies(), pg1);
    puts("");
    for (int i = 0; i < exploringEpisodeCounts; ++i) {
        offlineQ.run(exploringEpsilon);
        if (i % (exploringEpisodeCounts / 10) == 0) {
            showPolicy(offlineQ.getPolicies(), pg1);
            puts("");
        }
    }
    std::cout << "Exploiting State" << "\r\n";
    for (int i = 0; i < exploitingEpisodeCounts; ++i) {
        offlineQ.run(std::pow(1 + 1. / exploitingEpisodeCounts, - 3 * i) * exploringEpsilon);
        if (i % (exploitingEpisodeCounts / 10) == 0) {
            showPolicy(offlineQ.getPolicies(), pg1);
            puts("");
        }
    }

}
