#pragma once
#include "network.hpp"

template <class T1, class T2, class _BiPred>
void train(Network& n, const std::valarray<std::valarray<double>>& trainInputs, const std::valarray<T1>& trainOutputs, double learningRate, size_t epoch, size_t batchSize, const std::valarray<std::valarray<double>>& testInputs, const std::valarray<T2>& testOutputs, _BiPred&& testBiPred, size_t threadCounts = 1) {
    assert(trainInputs.size() == trainOutputs.size());       //assertion
    for (size_t e = 0; e < epoch; ++e) {
        std::cout << "epoch " << e << "\r\n";
        std::vector<size_t> indices = generateShuffledIndices(trainInputs.size());
        auto shuffledInputs = reorder(trainInputs, indices);
        auto shuffledOutputs = reorder(trainOutputs, indices);
        auto batchedInputs = batch(shuffledInputs, batchSize);
        auto batchedOutputs = batch(shuffledOutputs, batchSize);
        size_t p = 0;
        std::cout << "training";
        for (size_t b = 0; b < batchedInputs.size(); ++b) {
            n.batchedTrain(batchedInputs[b], batchedOutputs[b], learningRate * batchSize, threadCounts);
            if (b * batchSize > p) {
                std::cout << '.';
                p += indices.size() / 20;
            }
        }
        std::cout << "\r\n" << "all batched data is trained" << "\r\n";
        std::cout << "assessing accuracy: " << n.test(testInputs, testOutputs, testBiPred) << "\r\n";
    }
}