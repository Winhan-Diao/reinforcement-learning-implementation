#pragma once
#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <utility>
class AbstractBlock;
enum class ActionType: unsigned char;
#include "blocks.hpp"
#include "stream_utils.hpp"
using namespace std::literals;

class Playground {
public:
    uint16_t height;
    uint16_t width;
    std::vector<std::unique_ptr<AbstractBlock>> map;
    mutable std::mt19937 gen = std::mt19937(std::random_device()());

    Playground() = default;
    // i.toState; ii.rewards
    std::pair<AbstractBlock *, double> act(AbstractBlock *fromState, const ActionType& action) const;
    AbstractBlock *sample() const {
        return map.at(std::uniform_int_distribution(0, height * width - 1)(gen)).get();
    }
    friend std::ostream& operator<<(std::ostream& os, const Playground& p);
    friend std::istream& operator>>(std::istream& is, Playground& p);
};