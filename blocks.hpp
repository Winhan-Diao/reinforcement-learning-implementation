#pragma once
#include <utility>
#include <memory>
#include <random>
#include <unordered_map>
class Playground;
#include "playground.hpp"
#define R_DEST 1.0
#define R_NORM 0.0
#define R_BOUND -1.0
#define R_FORBID -1.0
#define R_WALL 0.0

enum class BlockType {
    INVALID,
    ROAD,
    SWAMP,
    ICE,
    DESTINATION,
    FORBIDDEN,
    WALL,
};

enum class ActionType: unsigned char {
    UP,
    RIGHT,
    DOWN,
    LEFT,
    NONE,
};
inline int32_t getMovement(const ActionType& action, const uint16_t& width) {
    switch (action) {
        case ActionType::UP:
            return -width;
        case ActionType::RIGHT:
            return 1;
        case ActionType::DOWN:
            return width;
        case ActionType::LEFT:
            return -1;
        case ActionType::NONE:
        default:
            return 0;
    }
}
// i.y; ii.x
inline std::pair<int32_t, int32_t> getYXMovement(const ActionType& action, const uint16_t& width) {
    switch (action) {
        case ActionType::UP:
            return {-1, 0};
        case ActionType::RIGHT:
            return {0, 1};
        case ActionType::DOWN:
            return {1, 0};
        case ActionType::LEFT:
            return {0, -1};
        case ActionType::NONE:
        default:
            return {0, 0};
    }
}
inline bool inBound(const ActionType& action, const uint16_t& width, const uint16_t& height, const uint16_t& x, const uint16_t& y) {
    switch (action) {
        case ActionType::UP:
            return y > 0;
        case ActionType::RIGHT:
            return x < width - 1;
        case ActionType::DOWN:
            return y < height - 1;
        case ActionType::LEFT:
            return x > 0;
        case ActionType::NONE:
        default:
            return true;
    }
}

class AbstractBlock {
protected:
    uint16_t x;
    uint16_t y;
    uint16_t flatten;
    const Playground& playground;
public:
    AbstractBlock() = default;
    AbstractBlock(uint16_t x, uint16_t y, const Playground& playground);
    // i.isValid; ii.reward 
    virtual std::pair<bool, double> stepIn() = 0;
    // i.nextBlock; ii.reward
    virtual std::pair<AbstractBlock *, double> stepOut(const ActionType& action);
    virtual char getSymbol() = 0;
    uint16_t getX() const { return x; }
    uint16_t getY() const { return y; }
    uint16_t getFlatten() const;
    virtual ~AbstractBlock() noexcept {
        std::cout << y << x << " delete once" << "\r\n";
    }
    friend Playground;
};

class RoadBlock: public AbstractBlock {
public:
    RoadBlock() = default;
    RoadBlock(uint16_t x, uint16_t y, const Playground& playground): AbstractBlock(x, y, playground) {}
    std::pair<bool, double> stepIn() override {
        return {true, R_NORM};
    }
    char getSymbol() override { return '.'; }
};

class SwampBlock: public AbstractBlock {
public:
    SwampBlock() = default;
    SwampBlock(uint16_t x, uint16_t y, const Playground& playground): AbstractBlock(x, y, playground) {}
    std::pair<bool, double> stepIn() override {
        return {true, R_NORM};
    }
    std::pair<AbstractBlock *, double> stepOut(const ActionType& action) override;
    char getSymbol() override { return '*'; }
};

class IceBlock: public AbstractBlock {
public:
    IceBlock() = default;
    IceBlock(uint16_t x, uint16_t y, const Playground& playground): AbstractBlock(x, y, playground) {}
    std::pair<bool, double> stepIn() override {
        return {true, R_NORM};
    }
    std::pair<AbstractBlock *, double> stepOut(const ActionType& action) override;
    char getSymbol() override { return '/'; }
};

class DestinationBlock: public AbstractBlock {
public:
    DestinationBlock() = default;
    DestinationBlock(uint16_t x, uint16_t y, const Playground& playground): AbstractBlock(x, y, playground) {}
    std::pair<bool, double> stepIn() override {
        return {true, R_DEST};
    }
    char getSymbol() override { return 'O'; }
};

class ForbiddenBlock: public AbstractBlock {
public:
    ForbiddenBlock() = default;
    ForbiddenBlock(uint16_t x, uint16_t y, const Playground& playground): AbstractBlock(x, y, playground) {}
    std::pair<bool, double> stepIn() override {
        return {true, R_FORBID};
    }
    char getSymbol() override { return 'X'; }
};

class WallBlock: public AbstractBlock {
public:
    WallBlock() = default;
    WallBlock(uint16_t x, uint16_t y, const Playground& playground): AbstractBlock(x, y, playground) {}
    std::pair<bool, double> stepIn() override {
        return {false, R_WALL};
    }
    char getSymbol() override { return '#'; }
};

std::unique_ptr<AbstractBlock> buildWithBlockType(const BlockType& blockType, uint16_t x, uint16_t y, const Playground& playground);

static std::unordered_map<char, BlockType> BLOCKTYPE_WITH_SYMBOL{{'.', BlockType::ROAD}, {'*', BlockType::SWAMP}, {'/', BlockType::ICE}, {'O', BlockType::DESTINATION}, {'X', BlockType::FORBIDDEN}, {'#', BlockType::WALL}};