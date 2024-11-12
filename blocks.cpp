#include "blocks.hpp"

AbstractBlock::AbstractBlock(uint16_t x, uint16_t y, const Playground& playground): x(x), y(y), flatten(playground.width * y + x), playground(playground) {};

std::pair<AbstractBlock *, double> AbstractBlock::stepOut(const ActionType& action) {
    if (inBound(action, playground.width, playground.height, x, y)) {
        auto movement = getMovement(action, playground.width);
        return {playground.map.at(y * playground.width + x + movement).get(), R_NORM};
    } else {
        return {this, R_BOUND};
    }
}

uint16_t AbstractBlock::getFlatten() const { return flatten; }

std::pair<AbstractBlock *, double> SwampBlock::stepOut(const ActionType& action) {
    return std::uniform_int_distribution(0, 1)(playground.gen)? AbstractBlock::stepOut(action): std::pair<AbstractBlock *, double>{this, 0}; 
}

std::pair<AbstractBlock *, double> IceBlock::stepOut(const ActionType& action) {
    if (std::uniform_int_distribution(0, 1)(playground.gen)) {
        if (inBound(action, playground.width, playground.height, x, y)) {
            auto [yMove, xMove] = getYXMovement(action, playground.width);
            auto movement = getMovement(action, playground.width);
            if (inBound(action, playground.width, playground.height, x + xMove, y + yMove)) {
                return {playground.map.at(y * playground.width + x + movement * 2).get(), R_NORM};
            } else {
                return {playground.map.at(y * playground.width + x + movement).get(), R_BOUND};
            }
        } else {
            return {this, R_BOUND};
        }
    } else {
        return AbstractBlock::stepOut(action);
    }
}

std::unique_ptr<AbstractBlock> buildWithBlockType(const BlockType& blockType, uint16_t x, uint16_t y, const Playground& playground) {
    switch (blockType) {
        case BlockType::ROAD:
            return std::make_unique<RoadBlock>(x, y, playground);
        case BlockType::SWAMP:
            return std::make_unique<SwampBlock>(x, y, playground);
        case BlockType::ICE:
            return std::make_unique<IceBlock>(x, y, playground);
        case BlockType::DESTINATION:
            return std::make_unique<DestinationBlock>(x, y, playground);
        case BlockType::FORBIDDEN:
            return std::make_unique<ForbiddenBlock>(x, y, playground);
        case BlockType::WALL:
            return std::make_unique<WallBlock>(x, y, playground);
        default:
            throw std::runtime_error{"Cannot build with the BlockType"};
    }
}
