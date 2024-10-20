#include "playground.hpp"

std::pair<AbstractBlock *, double> Playground::act(AbstractBlock *fromState, const ActionType& action) const {
    auto [toState, outReward] = fromState->stepOut(action);
    auto [valid, inReward] = toState->stepIn();
    return {valid? toState: fromState, outReward + inReward};
}

std::ostream& operator<<(std::ostream& os, const Playground& p) {
    os << "<playground>" << "\r\n";

    os << "<height>" << "\r\n";
    os << p.height << "\r\n";
    os << "</height>" << "\r\n";

    os << "<width>" << "\r\n";
    os << p.width << "\r\n";
    os << "</width>" << "\r\n";

    os << "<map>" << "\r\n";
    for (uint16_t h = 0; h < p.height; ++h) {
        for (uint16_t w = 0; w < p.width; ++w) {
            os << p.map.at(h * p.width + w)->getSymbol() << ' ';
        }
        os << "\r\n";
    }
    os << "</map>" << "\r\n";
    
    os << "</playground>" << "\r\n";
    return os;
}

std::istream& operator>>(std::istream& is, Playground& p) {
    skipTill(is, "<playground>"s);
    skipTill(is, "<height>"s);
    is >> p.height;
    skipTill(is, "<width>"s);
    is >> p.width;
    p.map = std::vector<std::unique_ptr<AbstractBlock>>(std::size_t(p.height * p.width));
    skipTill(is, "<map>"s);
    char tmp;
    for (uint16_t h = 0; h < p.height; ++h) {
        for (uint16_t w = 0; w < p.width; ++w) {
            is >> std::ws;
            tmp = is.get();
            p.map.at(h * p.width + w) = buildWithBlockType(BLOCKTYPE_WITH_SYMBOL[tmp], w, h, p);
        }
    }

    skipTill(is, "</playground>"s);
    return is;
}