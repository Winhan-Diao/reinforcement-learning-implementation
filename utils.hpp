#pragma once
#include <valarray>
#include <vector>

template <class _IterIn>
inline void toSumOne(_IterIn&& begin, _IterIn&& end) {
    auto sum = std::accumulate(begin, end, 0.0);
    if (sum)
        std::for_each(begin, end, [sum](auto& x) { x /= sum; });
    else {
        ssize_t counts = std::distance(begin, end);
        std::fill(begin, end, std::pow(counts, -1));
    }
}

inline double expectation(const std::valarray<double>& policies, const std::valarray<double>& qValues) {
    return (policies * qValues).sum();
}