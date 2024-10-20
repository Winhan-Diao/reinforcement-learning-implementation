#include <iostream>
#include <fstream>
#include <Windows.h>
#include "playground.hpp"
#include "methods.hpp"

using namespace std::literals;

int main() {
    SetConsoleOutputCP(CP_UTF8);
    std::cout << "∧∨<>";
    Playground pg1;
    if (std::ifstream is{"./pg-samples/pg5.dat", std::ios::binary}) {
        is >> pg1;
    }
    std::cout << pg1;
    std::cout << "Exploring State" << "\r\n";
    EGreedyMonteCarlo eMC(pg1, 10'000, .5, .9, [](ssize_t n) -> double { return .000'001 * std::pow(1 + .000'01, -n + 1); });
    showPolicy(eMC.policies, pg1);
    puts("");
    for (int i = 0; i < 500'000; ++i) {
        eMC.run(.9);
        if (i % 5000 == 0) {
            showPolicy(eMC.policies, pg1);
            puts("");
        }
    }
    std::cout << "Exploiting State" << "\r\n";
    for (int i = 0, k = 0; i < 500'000; ++i, ++k) {
        eMC.run(std::pow(1 + .000'01, -k + 1) * 0.9);
        if (i % 5000 == 0) {
            showPolicy(eMC.policies, pg1);
            puts("");
        }
    }



}