#include <iostream>
#include <Windows.h>
#include "samples.hpp"
#include "neural/samples.hpp"

using namespace std::literals;

int main() {
    SetConsoleOutputCP(CP_UTF8);
    std::cout << "∧∨<>" << "\r\n";      ///debug
    // onlineQLearningDemo("./pg-samples/pg6.dat", 1'000);
    // qLearningDemo("./pg-samples/pg5.dat");
    dqnDemo("./pg-samples/pg3.dat");
    // onlineQLearningDemo();
}