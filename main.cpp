#include <iostream>
#include <Windows.h>
#include "samples.hpp"

using namespace std::literals;

int main() {
    SetConsoleOutputCP(CP_UTF8);
    std::cout << "∧∨<>" << "\r\n";      ///debug
    // onlineQLearningDemo("./pg-samples/pg6.dat", 1'000);
    offlineQLearningDemo("./pg-samples/pg6.dat");
    // onlineQLearningDemo();
}