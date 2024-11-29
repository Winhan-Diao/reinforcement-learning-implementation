#include <iostream>
#include <Windows.h>
#include "samples.hpp"
#include "neural/samples.hpp"

using namespace std::literals;

int main() {
    SetConsoleOutputCP(CP_UTF8);
    _controlfp_s(NULL, 0, _EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW);

    // std::cout << "∧∨<>" << "\r\n";      ///debug

    size_t length;
    size_t epoch;
    double qLearn;
    double policyLearn;
    double discount;

    std::cout << "Input: length, epoch, qLearn, policyLearn, discount" << "\r\n";
    // std::cin >> length >> epoch >> qLearn >> policyLearn >> discount;

    // onlineQLearningDemo("./pg-samples/pg6.dat", 1'000);
    // qLearningDemo("./pg-samples/pg5.dat");
    // tabularNetwork<QLearning>("./pg-samples/pg5.dat");
    // tabularNetwork<MonteCarlo>("./pg-samples/pg5.dat", 10'000, .01, 10'000, 10'000);

    // dqnDemo("./pg-samples/pg5.dat");

    // qacDemo("./pg-samples/pg5.dat", 0, 1'000, .000'1, .000'05, 10'000, .9);

    // a2cDemo("./pg-samples/pg5.dat", 0, 1'000, .000'1, .000'05, 10'000, .9);
    // a2cDemo("./pg-samples/pg5.dat", length, qLearn, policyLearn, epoch, discount);

    a2cV4Demo("./pg-samples/pg5.dat", 0);

    // dqnDemo("./pg-samples/pg3.dat");

    // ppoClipDemo("./pg-samples/pg5.dat", 0);

}