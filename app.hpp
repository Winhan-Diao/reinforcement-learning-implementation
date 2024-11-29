#pragma once
#include <wx/wx.h>
#include <wx/spinctrl.h>
#include <vector>
#include <tuple>

/* 
learning rate;
discount;
starting point

exploring epoch length
exploring epsilon
exploiting epoch length
episode length
 */

using namespace std::literals;

enum Methods: unsigned char {
    MC,
    SARSA,
    TD,
    QL,
    DQN,
    _MAX,
};

static const std::vector<std::tuple<std::string>> METHODS_INFO = {{"Monte Carlo"s}, {"Sarsa"s}, {"Temporal Difference"s}, {"Q Learning"s}, {"Deep Q-Network"s}};

class MainPanel: public wxPanel {
public:
    MainPanel(wxWindow *parent);
};

class CustomFrame: public wxFrame {
public:
    CustomFrame();
};

class CustomApp: public wxApp {
    CustomFrame *frame;
public:
    CustomApp();
    bool OnInit() override;
    int OnExit() override;
};


wxDECLARE_APP(CustomApp);