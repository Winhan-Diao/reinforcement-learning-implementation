#include "app.hpp"

MainPanel::MainPanel(wxWindow *parent): wxPanel(parent, wxID_ANY, wxPoint(100, 100)) {
    wxSizer *mainSizer = new wxBoxSizer(wxHORIZONTAL);
        wxSizer *settingsSizer = new wxBoxSizer(wxVERTICAL);
            wxChoice *methodsChoice = new wxChoice(this, wxID_ANY);
            for (unsigned char i = 0; i < Methods::_MAX; ++i) {
                methodsChoice->Append(wxString(std::get<0>(METHODS_INFO[i])));
            }
            settingsSizer->Add(methodsChoice, 0, wxALL, 5);

            wxStaticBox *configsStaticBox = new wxStaticBox(this, wxID_ANY, wxT("This is a test..."));
            wxStaticBoxSizer *staticBoxSizer = new wxStaticBoxSizer(configsStaticBox, wxVERTICAL);
                wxSpinCtrl *samplesSpinCtrl = new wxSpinCtrl(configsStaticBox, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 7, 1);
                staticBoxSizer->Add(samplesSpinCtrl);
            settingsSizer->Add(staticBoxSizer, 0, wxALL, 5);
        mainSizer->Add(settingsSizer, 1);
        
        wxPanel *mapPanel = new wxPanel(this);
        mapPanel->SetBackgroundColour(wxColour(0xaabbcc));
        mainSizer->Add(mapPanel, 1, wxEXPAND | wxALL, 0);
    SetSizerAndFit(mainSizer);
}

CustomFrame::CustomFrame(): wxFrame(nullptr, wxID_ANY, "Hello!") {
    MainPanel *mainPanel = new MainPanel(this);
}

CustomApp::CustomApp(): wxApp(), frame(new CustomFrame()) {}

bool CustomApp::OnInit() {
    // SetProcessDPIAware();
    frame->Show(true);
    return true;
}

int CustomApp::OnExit() {
    return wxApp::OnExit();
}

wxIMPLEMENT_APP(CustomApp);