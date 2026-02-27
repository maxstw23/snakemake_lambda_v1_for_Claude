#include <fstream>
#include <iostream>
#include <TFile.h>
#include <TProfile.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TLine.h>
#include <TMath.h>

void FitSlope(const char *outFile, const char *outPlot, int order)
{

    const float y_up = 0.8;
    const float y_lo = -0.8;
    TF1 *fun;
    if (order == 1) fun = new TF1("fun", "[0]*x", -1, 1);
    if (order == 3) fun = new TF1("fun", "[0]*x+[1]*x*x*x", -1, 1);
    char name[200];
    float v1_p[9], v1_p_err[9], v1_n[9], v1_n_err[9];
    float d3v1dy3_p[9], d3v1dy3_p_err[9], d3v1dy3_n[9], d3v1dy3_n_err[9];
    float a1_p[9], a1_p_err[9], a1_n[9], a1_n_err[9];
    TCanvas *c = new TCanvas("c", "c", 1200, 900);
    c->Divide(4, 3);
    for (int i = 1; i <= 9; i++) // change to 12 to include merged centralities
    {
        c->cd(i);
        if (i==10) snprintf(name, 200, "cen%d.v1_pion.root", 13);
        else if (i==11) snprintf(name, 200, "cen%d.v1_pion.root", 57);
        else if (i==12) snprintf(name, 200, "cen%d.v1_pion.root", 89);
        else snprintf(name, 200, "cen%d.v1_pion.root", i);
        TFile *f = new TFile(name);
        TProfile *Flow_vEta_f_Selp = (TProfile *)f->Get("Flow_vEta_f_Selp");
        Flow_vEta_f_Selp->Scale(0.01);
        Flow_vEta_f_Selp->Fit("fun", "0", "", y_lo, y_up);
        v1_p[i - 1] = fun->GetParameter(0);
        v1_p_err[i - 1] = fun->GetParError(0);
        d3v1dy3_p[i - 1] = fun->GetParameter(1);
        d3v1dy3_p_err[i - 1] = fun->GetParError(1);

        TProfile *Flow_vEta_f_Seln = (TProfile *)f->Get("Flow_vEta_f_Seln");
        Flow_vEta_f_Seln->Scale(0.01);
        Flow_vEta_f_Seln->Fit("fun", "0", "", y_lo, y_up);
        v1_n[i - 1] = fun->GetParameter(0);
        v1_n_err[i - 1] = fun->GetParError(0);
        d3v1dy3_n[i - 1] = fun->GetParameter(1);
        d3v1dy3_n_err[i - 1] = fun->GetParError(1);
 
        // if no fit, continue (perhaps the profile is empty)
        if (Flow_vEta_f_Selp->GetFunction("fun") == nullptr || Flow_vEta_f_Seln->GetFunction("fun") == nullptr)
            continue;
        Flow_vEta_f_Selp->GetFunction("fun")->SetLineColor(kRed);
        Flow_vEta_f_Selp->GetFunction("fun")->ResetBit(TF1::kNotDraw);
        Flow_vEta_f_Seln->GetFunction("fun")->SetLineColor(kBlue);
        Flow_vEta_f_Seln->GetFunction("fun")->ResetBit(TF1::kNotDraw);
        Flow_vEta_f_Selp->SetMarkerStyle(22); 
        Flow_vEta_f_Selp->SetMarkerColor(kRed);
        Flow_vEta_f_Seln->SetMarkerStyle(22);
        Flow_vEta_f_Seln->SetMarkerColor(kBlue);
        Flow_vEta_f_Selp->GetYaxis()->SetRangeUser(-0.05, 0.05);
        Flow_vEta_f_Selp->Draw();
        Flow_vEta_f_Seln->Draw("SAME");
        TLine *line = new TLine(-1, 0, 1, 0);
        line->SetLineColor(kBlack);
        line->Draw();

        if (i==10) snprintf(name, 200, "cen%d.a1_pion.root", 13);
        else if (i==11) snprintf(name, 200, "cen%d.a1_pion.root", 57);
        else if (i==12) snprintf(name, 200, "cen%d.a1_pion.root", 89);
        else snprintf(name, 200, "cen%d.a1_pion.root", i);
        f = new TFile(name);
        Flow_vEta_f_Selp = (TProfile *)f->Get("Flow_vEta_f_Selp");
        Flow_vEta_f_Selp->Scale(0.01);
        Flow_vEta_f_Selp->Fit("fun", "0", "", y_lo, y_up);
        a1_p[i - 1] = fun->GetParameter(0);
        a1_p_err[i - 1] = fun->GetParError(0);
        Flow_vEta_f_Seln = (TProfile *)f->Get("Flow_vEta_f_Seln");
        Flow_vEta_f_Seln->Scale(0.01);
        Flow_vEta_f_Seln->Fit("fun", "0", "", y_lo, y_up);
        a1_n[i - 1] = fun->GetParameter(0);
        a1_n_err[i - 1] = fun->GetParError(0);
    }

    // Save plot
    c->SaveAs(outPlot);

    float cen[12] = {75, 65, 55, 45, 35, 25, 15, 7.5, 2.5}; //, 13, 57, 89};
    float err_cen[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0}; //, 0, 0, 0};

    //==============Deltav1 calculation=============//
    float delta_v1[9], delta_v1_err[9], delta_a1[9], delta_a1_err[9]; // change to 12 to include merged centralities

    for (int i = 0; i < 9; i++) // change to 12 to include merged centralities
    {
        delta_v1[i] = (v1_p[i] - v1_n[i]);
        delta_v1_err[i] = (sqrt(v1_p_err[i] * v1_p_err[i] + v1_n_err[i] * v1_n_err[i]));

        delta_a1[i] = (a1_p[i] - a1_n[i]);
        delta_a1_err[i] = (sqrt(a1_p_err[i] * a1_p_err[i] + a1_n_err[i] * a1_n_err[i]));
    }
    
    //==============Write to csv file=============//
    std::ofstream out(outFile);
    out << "cen,err_cen,v1_p,v1_p_err,v1_n,v1_n_err,d3v1dy3_p,d3v1dy3_p_err,d3v1dy3_n,d3v1dy3_n_err,delta_v1,delta_v1_err,a1_p,a1_p_err,a1_n,a1_n_err,delta_a1,delta_a1_err\n";
    for (int i = 0; i < 9; i++) // change to 12 to include merged centralities
    {
        out << cen[i] << ",";
        out << err_cen[i] << ",";
        out << v1_p[i] << ",";
        out << v1_p_err[i] << ",";
        out << v1_n[i] << ",";
        out << v1_n_err[i] << ",";
        out << d3v1dy3_p[i] << ",";
        out << d3v1dy3_p_err[i] << ",";
        out << d3v1dy3_n[i] << ",";
        out << d3v1dy3_n_err[i] << ",";
        out << delta_v1[i] << ",";
        out << delta_v1_err[i] << ",";
        out << a1_p[i] << ",";
        out << a1_p_err[i] << ",";
        out << a1_n[i] << ",";
        out << a1_n_err[i] << ",";
        out << delta_a1[i] << ",";
        out << delta_a1_err[i] << endl;
    }
    out.close();
}
