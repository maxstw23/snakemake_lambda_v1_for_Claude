#include <fstream>
#include <iostream>
#include <TFile.h>
#include <TProfile.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TLine.h>
#include <TMath.h>

#define npar 4
#define ncent 9

void read_model(const char* input, const char* output) // 1 for AMPT, 2 for UrQMD
{
    std::string input_str = input;
    // input_str format is data/model/{model}/{energy}.root, need to extract {model}
    std::string model = input_str.substr(input_str.find("model/")+6, input_str.rfind("/")-input_str.find("model/")-6);
    float centralities[ncent] = {75.0, 65.0, 55.0, 45.0, 35.0, 25.0, 15.0, 7.5, 2.5};
    float dv1dy_pos[npar][ncent] = {0.0};
    float dv1dy_pos_err[npar][ncent] = {0.0};
    float d3v1dy3_pos[npar][ncent] = {0.0};
    float d3v1dy3_pos_err[npar][ncent] = {0.0};
    float dv1dy_neg[npar][ncent] = {0.0};
    float dv1dy_neg_err[npar][ncent] = {0.0};
    float d3v1dy3_neg[npar][ncent] = {0.0};
    float d3v1dy3_neg_err[npar][ncent] = {0.0};
    TFile *f = new TFile(input,"READ");

    TString particle_urqmd[npar] = {"lambda", "proton", "kaon", "pion"};
    TString particle_ampt[npar] = {"lambda", "proton", "kplus", "piplus"};
    TString antiparticle_ampt[npar] = {"antilambda", "antiproton", "kminus", "piminus"};
    for (int par = 0; par < npar; par++)
    {
        for (int i = 0; i < ncent; i++)
        {   
            TGraphErrors* gpos, *gneg;
            TProfile* gpos_prof, *gneg_prof;
            if (!strcmp(model.c_str(), "ampt"))
            {
                gpos_prof = (TProfile*)f->Get(Form("h%s_v1_y_%d", particle_ampt[par].Data(), i+1));
                gneg_prof = (TProfile*)f->Get(Form("h%s_v1_y_%d", antiparticle_ampt[par].Data(), i+1));
            }
            else if (!strcmp(model.c_str(), "urqmd"))
            {
                gpos = (TGraphErrors*)f->Get(Form("hv1_%sPlus_cent%d", particle_urqmd[par].Data(), i));
                gneg = (TGraphErrors*)f->Get(Form("hv1_%sMinus_cent%d", particle_urqmd[par].Data(), i));
            }
            else
            {
                std::cout << "Error: unknown model" << std::endl;
                return;
            }

            // fit with third order poly
            TF1 *func = new TF1("func", "[0]*x + [1]*x*x*x", -0.8, 0.8);
            if (!strcmp(model.c_str(), "ampt")) gpos_prof->Fit("func", "R");
            else if (!strcmp(model.c_str(), "urqmd")) gpos->Fit("func", "R");
            dv1dy_pos[par][i] = func->GetParameter(0);
            dv1dy_pos_err[par][i] = func->GetParError(0);
            d3v1dy3_pos[par][i] = func->GetParameter(1);
            d3v1dy3_pos_err[par][i] = func->GetParError(1);

            if (!strcmp(model.c_str(), "ampt")) gneg_prof->Fit("func", "R");
            else if (!strcmp(model.c_str(), "urqmd")) gneg->Fit("func", "R");
            dv1dy_neg[par][i] = func->GetParameter(0);
            dv1dy_neg_err[par][i] = func->GetParError(0);
            d3v1dy3_neg[par][i] = func->GetParameter(1);
            d3v1dy3_neg_err[par][i] = func->GetParError(1);
        }
    }
    f->Close();

    TString out_tstr = output;
    TString out_lambda = out_tstr.Copy();
    // out_lambda.Replace(out_lambda.Length()-4, 4, "_lambda.csv");
    std::ofstream outs_lambda(out_lambda);
    outs_lambda << "centralities,dv1dy_lambda,dv1dy_lambda_err,d3v1dy3_lambda,d3v1dy3_lambda_err,dv1dy_lambdabar,dv1dy_lambdabar_err,d3v1dy3_lambdabar,d3v1dy3_lambdabar_err\n";
    for (int i = 0; i < 9; i++)
    {
        outs_lambda << centralities[i] << ",";
        outs_lambda << dv1dy_pos[0][i] << ",";
        outs_lambda << dv1dy_pos_err[0][i] << ",";
        outs_lambda << d3v1dy3_pos[0][i] << ",";
        outs_lambda << d3v1dy3_pos_err[0][i] << ",";
        outs_lambda << dv1dy_neg[0][i] << ",";
        outs_lambda << dv1dy_neg_err[0][i] << ",";
        outs_lambda << d3v1dy3_neg[0][i] << ",";
        outs_lambda << d3v1dy3_neg_err[0][i] << std::endl;
    }
    outs_lambda.close();

    TString out_proton = out_tstr.Copy();
    out_proton.Replace(out_proton.Length()-11, 11, "_proton.csv");
    std::ofstream outs_proton(out_proton);
    outs_proton << "centralities,dv1dy_proton,dv1dy_proton_err,d3v1dy3_proton,d3v1dy3_proton_err,dv1dy_antiproton,dv1dy_antiproton_err,d3v1dy3_antiproton,d3v1dy3_antiproton_err\n";
    for (int i = 0; i < 9; i++)
    {
        outs_proton << centralities[i] << ",";
        outs_proton << dv1dy_pos[1][i] << ",";
        outs_proton << dv1dy_pos_err[1][i] << ",";
        outs_proton << d3v1dy3_pos[1][i] << ",";
        outs_proton << d3v1dy3_pos_err[1][i] << ",";
        outs_proton << dv1dy_neg[1][i] << ",";
        outs_proton << dv1dy_neg_err[1][i] << ",";
        outs_proton << d3v1dy3_neg[1][i] << ",";
        outs_proton << d3v1dy3_neg_err[1][i] << std::endl;
    }
    outs_proton.close();

    TString out_kaon = out_tstr.Copy();
    out_kaon.Replace(out_kaon.Length()-11, 11, "_kaon.csv");
    std::ofstream outs_kaon(out_kaon);
    outs_kaon << "centralities,dv1dy_kplus,dv1dy_kplus_err,d3v1dy3_kplus,d3v1dy3_kplus_err,dv1dy_kminus,dv1dy_kminus_err,d3v1dy3_kminus,d3v1dy3_kminus_err\n";
    for (int i = 0; i < 9; i++)
    {
        outs_kaon << centralities[i] << ",";
        outs_kaon << dv1dy_pos[2][i] << ",";
        outs_kaon << dv1dy_pos_err[2][i] << ",";
        outs_kaon << d3v1dy3_pos[2][i] << ",";
        outs_kaon << d3v1dy3_pos_err[2][i] << ",";
        outs_kaon << dv1dy_neg[2][i] << ",";
        outs_kaon << dv1dy_neg_err[2][i] << ",";
        outs_kaon << d3v1dy3_neg[2][i] << ",";
        outs_kaon << d3v1dy3_neg_err[2][i] << std::endl;
    }
    outs_kaon.close();

    TString out_pion = out_tstr.Copy();
    out_pion.Replace(out_pion.Length()-11, 11, "_pion.csv");
    std::ofstream outs_pion(out_pion);
    outs_pion << "centralities,dv1dy_piplus,dv1dy_piplus_err,d3v1dy3_piplus,d3v1dy3_piplus_err,dv1dy_piminus,dv1dy_piminus_err,d3v1dy3_piminus,d3v1dy3_piminus_err\n";
    for (int i = 0; i < 9; i++)
    {
        outs_pion << centralities[i] << ",";
        outs_pion << dv1dy_pos[3][i] << ",";
        outs_pion << dv1dy_pos_err[3][i] << ",";
        outs_pion << d3v1dy3_pos[3][i] << ",";
        outs_pion << d3v1dy3_pos_err[3][i] << ",";
        outs_pion << dv1dy_neg[3][i] << ",";
        outs_pion << dv1dy_neg_err[3][i] << ",";
        outs_pion << d3v1dy3_neg[3][i] << ",";
        outs_pion << d3v1dy3_neg_err[3][i] << std::endl;
    }
    outs_pion.close();
}