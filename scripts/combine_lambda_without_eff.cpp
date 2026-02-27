#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include "TSystem.h"
#include "TF1.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TMath.h"
#include "TFile.h"
#include "TString.h"
#include <algorithm>

std::string lower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    return str;
}

std::string upper_case_first(std::string str) {
    std::transform(str.begin(), str.begin() + 1, str.begin(), ::toupper);
    return str;
}

void combine_lambda_without_eff(std::string inputDir="./", std::string outputDir="./", 
                                float pt_lo=0.42, float pt_hi=1.8, 
                                std::string particle="Lambda", std::string flow_case="v1", 
                                std::string energy="7p7GeV", float y_cut=0.6)   
{
    int ptbin_lo = (int)(pt_lo * 10);
    int ptbin_hi = (int)(pt_hi * 10) - 1;
    TFile f(inputDir.c_str(), "read");

    TH1D *hout[9][20], *hout_pt_east[9][14], *hout_pt_west[9][14];
    TProfile *pout[9][20], *pout_pt_east[9][14], *pout_pt_west[9][14];
    TH1D *huncorrected[9][20], *huncorrected_pt_east[9][14], *huncorrected_pt_west[9][14];
    TProfile *puncorrected[9][20], *puncorrected_pt_east[9][14], *puncorrected_pt_west[9][14];

    int num_ybin = (energy == "7p7GeV" || energy == "9p2GeV" || energy == "11p5GeV" || 
                    energy == "14p6GeV" || energy == "17p3GeV" || energy == "19p6GeV" || 
                    energy == "27GeV") ? 20 : 10;
                    
    float y_interval = 2.0 / num_ybin;
    
    // 1. Pre-calculate valid y bins based on y_cut
    std::vector<int> valid_y_east, valid_y_west;
    for (int ybin = 0; ybin < num_ybin; ybin++) {
        float y = -1.0 + y_interval * (ybin + 0.5); // bin center
        if (y >= -y_cut && y < 0) valid_y_east.push_back(ybin);
        if (y > 0 && y <= y_cut)  valid_y_west.push_back(ybin);
    }

    std::string particle_upper = upper_case_first(particle);

    for (int cen = 0; cen < 9; cen++) 
    {   
        // --- A. Integrate over pT (y-dependent result) ---
        for (int ybin = 0; ybin < num_ybin; ybin++) 
        {   
            // Start with an empty clone to avoid double counting
            pout[cen][ybin] = (TProfile*)f.Get(Form("h%s_EPD_%s_pt_%d_%d_%d", particle_upper.c_str(), flow_case.c_str(), cen, ybin, ptbin_lo))->Clone(Form("h%s_EPD_%s_%d_%d", particle_upper.c_str(), flow_case.c_str(), cen, ybin));
            pout[cen][ybin]->Reset(); 
            pout[cen][ybin]->Sumw2();
            pout[cen][ybin]->Approximate(true);
            
            puncorrected[cen][ybin] = (TProfile*)pout[cen][ybin]->Clone(Form("h%s_EPD_%s_%d_%d_uncorrected", particle_upper.c_str(), flow_case.c_str(), cen, ybin));
            
            hout[cen][ybin] = (TH1D*)f.Get(Form("h%sM_cen_y_pt_%d_%d_%d", particle_upper.c_str(), cen, ybin, ptbin_lo))->Clone(Form("h%sM_cen_y_%d_%d", particle_upper.c_str(), cen, ybin));
            hout[cen][ybin]->Reset();
            hout[cen][ybin]->Sumw2();
            
            huncorrected[cen][ybin] = (TH1D*)hout[cen][ybin]->Clone(Form("h%sM_cen_y_%d_%d_uncorrected", particle_upper.c_str(), cen, ybin));

            // Loop ALL valid pT bins
            for (int i = ptbin_lo; i <= ptbin_hi; i++)
            {
                TProfile *p1 = (TProfile *)f.Get(Form("h%s_EPD_%s_pt_%d_%d_%d", particle_upper.c_str(), flow_case.c_str(), cen, ybin, i));
                pout[cen][ybin]->Add(pout[cen][ybin], p1, 1.0, 1.0);
                puncorrected[cen][ybin]->Add(puncorrected[cen][ybin], p1, 1.0, 1.0);
                
                TH1D *h1 = (TH1D *)f.Get(Form("h%sM_cen_y_pt_%d_%d_%d", particle_upper.c_str(), cen, ybin, i));
                hout[cen][ybin]->Add(hout[cen][ybin], h1, 1.0, 1.0);
                huncorrected[cen][ybin]->Add(huncorrected[cen][ybin], h1, 1.0, 1.0);
            }

            // Post merging error correction
            pout[cen][ybin]->SetErrorOption("s");
            puncorrected[cen][ybin]->SetErrorOption("s");
            
            for (int i = 0; i < pout[cen][ybin]->GetNbinsX(); i++) {
                float content = pout[cen][ybin]->GetBinContent(i + 1); 
                float error = pout[cen][ybin]->GetBinError(i + 1); 
                float entries_new = puncorrected[cen][ybin]->GetBinEntries(i + 1); 
                float W = entries_new;          
                float H = content * entries_new; 
                float E = (error * error + content * content) * entries_new; 
                
                pout[cen][ybin]->SetBinEntries(i + 1, W);
                pout[cen][ybin]->SetBinContent(i + 1, H);
                pout[cen][ybin]->SetBinError(i + 1, sqrt(E));
                pout[cen][ybin]->Sumw2(false);
                pout[cen][ybin]->Sumw2();
            }
            pout[cen][ybin]->SetErrorOption("");
            puncorrected[cen][ybin]->SetErrorOption("");
        }
    
        // --- B. Integrate over y (pT-dependent result) ---
        for (int i = ptbin_lo; i <= ptbin_hi; i++)
        {   
            int pt_idx = i - ptbin_lo;
            
            // EAST Initialization
            pout_pt_east[cen][pt_idx] = (TProfile*)f.Get(Form("h%s_EPD_%s_pt_%d_%d_%d", particle_upper.c_str(), flow_case.c_str(), cen, valid_y_east[0], i))->Clone(Form("h%s_EPD_%s_pt_east_%d_%d", particle_upper.c_str(), flow_case.c_str(), cen, i));
            pout_pt_east[cen][pt_idx]->Reset();
            pout_pt_east[cen][pt_idx]->Sumw2();
            pout_pt_east[cen][pt_idx]->Approximate(true);
            
            puncorrected_pt_east[cen][pt_idx] = (TProfile*)pout_pt_east[cen][pt_idx]->Clone(Form("h%s_EPD_%s_pt_east_%d_%d_uncorrected", particle_upper.c_str(), flow_case.c_str(), cen, i));
            
            hout_pt_east[cen][pt_idx] = (TH1D*)f.Get(Form("h%sM_cen_y_pt_%d_%d_%d", particle_upper.c_str(), cen, valid_y_east[0], i))->Clone(Form("h%sM_cen_pt_east_%d_%d", particle_upper.c_str(), cen, i));
            hout_pt_east[cen][pt_idx]->Reset();
            hout_pt_east[cen][pt_idx]->Sumw2();
            
            huncorrected_pt_east[cen][pt_idx] = (TH1D*)hout_pt_east[cen][pt_idx]->Clone(Form("h%sM_cen_pt_east_%d_%d_uncorrected", particle_upper.c_str(), cen, i));

            // EAST Integration (strictly within -y_cut to 0)
            for (int ybin : valid_y_east) 
            {
                int next_weight = -1; // flip sign for negative rapidity
                
                TProfile *p1 = (TProfile *)f.Get(Form("h%s_EPD_%s_pt_%d_%d_%d", particle_upper.c_str(), flow_case.c_str(), cen, ybin, i));
                pout_pt_east[cen][pt_idx]->Add(pout_pt_east[cen][pt_idx], p1, 1.0, next_weight);
                puncorrected_pt_east[cen][pt_idx]->Add(puncorrected_pt_east[cen][pt_idx], p1, 1.0, 1.0);
                
                TH1D *h1 = (TH1D *)f.Get(Form("h%sM_cen_y_pt_%d_%d_%d", particle_upper.c_str(), cen, ybin, i));
                hout_pt_east[cen][pt_idx]->Add(hout_pt_east[cen][pt_idx], h1, 1.0, 1.0);
                huncorrected_pt_east[cen][pt_idx]->Add(huncorrected_pt_east[cen][pt_idx], h1, 1.0, 1.0);
            }

            // WEST Initialization
            pout_pt_west[cen][pt_idx] = (TProfile*)f.Get(Form("h%s_EPD_%s_pt_%d_%d_%d", particle_upper.c_str(), flow_case.c_str(), cen, valid_y_west[0], i))->Clone(Form("h%s_EPD_%s_pt_west_%d_%d", particle_upper.c_str(), flow_case.c_str(), cen, i));
            pout_pt_west[cen][pt_idx]->Reset();
            pout_pt_west[cen][pt_idx]->Sumw2();
            pout_pt_west[cen][pt_idx]->Approximate(true);
            
            puncorrected_pt_west[cen][pt_idx] = (TProfile*)pout_pt_west[cen][pt_idx]->Clone(Form("h%s_EPD_%s_pt_west_%d_%d_uncorrected", particle_upper.c_str(), flow_case.c_str(), cen, i));
            
            hout_pt_west[cen][pt_idx] = (TH1D*)f.Get(Form("h%sM_cen_y_pt_%d_%d_%d", particle_upper.c_str(), cen, valid_y_west[0], i))->Clone(Form("h%sM_cen_pt_west_%d_%d", particle_upper.c_str(), cen, i));
            hout_pt_west[cen][pt_idx]->Reset();
            hout_pt_west[cen][pt_idx]->Sumw2();
            
            huncorrected_pt_west[cen][pt_idx] = (TH1D*)hout_pt_west[cen][pt_idx]->Clone(Form("h%sM_cen_pt_west_%d_%d_uncorrected", particle_upper.c_str(), cen, i));
            
            // WEST Integration (strictly within 0 to y_cut)
            for (int ybin : valid_y_west) 
            {
                TProfile *p1 = (TProfile *)f.Get(Form("h%s_EPD_%s_pt_%d_%d_%d", particle_upper.c_str(), flow_case.c_str(), cen, ybin, i));
                pout_pt_west[cen][pt_idx]->Add(pout_pt_west[cen][pt_idx], p1, 1.0, 1.0);
                puncorrected_pt_west[cen][pt_idx]->Add(puncorrected_pt_west[cen][pt_idx], p1, 1.0, 1.0);
                
                TH1D *h1 = (TH1D *)f.Get(Form("h%sM_cen_y_pt_%d_%d_%d", particle_upper.c_str(), cen, ybin, i));
                hout_pt_west[cen][pt_idx]->Add(hout_pt_west[cen][pt_idx], h1, 1.0, 1.0);
                huncorrected_pt_west[cen][pt_idx]->Add(huncorrected_pt_west[cen][pt_idx], h1, 1.0, 1.0);
            }
            
            // Post merging error corrections for East
            pout_pt_east[cen][pt_idx]->SetErrorOption("s");
            for (int j = 0; j < pout_pt_east[cen][pt_idx]->GetNbinsX(); j++) {
                float content = pout_pt_east[cen][pt_idx]->GetBinContent(j + 1); 
                float error = pout_pt_east[cen][pt_idx]->GetBinError(j + 1); 
                float entries_new = puncorrected_pt_east[cen][pt_idx]->GetBinEntries(j + 1); 
                float W = entries_new;          
                float H = content * entries_new; 
                float E = (error * error + content * content) * entries_new; 
                pout_pt_east[cen][pt_idx]->SetBinEntries(j + 1, W);
                pout_pt_east[cen][pt_idx]->SetBinContent(j + 1, H);
                pout_pt_east[cen][pt_idx]->SetBinError(j + 1, sqrt(E));
                pout_pt_east[cen][pt_idx]->Sumw2(false);
                pout_pt_east[cen][pt_idx]->Sumw2();
            }
            pout_pt_east[cen][pt_idx]->SetErrorOption("");

            // Post merging error corrections for West
            pout_pt_west[cen][pt_idx]->SetErrorOption("s");
            for (int j = 0; j < pout_pt_west[cen][pt_idx]->GetNbinsX(); j++) {
                float content = pout_pt_west[cen][pt_idx]->GetBinContent(j + 1); 
                float error = pout_pt_west[cen][pt_idx]->GetBinError(j + 1); 
                float entries_new = puncorrected_pt_west[cen][pt_idx]->GetBinEntries(j + 1); 
                float W = entries_new;          
                float H = content * entries_new; 
                float E = (error * error + content * content) * entries_new; 
                pout_pt_west[cen][pt_idx]->SetBinEntries(j + 1, W);
                pout_pt_west[cen][pt_idx]->SetBinContent(j + 1, H);
                pout_pt_west[cen][pt_idx]->SetBinError(j + 1, sqrt(E));
                pout_pt_west[cen][pt_idx]->Sumw2(false);
                pout_pt_west[cen][pt_idx]->Sumw2();
            }
            pout_pt_west[cen][pt_idx]->SetErrorOption("");
        }
    }

    TFile fout(Form("%s/combined_%s_%s_%s.root", outputDir.c_str(), particle.c_str(), flow_case.c_str(), energy.c_str()), "recreate");
    for (int cen = 0; cen < 9; cen++) {
        for (int ybin = 0; ybin < num_ybin; ybin++) {
            pout[cen][ybin]->Write();
            hout[cen][ybin]->Write();
        }
        for (int i = ptbin_lo; i <= ptbin_hi; i++) {
            pout_pt_east[cen][i-ptbin_lo]->Write();
            hout_pt_east[cen][i-ptbin_lo]->Write();
            pout_pt_west[cen][i-ptbin_lo]->Write();
            hout_pt_west[cen][i-ptbin_lo]->Write();
        }
    }
    
    f.Close();
    fout.Close();
    std::cout << "Done" << std::endl;
}