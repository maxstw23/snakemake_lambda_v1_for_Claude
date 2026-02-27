#include <iostream>
#include <TFile.h>
#include <TH1D.h>
#include <TF1.h>
#include <TMath.h>
#include <TString.h>

// 1D Functional Form for Lambda Efficiency
Double_t eff_fit_1D(Double_t *x, Double_t *par) {
    Double_t pt = x[0];
    if (pt <= 0.01) return 0;
    // (p0 + p3*pt + p4*pt^2) * exp(-pow(p1/pt, p2))
    return (par[0] + par[3]*pt + par[4]*pt*pt) * TMath::Exp(-TMath::Power(par[1]/pt, par[2]));
}

void calculate_lambda_eff(const char* inputPath, const char* outputPath, double y_cut = 0.6) 
{
    TFile *fIn = TFile::Open(inputPath, "READ");
    if (!fIn || fIn->IsZombie()) return;

    TFile *fOut = TFile::Open(outputPath, "RECREATE");
    if (!fOut || fOut->IsZombie()) return;

    double P[5][9];
    for(int i=0; i<5; ++i) {
        for (int j=0; j<9; ++j) P[i][j] = 0.0;
    }

    for (int cen = 0; cen <= 8; ++cen) {
        TH1D *hMC = (TH1D*)fIn->Get(Form("hMCParPt_%d", cen));
        TH1D *hReco = (TH1D*)fIn->Get(Form("hKFPRecoParPt_%d", cen));

        if (!hMC || !hReco) continue;

        // Data Cleaning: Ensure r <= m for binomial logic
        for (int i = 1; i <= hMC->GetNbinsX(); ++i) {
            double m = hMC->GetBinContent(i);
            double r = hReco->GetBinContent(i);
            if (r > m && m > 0) hReco->SetBinContent(i, m);
            if (r < 0) hReco->SetBinContent(i, 0);
            if (m < 0) hMC->SetBinContent(i, 0);
        }

        TH1D *hEff = (TH1D*)hReco->Clone(Form("hEff_cen%d", cen));
        hEff->SetTitle(Form("Lambda Efficiency Cent %d (|y| < %.2f);p_{T} (GeV/c);#epsilon", cen, y_cut));
        hEff->Divide(hReco, hMC, 1.0, 1.0, "B");

        TF1 *func1D = new TF1(Form("fit1D_cen%d", cen), eff_fit_1D, 0.0, 1.8, 5);
        func1D->SetParameters(0.20, 0.75, 2.8, 0.02, -0.01);
        func1D->SetParLimits(0, 0.0, 0.6);

        hEff->Fit(func1D, "RQ");

        for (int j = 0; j < 5; j++) {
            P[j][cen] = func1D->GetParameter(j);
        }

        // Save only once per loop iteration
        fOut->cd();
        hEff->Write();
        func1D->Write();
        hMC->Write();
        hReco->Write();
    }

    // Close performs the final write; no redundant fOut->Write() here
    fOut->Close();
    fIn->Close();

    std::cout << "\n// Lambda Efficiency Parameters (|y| < " << y_cut << ")\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "const float P" << i << "_Lambda_1D[9] = {";
        for (int j = 0; j < 9; j++) {
            std::cout << (float)P[i][j] << (j == 8 ? "" : ", ");
        }
        std::cout << "};\n";
    }
}