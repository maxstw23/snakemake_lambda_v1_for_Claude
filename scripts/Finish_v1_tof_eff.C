#include "stdio.h"
#include "TFile.h"
#include <fstream>
#include <iostream>
#include "TTree.h"
#include "TObjArray.h"
#include <stdlib.h>
#include "TVector2.h"
#include "TString.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TMath.h"
#include "TText.h"

static Double_t chi(double res);
static Double_t resEventPlane(double chi);
static Double_t resEventPlaneK2(double chi);
static Double_t resEventPlaneK3(double chi);
static Double_t resEventPlaneK4(double chi);
static TH1D* Profile2D_X(TProfile2D* Tp2D,Float_t Lptcut,Float_t Hptcut,Float_t Letacut,Float_t Hetacut,Int_t centr, TH1* rc);
static TH1D* Profile2D_Y(TProfile2D* Tpp2D,Float_t Lptcut,Float_t Hptcut,Float_t Letacut,Float_t Hetacut,Int_t centr, TH1* rc);
static TH1D* Profile2D_Pt(TProfile2D* Tpp2D,Float_t Lptcut,Float_t Hptcut,Float_t Letacut,Float_t Hetacut,Int_t centr, TH1* rc);

const Double_t pi=TMath::Pi();
// const int opt_a1 = 0;    //0 means v1 and 1 means a1
const int opt_eff=0;    //Updated 25October for efficiency
// const Float_t ptHigh=1.2;   //pion~1.6, pr~2.8, K~1.6, ch~2
// const Float_t ptLow =0.28;   //ch,pion,K~0.2, pr~0.4
const Float_t etaHigh=1.0;   //1.3
const Float_t etaLow=-1.0;  //-1.3
Double_t halfpi  =pi/2.;
TString* histTitle;
TString xLabel = "Pseudorapidity";
char infile[60];
char outfile[60];

void Finish_v1_tof_eff(int cent, const char* inputDir, const char* outputDir, int opt_a1, Float_t ptLow, Float_t ptHigh) {  //, int ip, int op) {
Float_t  mRes, mResErr, mRes_c, mRes_cf, mRes_s, mRes_sf;
  //single histogram
    TH2D*     mHistYieldPart2D;
    TH2D*     mHistYieldPart2Dp;
    TH2D*     mHistYieldPart2Dn;
    
    TH1D*     mHistYieldPart2D_y;
    TH1D*     mHistYieldPart2Dp_y;
    TH1D*     mHistYieldPart2Dn_y;
      
    TH1D*     mHistYieldPart2D_x;
    TH1D*     mHistYieldPart2Dp_x;
    TH1D*     mHistYieldPart2Dn_x;

    TProfile2D* mHistYieldCos2Dp;
    TProfile2D* mHistYieldSin2Dp;
    TProfile2D* mHistYieldCos2Dn;
    TProfile2D* mHistYieldSin2Dn;
    TH1D*       mHistYieldSin_etap;
    TH1D*       mHistYieldCos_etap;
    TH1D*       mHistYieldSin_ptTPCp;
    TH1D*       mHistYieldCos_ptTPCp;
    TH1D*       mHistYieldSin_etan;
    TH1D*       mHistYieldCos_etan;
    TH1D*       mHistYieldSin_ptTPCn;
    TH1D*       mHistYieldCos_ptTPCn;
    TProfile2D* mHist_vObs2D_e_c_p1;
    TProfile2D* mHist_vObs2D_w_c_p1;
    TProfile2D* mHist_vObs2D_e_s_p1;
    TProfile2D* mHist_vObs2D_w_s_p1;
    TProfile2D* mHist_vObs2D_e_c_n1;
    TProfile2D* mHist_vObs2D_w_c_n1;
    TProfile2D* mHist_vObs2D_e_s_n1;
    TProfile2D* mHist_vObs2D_w_s_n1;

    TProfile*   mHist_vObsPt_TPC;
    TH1D*       mHist_vEta_e_c_p1;
    TH1D*       mHist_vEta_e_s_p1;
    TH1D*       mHist_vEta_w_c_p1;
    TH1D*       mHist_vEta_w_s_p1;
    TH1D*       mHist_vEta_e_c_n1;
    TH1D*       mHist_vEta_e_s_n1;
    TH1D*       mHist_vEta_w_c_n1;
    TH1D*       mHist_vEta_w_s_n1;
    TH1D*       mHist_vPt_TPC;
    TH1D*       mHist_vPt_TPC_e_c_p1;
    TH1D*       mHist_vPt_TPC_e_s_p1;
    TH1D*       mHist_vPt_TPC_w_c_p1;
    TH1D*       mHist_vPt_TPC_w_s_p1;
    TH1D*       mHist_vPt_TPC_e_c_n1;
    TH1D*       mHist_vPt_TPC_e_s_n1;
    TH1D*       mHist_vPt_TPC_w_c_n1;
    TH1D*       mHist_vPt_TPC_w_s_n1;
    TH1D*       mHist_vEta_x_p;
    TH1D*       mHist_vEta_x_n;
    TH1D*       mHist_vEta_y_p;
    TH1D*       mHist_vEta_y_n;
    TH1D*       mHist_vEta_e_p;
    TH1D*       mHist_vEta_e_n;
    TH1D*       mHist_vEta_w_p;
    TH1D*       mHist_vEta_w_n;
    TH1D*       mHist_vEta_f_n;
    TH1D*       mHist_vEta_f_p;
    TH1D*       mHist_vEta_e_ch;
    TH1D*       mHist_vEta_w_ch;
    TH1D*       mHist_vEta_ff;
    TH1D*       mHist_vPt_TPC_x_p;
    TH1D*       mHist_vPt_TPC_y_p;
    TH1D*       mHist_vPt_TPC_f_p;
    TH1D*       mHist_vPt_TPC_x_n;
    TH1D*       mHist_vPt_TPC_y_n;
    TH1D*       mHist_vPt_TPC_f_n;
    TH1D*       mHist_vPt_TPC_e_p;
    TH1D*       mHist_vPt_TPC_w_p;
    TH1D*       mHist_vPt_TPC_e_n;
    TH1D*       mHist_vPt_TPC_w_n;
    TH1D*       mHist_vPt_TPC_e_ch;
    TH1D*       mHist_vPt_TPC_w_ch;
    TH1D*       mHist_vPt_TPC_ff;

    TH1D*       mHistRes_Eta;
    TH1D*       mHistRes_Eta_sub;
    TH1D*       mHistRes_Eta_c;
    TH1D*       mHistRes_Eta_s;
    TH1D*       mHistRes_Eta_cf;
    TH1D*       mHistRes_Eta_sf;
    TH1D*       mHistRes_Pt;
    TH1D*       mHistRes_Pt_sub;
    TH1D*       mHistRes_Pt_c;
    TH1D*       mHistRes_Pt_s;
    TH1D*       mHistRes_Pt_cf;
    TH1D*       mHistRes_Pt_sf;
    TProfile* mHistCos;
    TProfile* mHistCos_cs;
    TProfile* mHistCos_ew;
    TH1F*     mHistRes;

char efffile[200];
snprintf(efffile,100,inputDir,cent);
TFile *f = new TFile(efffile,"READ");
TH1* mc = (TH1*)f->Get("Hist_Pt");
TH1* rc = (TH1*)f->Get("Hist_Pt_TOF");
rc->Divide(mc);

snprintf(outfile,60,outputDir,cent);
// if(opt_a1==1) snprintf(outfile,60,"Result/cen%d.a1_pion.root",cent);
TFile fout(outfile,"RECREATE");

  // Yield for particles correlated with the event plane
  mHistYieldPart2D =(TH2D*)f->Get("EtaPtDist");
  mHistYieldPart2Dp=(TH2D*)f->Get("EtaPtDistp");
  mHistYieldPart2Dn=(TH2D*)f->Get("EtaPtDistn");
    
    mHistYieldPart2D_y=mHistYieldPart2D->ProjectionY();
    mHistYieldPart2Dp_y=mHistYieldPart2Dp->ProjectionY();
    mHistYieldPart2Dn_y=mHistYieldPart2Dn->ProjectionY();
    //cout<<"mHistYieldPart2D_y->GetNbinsX()="<<mHistYieldPart2D_y->GetNbinsX()<<endl;

    Int_t ptLowbin=mHistYieldPart2D_y->FindBin(ptLow+1e-5);
    Int_t ptLowbinp=mHistYieldPart2Dp_y->FindBin(ptLow+1e-5);
    Int_t ptLowbinn=mHistYieldPart2Dn_y->FindBin(ptLow+1e-5);
    //cout<<"ptLowbin="<<ptLowbin<<endl;

    Int_t ptHighbin=mHistYieldPart2D_y->FindBin(ptHigh-1e-5);
    Int_t ptHighbinp=mHistYieldPart2Dp_y->FindBin(ptHigh-1e-5);
    Int_t ptHighbinn=mHistYieldPart2Dn_y->FindBin(ptHigh-1e-5);
    //cout<<"ptHighbin="<<ptHighbin<<endl;

    
    mHistYieldPart2D_x=mHistYieldPart2D->ProjectionX("EtaPtDist_ptsel",ptLowbin,ptHighbin);
    mHistYieldPart2Dp_x=mHistYieldPart2Dp->ProjectionX("EtaPtDistp_ptsel",ptLowbin,ptHighbin);
    mHistYieldPart2Dn_x=mHistYieldPart2Dn->ProjectionX("EtaPtDistn_ptsel",ptLowbin,ptHighbin);
  
      //mHistYieldPart2D_x->Draw();
      histTitle = new TString("Hist_Yield_cos1");
      mHistYieldCos2Dp = (TProfile2D*)f->Get(histTitle->Data());
    mHistYieldCos2Dp->RebinY(4); //Added 24July2023 rebin
      delete histTitle;

      histTitle = new TString("Hist_Yield_sin1");
      mHistYieldSin2Dp = (TProfile2D*)f->Get(histTitle->Data());
    mHistYieldSin2Dp->RebinY(4); //Added 24July2023 rebin
      delete histTitle;

      histTitle = new TString("Hist_Yield_cos2");
      mHistYieldCos2Dn = (TProfile2D*)f->Get(histTitle->Data());
    mHistYieldCos2Dn->RebinY(4); //Added 24July2023 rebin
      delete histTitle;

      histTitle = new TString("Hist_Yield_sin2");
      mHistYieldSin2Dn = (TProfile2D*)f->Get(histTitle->Data());
    mHistYieldSin2Dn->RebinY(4); //Added 24July2023 rebin
      delete histTitle;

      histTitle = new TString("Flow_YieldCos_eta_Selp");
      mHistYieldCos_etap=Profile2D_X(mHistYieldCos2Dp,ptLow,ptHigh,etaLow,etaHigh,cent,rc);//pi~.75,K~.65,p~1 //etaLow etaHigh added 1April 2023 in all TProfile2D_X calling
      mHistYieldCos_etap->SetName(histTitle->Data());
      delete histTitle;
      histTitle = new TString("Flow_YieldCos_eta_Seln");
      mHistYieldCos_etan=Profile2D_X(mHistYieldCos2Dn,ptLow,ptHigh,etaLow,etaHigh,cent,rc);//pi~.75,K~.65,p~1
      mHistYieldCos_etan->SetName(histTitle->Data());
      delete histTitle;

      histTitle = new TString("Flow_YieldSin_eta_Selp");
      mHistYieldSin_etap=Profile2D_X(mHistYieldSin2Dp,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHistYieldSin_etap->SetName(histTitle->Data());
      delete histTitle;
      histTitle = new TString("Flow_YieldSin_eta_Seln");
      mHistYieldSin_etan=Profile2D_X(mHistYieldSin2Dn,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHistYieldSin_etan->SetName(histTitle->Data());
      delete histTitle;

      histTitle = new TString("Flow_YieldCos_ptTPC_Selp");
      mHistYieldCos_ptTPCp = Profile2D_Y(mHistYieldCos2Dp,ptLow,ptHigh,etaLow,etaHigh,cent,rc);//-1.3 to 1.3 //Added 9March cent,rc to all Profile2D_Y and Profile2D_Pt calling
      mHistYieldCos_ptTPCp->SetName(histTitle->Data());
      delete histTitle;
      histTitle = new TString("Flow_YieldCos_ptTPC_Seln");
      mHistYieldCos_ptTPCn = Profile2D_Y(mHistYieldCos2Dn,ptLow,ptHigh,etaLow,etaHigh,cent,rc);//-1.3 to 1.3
      mHistYieldCos_ptTPCn->SetName(histTitle->Data());
      delete histTitle;

      histTitle = new TString("Flow_YieldSin_ptTPC_Selp");
      mHistYieldSin_ptTPCp = Profile2D_Y(mHistYieldSin2Dp,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHistYieldSin_ptTPCp->SetName(histTitle->Data());
      delete histTitle;
      histTitle = new TString("Flow_YieldSin_ptTPC_Seln");
      mHistYieldSin_ptTPCn = Profile2D_Y(mHistYieldSin2Dn,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHistYieldSin_ptTPCn->SetName(histTitle->Data());
      delete histTitle;
//use EPD instead of ZDC-SMD
    histTitle = new TString("Hist_cos_EPD");
    mHistCos = (TProfile*)f->Get(histTitle->Data());
    delete histTitle;
      double ZDCSMD_deltaResSub = 0.005,ZDCSMD_mResDelta=0.;
      double corr = mHistCos->GetBinContent(4);
      double corr_err = mHistCos->GetBinError(4);
      double ZDCSMD_resSub = (corr>0.) ? sqrt(corr) : 0.;
      double ZDCSMD_resSubErr = (corr>0.) ? corr_err/(2.*ZDCSMD_resSub) : 0.;
      double ZDCSMD_chiSub = chi(ZDCSMD_resSub);
      double ZDCSMD_chiSubDelta = chi((ZDCSMD_resSub+ZDCSMD_deltaResSub));

        Float_t Res_c = mHistCos->GetBinContent(5);
        Float_t Res_c_err = mHistCos->GetBinError(5);
        Float_t Res_s = mHistCos->GetBinContent(6);
        Float_t Res_s_err = mHistCos->GetBinError(6);
      double ZDCSMDc_mResDelta=0.,ZDCSMDs_mResDelta=0.,ZDCSMDcf_mResDelta=0.,ZDCSMDsf_mResDelta=0.;
      double ZDCSMDc_resSub = (Res_c>0.) ? sqrt(2*Res_c) : 0.;
      double ZDCSMDc_resSubErr = (Res_c>0.) ? 2.*Res_c_err/(2.*ZDCSMDc_resSub) : 0.;
      double ZDCSMDc_chiSub = chi(ZDCSMDc_resSub);
      double ZDCSMDc_chiSubDelta = chi((ZDCSMDc_resSub+ZDCSMD_deltaResSub));
      double ZDCSMDs_resSub = (Res_s>0.) ? sqrt(2*Res_s) : 0.;
      double ZDCSMDs_resSubErr = (Res_s>0.) ? 2.*Res_s_err/(2.*ZDCSMDs_resSub) : 0.;
      double ZDCSMDs_chiSub = chi(ZDCSMDs_resSub);
      double ZDCSMDs_chiSubDelta = chi((ZDCSMDs_resSub+ZDCSMD_deltaResSub));

            mRes =resEventPlane(sqrt(2.) * ZDCSMD_chiSub);
            ZDCSMD_mResDelta = resEventPlane(sqrt(2.) * ZDCSMD_chiSubDelta);
            mRes_c =resEventPlane(ZDCSMDc_chiSub);
            ZDCSMDc_mResDelta = resEventPlane(ZDCSMDc_chiSubDelta);
            mRes_cf=resEventPlane(sqrt(2.) * ZDCSMDc_chiSub);
            ZDCSMDcf_mResDelta = resEventPlane(sqrt(2.) * ZDCSMDc_chiSubDelta);
            mRes_s =resEventPlane(ZDCSMDs_chiSub);
            ZDCSMDs_mResDelta = resEventPlane(ZDCSMDs_chiSubDelta);
            mRes_sf=resEventPlane(sqrt(2.) * ZDCSMDs_chiSub);
            ZDCSMDsf_mResDelta = resEventPlane(sqrt(2.) * ZDCSMDs_chiSubDelta);
            mResErr = ZDCSMD_resSubErr * fabs ((double)mRes - ZDCSMD_mResDelta) / ZDCSMD_deltaResSub;
       cout<<"mRes = "<<mRes<<" mResErr = "<<mResErr<<endl;
       cout<<"mRes_c="<<mRes_c<<" mRes_cErr="<<
                ZDCSMDc_resSubErr*fabs ((double)mRes_c - ZDCSMDc_mResDelta) / ZDCSMD_deltaResSub<<endl;
        cout<<"mRes_s="<<mRes_s<<" mRes_sErr="<<
                ZDCSMDs_resSubErr*fabs ((double)mRes_s - ZDCSMDs_mResDelta) / ZDCSMD_deltaResSub<<endl;
        cout<<"mRes_cf="<<mRes_cf<<" mRes_cfErr="<<
                ZDCSMDc_resSubErr*fabs ((double)mRes_cf - ZDCSMDcf_mResDelta) / ZDCSMD_deltaResSub<<endl;
        cout<<"mRes_sf="<<mRes_sf<<" mRes_sfErr="<<
                ZDCSMDs_resSubErr*fabs ((double)mRes_sf - ZDCSMDsf_mResDelta) / ZDCSMD_deltaResSub<<endl;

     // Resolution histograms
      histTitle = new TString("Flow_Res_Eta_Sel");
      mHistRes_Eta = mHistYieldCos2Dp->ProjectionX();//Profile2D_X(mHistYieldCos2Dp,.0,2.0,cent,rc);
      mHistRes_Eta->SetName(histTitle->Data());
      delete histTitle;

      histTitle = new TString("Flow_Res_Eta_sub_Sel");
      mHistRes_Eta_sub = mHistYieldCos2Dp->ProjectionX();//Profile2D_X(mHistYieldCos2Dp,.0,2.0,cent,rc);
      mHistRes_Eta_sub->SetName(histTitle->Data());
      delete histTitle;

      histTitle = new TString("Flow_Res_Eta_c_Sel");
      mHistRes_Eta_c = mHistYieldCos2Dp->ProjectionX();//Profile2D_X(mHistYieldCos2Dp,.0,2.0,cent,rc);
      mHistRes_Eta_c->SetName(histTitle->Data());
      delete histTitle;

      histTitle = new TString("Flow_Res_Eta_s_Sel");
      mHistRes_Eta_s = mHistYieldCos2Dp->ProjectionX();//Profile2D_X(mHistYieldSin2Dp,.0,2.0,cent,rc);
      mHistRes_Eta_s->SetName(histTitle->Data());
      delete histTitle;

      histTitle = new TString("Flow_Res_Eta_cf_Sel");
      mHistRes_Eta_cf = mHistYieldCos2Dp->ProjectionX();//Profile2D_X(mHistYieldCos2Dp,.0,2.0,cent,rc);
      mHistRes_Eta_cf->SetName(histTitle->Data());
      delete histTitle;

      histTitle = new TString("Flow_Res_Eta_sf_Sel");
      mHistRes_Eta_sf = mHistYieldCos2Dp->ProjectionX();//Profile2D_X(mHistYieldSin2Dp,.0,2.0,cent,rc);
      mHistRes_Eta_sf->SetName(histTitle->Data());
      delete histTitle;

      histTitle = new TString("Flow_Res_Pt_Sel");
    mHistRes_Pt = mHistYieldCos2Dp->ProjectionY(); //Profile2D_Y(mHistYieldCos2Dp,-1,1,cent,rc); //Added 9March why ProjectionX for eta histograms and Profile2d_Y for pt histograms?
      mHistRes_Pt ->SetName(histTitle->Data());
      delete histTitle;

      histTitle = new TString("Flow_Res_Pt_sub_Sel");
    mHistRes_Pt_sub = mHistYieldCos2Dp->ProjectionY();//Profile2D_Y(mHistYieldCos2Dp,-1,1,cent,rc);
      mHistRes_Pt_sub->SetName(histTitle->Data());
      delete histTitle;

      histTitle = new TString("Flow_Res_Pt_c_Sel");
    mHistRes_Pt_c = mHistYieldCos2Dp->ProjectionY();//Profile2D_Y(mHistYieldCos2Dp,-1,1,cent,rc);
      mHistRes_Pt_c->SetName(histTitle->Data());
      delete histTitle;

      histTitle = new TString("Flow_Res_Pt_s_Sel");
    mHistRes_Pt_s = mHistYieldCos2Dp->ProjectionY();//Profile2D_Y(mHistYieldSin2Dp,-1,1,cent,rc);
      mHistRes_Pt_s->SetName(histTitle->Data());
      delete histTitle;

      histTitle = new TString("Flow_Res_Pt_cf_Sel");
    mHistRes_Pt_cf = mHistYieldCos2Dp->ProjectionY();//Profile2D_Y(mHistYieldCos2Dp,-1,1,cent,rc);
      mHistRes_Pt_cf->SetName(histTitle->Data());
      delete histTitle;

      histTitle = new TString("Flow_Res_Pt_sf_Sel");
    mHistRes_Pt_sf = mHistYieldCos2Dp->ProjectionY(); //Profile2D_Y(mHistYieldSin2Dp,-1,1,cent,rc);
      mHistRes_Pt_sf->SetName(histTitle->Data());
      delete histTitle;

      int Neta = mHistRes_Eta->GetNbinsX();
      for(int i=1;i<=Neta;i++) {
        mHistRes_Eta->SetBinContent(i,mRes);
        mHistRes_Eta->SetBinError(i,mResErr);
        mHistRes_Eta_sub->SetBinContent(i,ZDCSMD_resSub);
        mHistRes_Eta_sub->SetBinError(i,ZDCSMD_resSubErr);
        mHistRes_Eta_c->SetBinContent(i,mRes_c);
        mHistRes_Eta_c->SetBinError(i,ZDCSMDc_resSubErr*fabs ((double)mRes_c - ZDCSMDc_mResDelta) / ZDCSMD_deltaResSub);
        mHistRes_Eta_s->SetBinContent(i,mRes_s);
        mHistRes_Eta_s->SetBinError(i,ZDCSMDs_resSubErr*fabs ((double)mRes_s - ZDCSMDs_mResDelta) / ZDCSMD_deltaResSub);
        mHistRes_Eta_cf->SetBinContent(i,mRes_cf);
        mHistRes_Eta_cf->SetBinError(i,ZDCSMDc_resSubErr*fabs ((double)mRes_cf - ZDCSMDcf_mResDelta) / ZDCSMD_deltaResSub);
        mHistRes_Eta_sf->SetBinContent(i,mRes_sf);
        mHistRes_Eta_sf->SetBinError(i,ZDCSMDs_resSubErr*fabs ((double)mRes_sf - ZDCSMDsf_mResDelta) / ZDCSMD_deltaResSub);
      }
      int Npt = mHistRes_Pt->GetNbinsX();
      for(int i=1;i<=Npt;i++) {
        mHistRes_Pt->SetBinContent(i,mRes);
        mHistRes_Pt->SetBinError(i,mResErr);
        mHistRes_Pt_sub->SetBinContent(i,ZDCSMD_resSub);
        mHistRes_Pt_sub->SetBinError(i,ZDCSMD_resSubErr);
        mHistRes_Pt_c->SetBinContent(i,mRes_c);
        mHistRes_Pt_c->SetBinError(i,ZDCSMDc_resSubErr*fabs ((double)mRes_c - ZDCSMDc_mResDelta) / ZDCSMD_deltaResSub);
        mHistRes_Pt_s->SetBinContent(i,mRes_s);
        mHistRes_Pt_s->SetBinError(i,ZDCSMDs_resSubErr*fabs ((double)mRes_s - ZDCSMDs_mResDelta) / ZDCSMD_deltaResSub);
        mHistRes_Pt_cf->SetBinContent(i,mRes_cf);
        mHistRes_Pt_cf->SetBinError(i,ZDCSMDc_resSubErr*fabs ((double)mRes_cf - ZDCSMDcf_mResDelta) / ZDCSMD_deltaResSub);
        mHistRes_Pt_sf->SetBinContent(i,mRes_sf);
        mHistRes_Pt_sf->SetBinError(i,ZDCSMDs_resSubErr*fabs ((double)mRes_sf - ZDCSMDsf_mResDelta) / ZDCSMD_deltaResSub);
      }

        // Read 2D east, west, cos, sin
      histTitle = new TString("p_v1_e_c_obs1");
if(opt_a1==1) histTitle = new TString("p_a1_e_c_obs1");
      mHist_vObs2D_e_c_p1= (TProfile2D*)f->Get(histTitle->Data());
    mHist_vObs2D_e_c_p1->RebinY(4); //Added 24July2023 rebin
      delete histTitle;

      histTitle = new TString("p_v1_e_c_obs2");
if(opt_a1==1) histTitle = new TString("p_a1_e_c_obs2");
      mHist_vObs2D_e_c_n1= (TProfile2D*)f->Get(histTitle->Data());
    mHist_vObs2D_e_c_n1->RebinY(4); //Added 24July2023 rebin
      delete histTitle;

      histTitle = new TString("p_v1_e_s_obs1");
if(opt_a1==1) histTitle = new TString("p_a1_e_s_obs1");
      mHist_vObs2D_e_s_p1= (TProfile2D*)f->Get(histTitle->Data());
    mHist_vObs2D_e_s_p1->RebinY(4); //Added 24July2023 rebin
      delete histTitle;

      histTitle = new TString("p_v1_e_s_obs2");
if(opt_a1==1) histTitle = new TString("p_a1_e_s_obs2");
      mHist_vObs2D_e_s_n1= (TProfile2D*)f->Get(histTitle->Data());
    mHist_vObs2D_e_s_n1->RebinY(4); //Added 24July2023 rebin
      delete histTitle;

      histTitle = new TString("p_v1_w_c_obs1");
if(opt_a1==1) histTitle = new TString("p_a1_w_c_obs1");
      mHist_vObs2D_w_c_p1= (TProfile2D*)f->Get(histTitle->Data());
    mHist_vObs2D_w_c_p1->RebinY(4); //Added 24July2023 rebin
      delete histTitle;

      histTitle = new TString("p_v1_w_c_obs2");
if(opt_a1==1) histTitle = new TString("p_a1_w_c_obs2");
      mHist_vObs2D_w_c_n1= (TProfile2D*)f->Get(histTitle->Data());
    mHist_vObs2D_w_c_n1->RebinY(4); //Added 24July2023 rebin
      delete histTitle;

      histTitle = new TString("p_v1_w_s_obs1");
if(opt_a1==1) histTitle = new TString("p_a1_w_s_obs1");
      mHist_vObs2D_w_s_p1= (TProfile2D*)f->Get(histTitle->Data());
    mHist_vObs2D_w_s_p1->RebinY(4); //Added 24July2023 rebin
      delete histTitle;

      histTitle = new TString("p_v1_w_s_obs2");
if(opt_a1==1) histTitle = new TString("p_a1_w_s_obs2");
      mHist_vObs2D_w_s_n1= (TProfile2D*)f->Get(histTitle->Data());
    mHist_vObs2D_w_s_n1->RebinY(4); //Added 24July2023 rebin
      delete histTitle;

      histTitle = new TString("Flow_vEta_e_c_Selp1");
      mHist_vEta_e_c_p1 = Profile2D_X(mHist_vObs2D_e_c_p1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vEta_e_c_p1 ->SetName(histTitle->Data());
      mHist_vEta_e_c_p1 ->Scale(0.5);
      mHist_vEta_e_c_p1->Divide(mHistRes_Eta_c);
      mHist_vEta_e_c_p1->Divide(mHistYieldCos_etap);
      delete histTitle;
      histTitle = new TString("Flow_vEta_e_c_Seln1");
      mHist_vEta_e_c_n1 = Profile2D_X(mHist_vObs2D_e_c_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vEta_e_c_n1 ->SetName(histTitle->Data());
      mHist_vEta_e_c_n1 ->Scale(0.5);
      mHist_vEta_e_c_n1->Divide(mHistRes_Eta_c);
      mHist_vEta_e_c_n1->Divide(mHistYieldCos_etan);
      delete histTitle;

      histTitle = new TString("Flow_vEta_w_c_Selp1");
      mHist_vEta_w_c_p1 = Profile2D_X(mHist_vObs2D_w_c_p1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vEta_w_c_p1 ->SetName(histTitle->Data());
      mHist_vEta_w_c_p1 ->Scale(0.5);
      mHist_vEta_w_c_p1->Divide(mHistRes_Eta_c);
      mHist_vEta_w_c_p1->Divide(mHistYieldCos_etap);
      delete histTitle;
      histTitle = new TString("Flow_vEta_w_c_Seln1");
      mHist_vEta_w_c_n1 = Profile2D_X(mHist_vObs2D_w_c_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vEta_w_c_n1 ->SetName(histTitle->Data());
      mHist_vEta_w_c_n1 ->Scale(0.5);
      mHist_vEta_w_c_n1->Divide(mHistRes_Eta_c);
      mHist_vEta_w_c_n1->Divide(mHistYieldCos_etan);
      delete histTitle;

      histTitle = new TString("Flow_vEta_e_s_Selp1");
      mHist_vEta_e_s_p1 = Profile2D_X(mHist_vObs2D_e_s_p1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vEta_e_s_p1 ->SetName(histTitle->Data());
      mHist_vEta_e_s_p1 ->Scale(0.5);
      mHist_vEta_e_s_p1->Divide(mHistRes_Eta_s);
      mHist_vEta_e_s_p1->Divide(mHistYieldSin_etap);
      delete histTitle;
      histTitle = new TString("Flow_vEta_e_s_Seln1");
      mHist_vEta_e_s_n1 = Profile2D_X(mHist_vObs2D_e_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vEta_e_s_n1 ->SetName(histTitle->Data());
      mHist_vEta_e_s_n1 ->Scale(0.5);
      mHist_vEta_e_s_n1->Divide(mHistRes_Eta_s);
      mHist_vEta_e_s_n1->Divide(mHistYieldSin_etan);
      delete histTitle;

      histTitle = new TString("Flow_vEta_w_s_Selp1");
      mHist_vEta_w_s_p1 = Profile2D_X(mHist_vObs2D_w_s_p1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vEta_w_s_p1 ->SetName(histTitle->Data());
      mHist_vEta_w_s_p1 ->Scale(0.5);
      mHist_vEta_w_s_p1->Divide(mHistRes_Eta_s);
      mHist_vEta_w_s_p1->Divide(mHistYieldSin_etap);
      delete histTitle;
      histTitle = new TString("Flow_vEta_w_s_Seln1");
      mHist_vEta_w_s_n1 = Profile2D_X(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vEta_w_s_n1 ->SetName(histTitle->Data());
      mHist_vEta_w_s_n1 ->Scale(0.5);
      mHist_vEta_w_s_n1->Divide(mHistRes_Eta_s);
      mHist_vEta_w_s_n1->Divide(mHistYieldSin_etan);
      delete histTitle;

      histTitle = new TString("Flow_vPt_TPC_e_c_Selp1");
      mHist_vPt_TPC_e_c_p1 = Profile2D_Pt(mHist_vObs2D_e_c_p1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vPt_TPC_e_c_p1 ->SetName(histTitle->Data());
      mHist_vPt_TPC_e_c_p1 ->Scale(0.5);
      mHist_vPt_TPC_e_c_p1->Divide(mHistRes_Pt_c);
      mHist_vPt_TPC_e_c_p1->Divide(mHistYieldCos_ptTPCp);
      delete histTitle;
      histTitle = new TString("Flow_vPt_TPC_e_c_Seln1");
      mHist_vPt_TPC_e_c_n1 = Profile2D_Pt(mHist_vObs2D_e_c_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vPt_TPC_e_c_n1 ->SetName(histTitle->Data());
      mHist_vPt_TPC_e_c_n1 ->Scale(0.5);
      mHist_vPt_TPC_e_c_n1->Divide(mHistRes_Pt_c);
      mHist_vPt_TPC_e_c_n1->Divide(mHistYieldCos_ptTPCn);
      delete histTitle;

      histTitle = new TString("Flow_vPt_TPC_w_c_Selp1");
      mHist_vPt_TPC_w_c_p1 = Profile2D_Pt(mHist_vObs2D_w_c_p1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vPt_TPC_w_c_p1 ->SetName(histTitle->Data());
      mHist_vPt_TPC_w_c_p1 ->Scale(0.5);
      mHist_vPt_TPC_w_c_p1->Divide(mHistRes_Pt_c);
      mHist_vPt_TPC_w_c_p1->Divide(mHistYieldCos_ptTPCp);
      delete histTitle;
      histTitle = new TString("Flow_vPt_TPC_w_c_Seln1");
      mHist_vPt_TPC_w_c_n1 = Profile2D_Pt(mHist_vObs2D_w_c_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vPt_TPC_w_c_n1 ->SetName(histTitle->Data());
      mHist_vPt_TPC_w_c_n1 ->Scale(0.5);
      mHist_vPt_TPC_w_c_n1->Divide(mHistRes_Pt_c);
      mHist_vPt_TPC_w_c_n1->Divide(mHistYieldCos_ptTPCn);
      delete histTitle;

      histTitle = new TString("Flow_vPt_TPC_e_s_Selp1");
      mHist_vPt_TPC_e_s_p1 = Profile2D_Pt(mHist_vObs2D_e_s_p1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vPt_TPC_e_s_p1 ->SetName(histTitle->Data());
      mHist_vPt_TPC_e_s_p1 ->Scale(0.5);
      mHist_vPt_TPC_e_s_p1->Divide(mHistRes_Pt_s);
      mHist_vPt_TPC_e_s_p1->Divide(mHistYieldSin_ptTPCp);
      delete histTitle;
      histTitle = new TString("Flow_vPt_TPC_e_s_Seln1");
      mHist_vPt_TPC_e_s_n1 = Profile2D_Pt(mHist_vObs2D_e_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vPt_TPC_e_s_n1 ->SetName(histTitle->Data());
      mHist_vPt_TPC_e_s_n1 ->Scale(0.5);
      mHist_vPt_TPC_e_s_n1->Divide(mHistRes_Pt_s);
      mHist_vPt_TPC_e_s_n1->Divide(mHistYieldSin_ptTPCn);
      delete histTitle;

      histTitle = new TString("Flow_vPt_TPC_w_s_Selp1");
      mHist_vPt_TPC_w_s_p1 = Profile2D_Pt(mHist_vObs2D_w_s_p1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vPt_TPC_w_s_p1 ->SetName(histTitle->Data());
      mHist_vPt_TPC_w_s_p1 ->Scale(0.5);
      mHist_vPt_TPC_w_s_p1->Divide(mHistRes_Pt_s);
      mHist_vPt_TPC_w_s_p1->Divide(mHistYieldSin_ptTPCp);
      delete histTitle;
      histTitle = new TString("Flow_vPt_TPC_w_s_Seln1");
      mHist_vPt_TPC_w_s_n1 = Profile2D_Pt(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
      mHist_vPt_TPC_w_s_n1 ->SetName(histTitle->Data());
      mHist_vPt_TPC_w_s_n1 ->Scale(0.5);
      mHist_vPt_TPC_w_s_n1->Divide(mHistRes_Pt_s);
      mHist_vPt_TPC_w_s_n1->Divide(mHistYieldSin_ptTPCn);
      delete histTitle;

          TString* histTitl = new TString("Flow_vEta_x_Selp");
          mHist_vEta_x_p = Profile2D_X(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
	  mHist_vEta_x_p->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Neta;i++) {
                        float est = mHist_vEta_e_c_p1->GetBinContent(i);
                        float wst = mHist_vEta_w_c_p1->GetBinContent(i);
                        float estErr = pow(mHist_vEta_e_c_p1->GetBinError(i),2);
                        float wstErr = pow(mHist_vEta_w_c_p1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;   //Equation 6.57 and 6.58 in Statistical Data Analysis by Glen Cowan
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vEta_x_p->SetBinContent(i,content);
                        mHist_vEta_x_p->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vEta_x_Seln");
          mHist_vEta_x_n = Profile2D_X(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vEta_x_n->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Neta;i++) {
                        float est = mHist_vEta_e_c_n1->GetBinContent(i);
                        float wst = mHist_vEta_w_c_n1->GetBinContent(i);
                        float estErr = pow(mHist_vEta_e_c_n1->GetBinError(i),2);
                        float wstErr = pow(mHist_vEta_w_c_n1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vEta_x_n->SetBinContent(i,content);
                        mHist_vEta_x_n->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vEta_y_Selp");
          mHist_vEta_y_p = Profile2D_X(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vEta_y_p->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Neta;i++) {
                        float est = mHist_vEta_e_s_p1->GetBinContent(i);
                        float wst = mHist_vEta_w_s_p1->GetBinContent(i);
                        float estErr = pow(mHist_vEta_e_s_p1->GetBinError(i),2);
                        float wstErr = pow(mHist_vEta_w_s_p1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vEta_y_p->SetBinContent(i,content);
                        mHist_vEta_y_p->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vEta_y_Seln");
          mHist_vEta_y_n = Profile2D_X(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vEta_y_n->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Neta;i++) {
                        float est = mHist_vEta_e_s_n1->GetBinContent(i);
                        float wst = mHist_vEta_w_s_n1->GetBinContent(i);
                        float estErr = pow(mHist_vEta_e_s_n1->GetBinError(i),2);
                        float wstErr = pow(mHist_vEta_w_s_n1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vEta_y_n->SetBinContent(i,content);
                        mHist_vEta_y_n->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vEta_e_Selp");
          mHist_vEta_e_p = Profile2D_X(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vEta_e_p->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Neta;i++) {
                        float est = mHist_vEta_e_c_p1->GetBinContent(i);
                        float wst = mHist_vEta_e_s_p1->GetBinContent(i);
                        float estErr = pow(mHist_vEta_e_c_p1->GetBinError(i),2);
                        float wstErr = pow(mHist_vEta_e_s_p1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;      //using maximum likelihood estimation
                    if(opt_a1==1) content = wst*estErr-est*wstErr; //est*wstErr-wst*estErr;  //updated 25October
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vEta_e_p->SetBinContent(i,content);
                        mHist_vEta_e_p->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vEta_e_Seln");
          mHist_vEta_e_n = Profile2D_X(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vEta_e_n->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<Neta;i++) {
                        float est = mHist_vEta_e_c_n1->GetBinContent(i);
                        float wst = mHist_vEta_e_s_n1->GetBinContent(i);
                        float estErr = pow(mHist_vEta_e_c_n1->GetBinError(i),2);
                        float wstErr = pow(mHist_vEta_e_s_n1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                    if(opt_a1==1) content = wst*estErr-est*wstErr; //est*wstErr-wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vEta_e_n->SetBinContent(i,content);
                        mHist_vEta_e_n->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vEta_w_Selp");
          mHist_vEta_w_p = Profile2D_X(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vEta_w_p->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<Neta;i++) {
                        float est = mHist_vEta_w_c_p1->GetBinContent(i);
                        float wst = mHist_vEta_w_s_p1->GetBinContent(i);
                        float estErr = pow(mHist_vEta_w_c_p1->GetBinError(i),2);
                        float wstErr = pow(mHist_vEta_w_s_p1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                    if(opt_a1==1) content = wst*estErr-est*wstErr;// est*wstErr-wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vEta_w_p->SetBinContent(i,content);
                        mHist_vEta_w_p->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vEta_w_Seln");
          mHist_vEta_w_n = Profile2D_X(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vEta_w_n->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<Neta;i++) {
                        float est = mHist_vEta_w_c_n1->GetBinContent(i);
                        float wst = mHist_vEta_w_s_n1->GetBinContent(i);
                        float estErr = pow(mHist_vEta_w_c_n1->GetBinError(i),2);
                        float wstErr = pow(mHist_vEta_w_s_n1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                    if(opt_a1==1) content = wst*estErr-est*wstErr; //est*wstErr-wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vEta_w_n->SetBinContent(i,content);
                        mHist_vEta_w_n->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vEta_e_ch");
          mHist_vEta_e_ch = Profile2D_X(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vEta_e_ch->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<Neta;i++) {
                        float est = mHist_vEta_e_p->GetBinContent(i);
                        float wst = mHist_vEta_e_n->GetBinContent(i);
                        float estErr = pow(mHist_vEta_e_p->GetBinError(i),2);
                        float wstErr = pow(mHist_vEta_e_n->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vEta_e_ch->SetBinContent(i,content);
                        mHist_vEta_e_ch->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vEta_w_ch");
          mHist_vEta_w_ch = Profile2D_X(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vEta_w_ch->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<Neta;i++) {
                        float est = mHist_vEta_w_p->GetBinContent(i);
                        float wst = mHist_vEta_w_n->GetBinContent(i);
                        float estErr = pow(mHist_vEta_w_p->GetBinError(i),2);
                        float wstErr = pow(mHist_vEta_w_n->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vEta_w_ch->SetBinContent(i,content);
                        mHist_vEta_w_ch->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vEta_f_Selp");
          mHist_vEta_f_p = Profile2D_X(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vEta_f_p->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Neta;i++) {
                        float est = mHist_vEta_x_p->GetBinContent(i);
                        float wst = mHist_vEta_y_p->GetBinContent(i);
                        float estErr = pow(mHist_vEta_x_p->GetBinError(i),2);
                        float wstErr = pow(mHist_vEta_y_p->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                    if(opt_a1==1) content = wst*estErr-est*wstErr; //est*wstErr-wst*estErr; //updated 25October
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vEta_f_p->SetBinContent(i,content);
                        mHist_vEta_f_p->SetBinError(i,error);
                }
          histTitl = new TString("Flow_vEta_f_Seln");
          mHist_vEta_f_n = Profile2D_X(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vEta_f_n->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Neta;i++) {
                        float est = mHist_vEta_x_n->GetBinContent(i);
                        float wst = mHist_vEta_y_n->GetBinContent(i);
                        float estErr = pow(mHist_vEta_x_n->GetBinError(i),2);
                        float wstErr = pow(mHist_vEta_y_n->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                    if(opt_a1==1) content = wst*estErr-est*wstErr; //est*wstErr-wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vEta_f_n->SetBinContent(i,content);
                        mHist_vEta_f_n->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vEta_ff_Sel");
          mHist_vEta_ff = Profile2D_X(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
	  mHist_vEta_ff->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Neta;i++) {
                        float est = mHist_vEta_f_p->GetBinContent(i);
                        float wst = mHist_vEta_f_n->GetBinContent(i);
                        float estErr = pow(mHist_vEta_f_p->GetBinError(i),2);
                        float wstErr = pow(mHist_vEta_f_n->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vEta_ff->SetBinContent(i,content);
                        mHist_vEta_ff->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vPt_TPC_x_Selp");
          mHist_vPt_TPC_x_p = Profile2D_Pt(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
	  mHist_vPt_TPC_x_p->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Npt;i++) {
                        float est = mHist_vPt_TPC_e_c_p1->GetBinContent(i);
                        float wst = mHist_vPt_TPC_w_c_p1->GetBinContent(i);
                        float estErr = pow(mHist_vPt_TPC_e_c_p1->GetBinError(i),2);
                        float wstErr = pow(mHist_vPt_TPC_w_c_p1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vPt_TPC_x_p->SetBinContent(i,content);
                        mHist_vPt_TPC_x_p->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vPt_TPC_x_Seln");
          mHist_vPt_TPC_x_n = Profile2D_Pt(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vPt_TPC_x_n->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Npt;i++) {
                        float est = mHist_vPt_TPC_e_c_n1->GetBinContent(i);
                        float wst = mHist_vPt_TPC_w_c_n1->GetBinContent(i);
                        float estErr = pow(mHist_vPt_TPC_e_c_n1->GetBinError(i),2);
                        float wstErr = pow(mHist_vPt_TPC_w_c_n1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vPt_TPC_x_n->SetBinContent(i,content);
                        mHist_vPt_TPC_x_n->SetBinError(i,error);
                }
          histTitl = new TString("Flow_vPt_TPC_y_Selp");
          mHist_vPt_TPC_y_p = Profile2D_Pt(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vPt_TPC_y_p->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Npt;i++) {
                        float est = mHist_vPt_TPC_e_s_p1->GetBinContent(i);
                        float wst = mHist_vPt_TPC_w_s_p1->GetBinContent(i);
                        float estErr = pow(mHist_vPt_TPC_e_s_p1->GetBinError(i),2);
                        float wstErr = pow(mHist_vPt_TPC_w_s_p1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vPt_TPC_y_p->SetBinContent(i,content);
                        mHist_vPt_TPC_y_p->SetBinError(i,error);
                }
          histTitl = new TString("Flow_vPt_TPC_y_Seln");
          mHist_vPt_TPC_y_n = Profile2D_Pt(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vPt_TPC_y_n->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Npt;i++) {
                        float est = mHist_vPt_TPC_e_s_n1->GetBinContent(i);
                        float wst = mHist_vPt_TPC_w_s_n1->GetBinContent(i);
                        float estErr = pow(mHist_vPt_TPC_e_s_n1->GetBinError(i),2);
                        float wstErr = pow(mHist_vPt_TPC_w_s_n1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vPt_TPC_y_n->SetBinContent(i,content);
                        mHist_vPt_TPC_y_n->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vPt_TPC_f_Selp");
          mHist_vPt_TPC_f_p = Profile2D_Pt(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
	  mHist_vPt_TPC_f_p->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Npt;i++) {
                        float est = mHist_vPt_TPC_x_p->GetBinContent(i);
                        float wst = mHist_vPt_TPC_y_p->GetBinContent(i);
                        float estErr = pow(mHist_vPt_TPC_x_p->GetBinError(i),2);
                        float wstErr = pow(mHist_vPt_TPC_y_p->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                    if(opt_a1==1) content = wst*estErr-est*wstErr; //est*wstErr-wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vPt_TPC_f_p->SetBinContent(i,content);
                        mHist_vPt_TPC_f_p->SetBinError(i,error);
                }
          histTitl = new TString("Flow_vPt_TPC_f_Seln");
          mHist_vPt_TPC_f_n = Profile2D_Pt(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vPt_TPC_f_n->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Npt;i++) {
                        float est = mHist_vPt_TPC_x_n->GetBinContent(i);
                        float wst = mHist_vPt_TPC_y_n->GetBinContent(i);
                        float estErr = pow(mHist_vPt_TPC_x_n->GetBinError(i),2);
                        float wstErr = pow(mHist_vPt_TPC_y_n->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                    if(opt_a1==1) content = wst*estErr-est*wstErr; //est*wstErr-wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vPt_TPC_f_n->SetBinContent(i,content);
                        mHist_vPt_TPC_f_n->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vPt_TPC_e_Selp");
          mHist_vPt_TPC_e_p = Profile2D_Pt(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vPt_TPC_e_p->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Npt;i++) {
                        float est = mHist_vPt_TPC_e_c_p1->GetBinContent(i);
                        float wst = mHist_vPt_TPC_e_s_p1->GetBinContent(i);
                        float estErr = pow(mHist_vPt_TPC_e_c_p1->GetBinError(i),2);
                        float wstErr = pow(mHist_vPt_TPC_e_s_p1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                    if(opt_a1==1) content = wst*estErr-est*wstErr; //est*wstErr-wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vPt_TPC_e_p->SetBinContent(i,content);
                        mHist_vPt_TPC_e_p->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vPt_TPC_e_Seln");
          mHist_vPt_TPC_e_n = Profile2D_Pt(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vPt_TPC_e_n->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Npt;i++) {
                        float est = mHist_vPt_TPC_e_c_n1->GetBinContent(i);
                        float wst = mHist_vPt_TPC_e_s_n1->GetBinContent(i);
                        float estErr = pow(mHist_vPt_TPC_e_c_n1->GetBinError(i),2);
                        float wstErr = pow(mHist_vPt_TPC_e_s_n1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                    if(opt_a1==1) content = wst*estErr-est*wstErr; //est*wstErr-wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vPt_TPC_e_n->SetBinContent(i,content);
                        mHist_vPt_TPC_e_n->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vPt_TPC_w_Selp");
          mHist_vPt_TPC_w_p = Profile2D_Pt(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vPt_TPC_w_p->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Npt;i++) {
                        float est = mHist_vPt_TPC_w_c_p1->GetBinContent(i);
                        float wst = mHist_vPt_TPC_w_s_p1->GetBinContent(i);
                        float estErr = pow(mHist_vPt_TPC_w_c_p1->GetBinError(i),2);
                        float wstErr = pow(mHist_vPt_TPC_w_s_p1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                    if(opt_a1==1) content = wst*estErr-est*wstErr; //est*wstErr-wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vPt_TPC_w_p->SetBinContent(i,content);
                        mHist_vPt_TPC_w_p->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vPt_TPC_w_Seln");
          mHist_vPt_TPC_w_n = Profile2D_Pt(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vPt_TPC_w_n->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Npt;i++) {
                        float est = mHist_vPt_TPC_w_c_n1->GetBinContent(i);
                        float wst = mHist_vPt_TPC_w_s_n1->GetBinContent(i);
                        float estErr = pow(mHist_vPt_TPC_w_c_n1->GetBinError(i),2);
                        float wstErr = pow(mHist_vPt_TPC_w_s_n1->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                    if(opt_a1==1) content = wst*estErr-est*wstErr; //est*wstErr-wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vPt_TPC_w_n->SetBinContent(i,content);
                        mHist_vPt_TPC_w_n->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vPt_TPC_e_ch");
          mHist_vPt_TPC_e_ch = Profile2D_Pt(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vPt_TPC_e_ch->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Npt;i++) {
                        float est = mHist_vPt_TPC_e_p->GetBinContent(i);
                        float wst = mHist_vPt_TPC_e_n->GetBinContent(i);
                        float estErr = pow(mHist_vPt_TPC_e_p->GetBinError(i),2);
                        float wstErr = pow(mHist_vPt_TPC_e_n->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vPt_TPC_e_ch->SetBinContent(i,content);
                        mHist_vPt_TPC_e_ch->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vPt_TPC_w_ch");
          mHist_vPt_TPC_w_ch = Profile2D_Pt(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vPt_TPC_w_ch->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Npt;i++) {
                        float est = mHist_vPt_TPC_w_p->GetBinContent(i);
                        float wst = mHist_vPt_TPC_w_n->GetBinContent(i);
                        float estErr = pow(mHist_vPt_TPC_w_p->GetBinError(i),2);
                        float wstErr = pow(mHist_vPt_TPC_w_n->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vPt_TPC_w_ch->SetBinContent(i,content);
                        mHist_vPt_TPC_w_ch->SetBinError(i,error);
                }

          histTitl = new TString("Flow_vPt_TPC_ff_Sel");
          mHist_vPt_TPC_ff = Profile2D_Pt(mHist_vObs2D_w_s_n1,ptLow,ptHigh,etaLow,etaHigh,cent,rc);
          mHist_vPt_TPC_ff->SetName(histTitl->Data());
          delete histTitl;
                for(int i=1;i<=Npt;i++) {
                        float est = mHist_vPt_TPC_f_p->GetBinContent(i);
                        float wst = mHist_vPt_TPC_f_n->GetBinContent(i);
                        float estErr = pow(mHist_vPt_TPC_f_p->GetBinError(i),2);
                        float wstErr = pow(mHist_vPt_TPC_f_n->GetBinError(i),2);
                        float content = est*wstErr+wst*estErr;
                        content *= (estErr>0 && wstErr>0)? 1./(estErr+wstErr):0.;
                        float error = (estErr>0 && wstErr>0)? sqrt(1./(1./estErr+1./wstErr)):0.;
                        mHist_vPt_TPC_ff->SetBinContent(i,content);
                        mHist_vPt_TPC_ff->SetBinError(i,error);
                }


mHistYieldPart2D->Write();
mHistYieldPart2Dp->Write();
mHistYieldPart2Dn->Write();
    
mHistYieldPart2D_x->Write();
mHistYieldPart2Dp_x->Write();
mHistYieldPart2Dn_x->Write();

fout.Write();
fout.Close();
}

//-----------------------------------------------------------------------
static Double_t resEventPlaneK2(double chi) {
  // Calculates the event plane resolution as a function of chi
  //  for the case k=2.

  double con = 0.626657;                   // sqrt(pi/2)/2
  double arg = chi * chi / 4.;

  double besselOneHalf = sqrt(arg/halfpi) * sinh(arg)/arg;
  double besselThreeHalfs = sqrt(arg/halfpi) * (cosh(arg)/arg - sinh(arg)/(arg*arg));
  Double_t res = con * chi * exp(-arg) * (besselOneHalf + besselThreeHalfs);

  return res;
}

//-----------------------------------------------------------------------

static Double_t resEventPlaneK3(double chi) {
  // Calculates the event plane resolution as a function of chi
  //  for the case k=3.

  double con = 0.626657;                   // sqrt(pi/2)/2
  double arg = chi * chi / 4.;

  Double_t res = con * chi * exp(-arg) * (TMath::BesselI1(arg) +
                                          TMath::BesselI(2, arg));

  return res;
}

//-----------------------------------------------------------------------

static Double_t resEventPlaneK4(double chi) {
  // Calculates the event plane resolution as a function of chi
  //  for the case k=4.

  double con = 0.626657;                   // sqrt(pi/2)/2
  double arg = chi * chi / 4.;

  double besselOneHalf = sqrt(arg/halfpi) * sinh(arg)/arg;
  double besselThreeHalfs = sqrt(arg/halfpi) * (cosh(arg)/arg - sinh(arg)/(arg*arg));
  double besselFiveHalfs = besselOneHalf - 3*besselThreeHalfs/arg;

  Double_t res = con * chi * exp(-arg) * (besselThreeHalfs + besselFiveHalfs);

  return res;
}
//------------------------------------------------------------------------
static Double_t chi(double res) {
  // Calculates chi from the event plane resolution

  double chi   = 2.0;
  double delta = 1.0;

  for (int i = 0; i < 15; i++) {
//    chi   = (resEventPlane(chi) < res) ? chi + delta : chi - delta;
//    delta = delta / 2.;
      while(resEventPlane(chi) < res) {chi += delta;}
      delta = delta / 2.;
      while(resEventPlane(chi) > res) {chi -= delta;}
      delta = delta / 2.;
  }

  return chi;
}

//-----------------------------------------------------------------------

static Double_t resEventPlane(double chi) {
  // Calculates the event plane resolution as a function of chi

  double con = 0.626657;                   // sqrt(pi/2)/2
  double arg = chi * chi / 4.;

  Double_t res = con * chi * exp(-arg) * (TMath::BesselI0(arg) +
                                          TMath::BesselI1(arg));

  return res;
}
//-----------------------------------------------------------------------
TH1D* Profile2D_X(TProfile2D* Tp2D,Float_t Lptcut,Float_t Hptcut,Float_t Letacut,Float_t Hetacut,Int_t centr,TH1* rc){
//Updated 1April 2023
//General projection from TProfile2D to TH1D in x direction
//For example from vObs2D to vEta

    //=====================27 GeV from Zhiwan (-70<Vz<70, |Vr|<2, Vz_diff<4,nHitsFits>15, DCA<3, |eta|<1, flag>0, qaTruth>50, on MC |eta|<1 ======//  //Updated 25October for efficiency
        /*
        //pion (DCA<1, 1.05>nHitsFits/nHitsPoss >0.52)
        const float PP0[9] = {8.71218e-01,8.72933e-01,8.68838e-01,8.66543e-01,8.60677e-01,8.56335e-01,8.50640e-01,8.41993e-01,8.34163e-01};
        const float PP1[9] = {1.23429e-01,1.23299e-01,1.22216e-01,1.21815e-01,1.24344e-01,1.26645e-01,1.27493e-01,1.28687e-01,1.30046e-01};
        const float PP2[9] = {3.79632e+00,3.75484e+00,4.02690e+00,3.60593e+00,3.72778e+00,3.82028e+00,3.70317e+00,3.57753e+00,3.61064e+00};
        */
        //kaon
        const float PP0[9] = {8.35664e-01,8.31100e-01,8.34064e-01,8.34125e-01,8.29048e-01,8.30026e-01,8.24533e-01,8.18055e-01,8.18421e-01};
        const float PP1[9] = {2.02515e-01,2.02730e-01,2.04682e-01,2.04283e-01,2.01435e-01,2.03700e-01,2.03938e-01,2.05946e-01,2.04584e-01};
        const float PP2[9] = {1.95632e+00,2.13710e+00,2.05078e+00,1.94908e+00,1.96090e+00,1.95169e+00,1.95041e+00,1.94249e+00,1.94722e+00};
        /*
        //proton
        const float PP0[9] = {9.30833e-01,9.24048e-01,9.31281e-01,9.30416e-01,9.27350e-01,9.23627e-01,9.19565e-01,9.13709e-01,9.10889e-01};
        const float PP1[9] = {1.68283e-01,1.55871e-01,1.67427e-01,1.71667e-01,1.69064e-01,1.74439e-01,1.68201e-01,1.70451e-01,1.68029e-01};
        const float PP2[9] = {4.37943e+00,5.36994e+00,4.18118e+00,4.43566e+00,4.67087e+00,4.47076e+00,4.16892e+00,4.55965e+00,4.39574e+00};
        */

    
TH1D* p0=(TH1D*)Tp2D->ProjectionX("p0x",0,0);
TH1D* pp=(TH1D*)Tp2D->ProjectionY("ppy",0,0);
Int_t yLow = pp->FindBin(Lptcut+1e-5); //Updated March 14 2023
Int_t yHigh= pp->FindBin(Hptcut-1e-5); //Updated March 14 2023
Int_t xLow = p0->FindBin(Letacut+1e-5); //Updated 1April 2023
Int_t xHigh= p0->FindBin(Hetacut-1e-5); //Updated 1April 2023

    for(int i=1;i<p0->GetNbinsX()+1;i++){ //Updated 1April 2023
            p0->SetBinContent(i,0.0);
            p0->SetBinError(i,0.0);
    }
    
//Int_t xHigh= p0->GetNbinsX();
        for(int i=xLow;i<xHigh+1;i++) {
                Float_t content = 0.0,contentS=0.0,error = 0.0;
                Float_t entry = 0,sumweight2=0; //Added 28July 2023 for effective entries
                for(int j=yLow;j<yHigh+1;j++) {
                        int bin = Tp2D->GetBin(i,j);
                        Float_t binContent = Tp2D->GetBinContent(bin);
                        Float_t binEntry   = Tp2D->GetBinEntries(bin);
                        Float_t binEffectiveEntry   = Tp2D->GetBinEffectiveEntries(bin); //Added 29March23
                    Float_t binSumweight2=(binEffectiveEntry < 1e-5)? 0:binEntry*binEntry/binEffectiveEntry;//Added 28July 2023 for effective entries
                        Float_t binError   = Tp2D->GetBinError(bin);
                        
                    if(opt_eff==1){ //Added 9March to not get an error while running for cen19.v2_pion.root
                        
                        Float_t binCenter  = pp->GetBinCenter(j);
                        Float_t eff1= (centr>0)? PP0[centr-1]*exp(-pow(PP1[centr-1]/binCenter,PP2[centr-1])) : PP0[0]*exp(-pow(PP1[0]/binCenter,PP2[0])); //Updated 25October for efficiency 27GeV
                        
                        Int_t iBin = rc->FindBin(binCenter);
                        Float_t eff2 = rc->GetBinContent(iBin);
                        if(eff2==0) eff2 = 1;
                        binEntry /= (eff1*eff2);
                        //binEffectiveEntry /= eff1*eff2; //Added 29March23 does effective entry scale the same way? No effective entries is invariating under scaling the weights thus should not change, however sum of weight square does increase to keep effective entries the same
                        binSumweight2 /= (eff1*eff1*eff2*eff2); //Added 28July 2023 for effective entries
                        
                    }
                    //Float_t ss = binError*sqrt((float)binEntry);
                    Float_t ss = binError*sqrt((float)binEffectiveEntry); //Added 28July 2023 for effective entries Root uses binEffectiveEntry for TProfile error, the difference is small but good to do it properly
                    content += binContent*binEntry;
                    contentS+= (ss*ss+binContent*binContent)*binEntry;
                    entry   += (float)binEntry; //Updated int to float 28July 2023 for effective entries
                    sumweight2 +=(float) binSumweight2; //Added 28July 2023 for effective entries
                }
                if(entry < 1e-5) continue;
                content /= (float)entry;
                error = (contentS/(float)entry < content*content)? 0:sqrt(contentS/(float)entry-content*content);
                //error   /= sqrt((float)entry); //Here we use W(j) for std deviation divison and should be properly updtated see RootTProfile documentation
                error=(error*sqrt(sumweight2))/float(entry); //Root divides by BinEffectiveEntries instead of binentries which is not updated in the TProfile documentation //Updated 28July 2023 for effective entries
                p0->SetBinContent(i,content);
                p0->SetBinError(i,error);
        }
return p0;
}
//-----------------------------------------------------------------------
TH1D* Profile2D_Y(TProfile2D* Tpp2D,Float_t Lptcut,Float_t Hptcut, Float_t Letacut,Float_t Hetacut,Int_t centr, TH1* rc){ //Added Int_t centr,TH1* rc 9March 2023
    //Added 9March
    //=====================27 GeV from Zhiwan (-70<Vz<70, |Vr|<2, Vz_diff<4,nHitsFits>15, DCA<3, |eta|<1, flag>0, qaTruth>50, on MC |eta|<1 ======//  //Updated 25October for efficiency
        /*
        //pion (DCA<1, 1.05>nHitsFits/nHitsPoss >0.52)
        const float PP0[9] = {8.71218e-01,8.72933e-01,8.68838e-01,8.66543e-01,8.60677e-01,8.56335e-01,8.50640e-01,8.41993e-01,8.34163e-01};
        const float PP1[9] = {1.23429e-01,1.23299e-01,1.22216e-01,1.21815e-01,1.24344e-01,1.26645e-01,1.27493e-01,1.28687e-01,1.30046e-01};
        const float PP2[9] = {3.79632e+00,3.75484e+00,4.02690e+00,3.60593e+00,3.72778e+00,3.82028e+00,3.70317e+00,3.57753e+00,3.61064e+00};
        */
        //kaon
        const float PP0[9] = {8.35664e-01,8.31100e-01,8.34064e-01,8.34125e-01,8.29048e-01,8.30026e-01,8.24533e-01,8.18055e-01,8.18421e-01};
        const float PP1[9] = {2.02515e-01,2.02730e-01,2.04682e-01,2.04283e-01,2.01435e-01,2.03700e-01,2.03938e-01,2.05946e-01,2.04584e-01};
        const float PP2[9] = {1.95632e+00,2.13710e+00,2.05078e+00,1.94908e+00,1.96090e+00,1.95169e+00,1.95041e+00,1.94249e+00,1.94722e+00};
        /*
        //proton
        const float PP0[9] = {9.30833e-01,9.24048e-01,9.31281e-01,9.30416e-01,9.27350e-01,9.23627e-01,9.19565e-01,9.13709e-01,9.10889e-01};
        const float PP1[9] = {1.68283e-01,1.55871e-01,1.67427e-01,1.71667e-01,1.69064e-01,1.74439e-01,1.68201e-01,1.70451e-01,1.68029e-01};
        const float PP2[9] = {4.37943e+00,5.36994e+00,4.18118e+00,4.43566e+00,4.67087e+00,4.47076e+00,4.16892e+00,4.55965e+00,4.39574e+00};
        */

TH1D* p0=(TH1D*)Tpp2D->ProjectionY("p0Pt",0,0);
TH1D* pp=(TH1D*)Tpp2D->ProjectionX("ppEta",0,0);
Int_t yLow = pp->FindBin(Letacut+1e-5);    //Updated 14March
Int_t yHigh= pp->FindBin(Hetacut-1e-5);    //Updated 14March
Int_t xLow = p0->FindBin(Lptcut+1e-5); //Updated 1April 2023
Int_t xHigh= p0->FindBin(Hptcut-1e-5); //Updated 1April 2023
//Int_t xHigh= p0->GetNbinsX();

    for(int i=1;i<p0->GetNbinsX()+1;i++){ //Updated 1April 2023
            p0->SetBinContent(i,0.0);
            p0->SetBinError(i,0.0);
    }

        for(int i=xLow;i<xHigh+1;i++) {
                Float_t content = 0.0,contentS=0.0,error = 0.0;
                Float_t entry = 0, sumweight2=0.0; //Added 28July2023 for effective entries
                for(int j=yLow;j<yHigh+1;j++) {
                        int bin = Tpp2D->GetBin(j,i);
                        Float_t binContent = Tpp2D->GetBinContent(bin);
                        Float_t binEntry   = Tpp2D->GetBinEntries(bin);
                        Float_t binEffectiveEntry   = Tpp2D->GetBinEffectiveEntries(bin); //Added 29March23
                    Float_t binSumweight2=(binEffectiveEntry < 1e-5)? 0:binEntry*binEntry/binEffectiveEntry;//Added 28July 2023 for effective entries
                        Float_t binError   = Tpp2D->GetBinError(bin);

                        if(opt_eff==1){ //Added 9March
                            Float_t binCenter  = p0->GetBinCenter(i);
                            Float_t eff1= (centr>0)? PP0[centr-1]*exp(-pow(PP1[centr-1]/binCenter,PP2[centr-1])) : PP0[0]*exp(-pow(PP1[0]/binCenter,PP2[0])); //Updated 25October for efficiency
                        
                            Int_t iBin = rc->FindBin(binCenter);
                            Float_t eff2 = rc->GetBinContent(iBin);
                            if(eff2==0) eff2 = 1;
                            binEntry /= eff1*eff2;
                            binSumweight2 /= (eff1*eff1*eff2*eff2); //Added 28July 2023 for effective entries
                        
                        }

                        //Float_t ss = binError*sqrt((float)binEntry);
                        Float_t ss = binError*sqrt((float)binEffectiveEntry); //Added 28July 2023 for effective entries
                        content += binContent*binEntry;
                        contentS+= (ss*ss+binContent*binContent)*binEntry;
                        entry   += (int)binEntry;
                        sumweight2 +=(float) binSumweight2; //Added 28July 2023 for effective entries
                }
                if(entry < 1e-5) continue;
                content /= (float)entry;
                error = (contentS/(float)entry < content*content)? 0:sqrt(contentS/(float)entry-content*content);
                //error   /= sqrt((float)entry); //Here we use W(j) for std deviation divison and should be properly updtated see RootTProfile documentation
                error=(error*sqrt(sumweight2))/float(entry); //Root divides by BinEffectiveEntries instead of binentries which is not updated in the TProfile documentation //Added 28July 2023 for effective entries
                p0->SetBinContent(i,content);
                p0->SetBinError(i,error);
        }
return p0;
}
//--------------------------------------------------------------------------
TH1D* Profile2D_Pt(TProfile2D* Tpp2D,Float_t Lptcut,Float_t Hptcut, Float_t Letacut,Float_t Hetacut, Int_t centr,TH1* rc){//Added Int_t centr,TH1* rc 9March 2023
//special for vPt values flipped for negative rapidity
    
    //Added 9March
    //=====================27 GeV from Zhiwan (-70<Vz<70, |Vr|<2, Vz_diff<4,nHitsFits>15, DCA<3, |eta|<1, flag>0, qaTruth>50, on MC |eta|<1 ======//  //Updated 25October for efficiency
        /*
        //pion (DCA<1, 1.05>nHitsFits/nHitsPoss >0.52)
        const float PP0[9] = {8.71218e-01,8.72933e-01,8.68838e-01,8.66543e-01,8.60677e-01,8.56335e-01,8.50640e-01,8.41993e-01,8.34163e-01};
        const float PP1[9] = {1.23429e-01,1.23299e-01,1.22216e-01,1.21815e-01,1.24344e-01,1.26645e-01,1.27493e-01,1.28687e-01,1.30046e-01};
        const float PP2[9] = {3.79632e+00,3.75484e+00,4.02690e+00,3.60593e+00,3.72778e+00,3.82028e+00,3.70317e+00,3.57753e+00,3.61064e+00};
        */
        //kaon
        const float PP0[9] = {8.35664e-01,8.31100e-01,8.34064e-01,8.34125e-01,8.29048e-01,8.30026e-01,8.24533e-01,8.18055e-01,8.18421e-01};
        const float PP1[9] = {2.02515e-01,2.02730e-01,2.04682e-01,2.04283e-01,2.01435e-01,2.03700e-01,2.03938e-01,2.05946e-01,2.04584e-01};
        const float PP2[9] = {1.95632e+00,2.13710e+00,2.05078e+00,1.94908e+00,1.96090e+00,1.95169e+00,1.95041e+00,1.94249e+00,1.94722e+00};
        /*
        //proton
        const float PP0[9] = {9.30833e-01,9.24048e-01,9.31281e-01,9.30416e-01,9.27350e-01,9.23627e-01,9.19565e-01,9.13709e-01,9.10889e-01};
        const float PP1[9] = {1.68283e-01,1.55871e-01,1.67427e-01,1.71667e-01,1.69064e-01,1.74439e-01,1.68201e-01,1.70451e-01,1.68029e-01};
        const float PP2[9] = {4.37943e+00,5.36994e+00,4.18118e+00,4.43566e+00,4.67087e+00,4.47076e+00,4.16892e+00,4.55965e+00,4.39574e+00};
        */

TH1D* p0=(TH1D*)Tpp2D->ProjectionY("p0Pt",0,0);
TH1D* pp=(TH1D*)Tpp2D->ProjectionX("ppEta",0,0);
Int_t yLow = pp->FindBin(Letacut+1e-5);    //Updated 14March
Int_t yHigh= pp->FindBin(Hetacut-1e-5);    //Updated 14March
Int_t yZero= pp->FindBin(0);
Int_t xLow = p0->FindBin(Lptcut+1e-5); //Updated 1April 2023
Int_t xHigh= p0->FindBin(Hptcut-1e-5); //Updated 1April 2023
//Int_t xHigh= p0->GetNbinsX();

    for(int i=1;i<p0->GetNbinsX()+1;i++){ //Updated 1April 2023
            p0->SetBinContent(i,0.0);
            p0->SetBinError(i,0.0);
    }

        for(int i=xLow;i<xHigh+1;i++) {
                Float_t content = 0.0,contentS=0.0,error = 0.0;
                Float_t entry = 0,sumweight2=0; //Added 28July 2023 for effective entries
                for(int j=yLow;j<yHigh+1;j++) {
                        int bin = Tpp2D->GetBin(j,i);
                        Float_t binContent = Tpp2D->GetBinContent(bin);
                        if(j<yZero) binContent *= -1.;
                        Float_t binEntry   = Tpp2D->GetBinEntries(bin);
                        Float_t binEffectiveEntry   = Tpp2D->GetBinEffectiveEntries(bin); //Added 29March23
                    Float_t binSumweight2=(binEffectiveEntry < 1e-5)? 0:binEntry*binEntry/binEffectiveEntry;//Added 28July 2023 for effective entries

                        Float_t binError   = Tpp2D->GetBinError(bin);
                        if(opt_eff==1){ //Added 9March
                            Float_t binCenter  = p0->GetBinCenter(i);
                            Float_t eff1= (centr>0)? PP0[centr-1]*exp(-pow(PP1[centr-1]/binCenter,PP2[centr-1])) : PP0[0]*exp(-pow(PP1[0]/binCenter,PP2[0])); //Updated 25October for efficiency
                        
                            Int_t iBin = rc->FindBin(binCenter);
                            Float_t eff2 = rc->GetBinContent(iBin);
                            if(eff2==0) eff2 = 1;
                            binEntry /= eff1*eff2;
                            binSumweight2 /= (eff1*eff1*eff2*eff2); //Added 28July 2023 for effective entries
                        }
                    
                        //Float_t ss = binError*sqrt((float)binEntry);
                        Float_t ss = binError*sqrt((float)binEffectiveEntry); //Added 28July 2023 for effective entries
                        content += binContent*binEntry;
                        contentS+= (ss*ss+binContent*binContent)*binEntry;
                        entry   += (int)binEntry;
                        sumweight2 +=(float) binSumweight2; //Added 28July 2023 for effective entries
                }
                if(entry < 1e-5) continue;
                content /= (float)entry;
                error = (contentS/(float)entry < content*content)? 0:sqrt(contentS/(float)entry-content*content);
                //error   /= sqrt((float)entry);//Here we use W(j) for std deviation divison and should be properly updtated see RootTProfile documentation
                error=(error*sqrt(sumweight2))/float(entry); //Root divides by BinEffectiveEntries instead of binentries which is not updated in the TProfile documentation //Added 28July 2023 for effective entries
                p0->SetBinContent(i,content);
                p0->SetBinError(i,error);
        }
return p0;
}



