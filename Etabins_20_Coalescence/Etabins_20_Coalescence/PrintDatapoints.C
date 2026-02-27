//To be used in the folder containing the root files with systematics
#include <iostream>
#include <fstream> // Include this header for file operations

void PrintDatapoints(char energy[300]="9GeV",char particlename[300]="pions"){
    //char energy[300]="9GeV";
    //char particlename[300]="pions";
    char printhistname[300]="Flow_vEta_f_Selp_13_rebinned"; //to print at the terminal, all histogram values are written to the text file
    
    char outfilename[300];
    snprintf(outfilename,sizeof(outfilename),"txtfiles/%s_%s_datapoints.txt",energy,particlename);
    std::ofstream outtextfile(outfilename);
    
    char infilename[300];
    char currenthistname[300];
    TH1D* hist;
    
    snprintf(infilename,sizeof(infilename),"%s_%s_withsystematics.root",energy,particlename);
    // Define the input files and the name of the histogram
    TFile* file=new TFile(infilename,"READ");
    
    /*
    Centrality(%) cen(used in code)             Centrality(%)  cen(used in code)
    90-100          0                               30-40           5
    80-90           0                               20-30           6
    70-80           1                               10-20           7
    60-70           2                               5-10            8
    50-60           3                               0-5             9
    40-50           4
    */
    
    
    std::vector<std::string> hist_names = {"Flow_vEta_f_Selp_13_rebinned", "Flow_vEta_f_Seln_13_rebinned", "Deltav1_vEta_13_rebinned","Flow_vEta_f_Selp_14_rebinned", "Flow_vEta_f_Seln_14_rebinned", "Deltav1_vEta_14_rebinned","Flow_vEta_f_Selp_57_rebinned", "Flow_vEta_f_Seln_57_rebinned", "Deltav1_vEta_57_rebinned","Flow_vEta_f_Selp_89_rebinned", "Flow_vEta_f_Seln_89_rebinned", "Deltav1_vEta_89_rebinned",
        "Flow_vPt_TPC_f_Selp_13_rebinned", "Flow_vPt_TPC_f_Seln_13_rebinned", "Deltav1_vPt_TPC_13_rebinned","Flow_vPt_TPC_f_Selp_14_rebinned", "Flow_vPt_TPC_f_Seln_14_rebinned", "Deltav1_vPt_TPC_14_rebinned","Flow_vPt_TPC_f_Selp_57_rebinned", "Flow_vPt_TPC_f_Seln_57_rebinned", "Deltav1_vPt_TPC_57_rebinned","Flow_vPt_TPC_f_Selp_89_rebinned", "Flow_vPt_TPC_f_Seln_89_rebinned", "Deltav1_vPt_TPC_89_rebinned",
        "v1_vCent_Selp_linear","v1_vCent_Seln_linear","deltav1_vCent_linear",
        "v1_vCent_Selp_cubic","v1_vCent_Seln_cubic","deltav1_vCent_cubic",
        "v1slopes_linear_combinedcent_pos","v1slopes_linear_combinedcent_neg","v1slopes_linear_combinedcent_deltav1",
        "v1slopes_cubic_combinedcent_pos","v1slopes_cubic_combinedcent_neg","v1slopes_cubic_combinedcent_deltav1",
        
    };
    
    for (const auto& hist_name : hist_names){
        //cout<<"histname="<<hist_name<<endl;
        
        snprintf(currenthistname,sizeof(currenthistname),"%s_bincenters",hist_name.c_str());
        hist = (TH1D*) file->Get(currenthistname);
        if(hist){
            if(hist_name==printhistname){
                cout<<"double "<<currenthistname<<"["<<hist->GetNbinsX()<<"]={";
                for(int i=1;i<=hist->GetNbinsX();i++){
                    if(i!=1){cout<<",";}
                    cout<<(hist->GetBinContent(i));
                }
                cout<<"};"<<endl;
            }
            outtextfile<<"double "<<currenthistname<<"["<<hist->GetNbinsX()<<"]={";
            for(int i=1;i<=hist->GetNbinsX();i++){
                if(i!=1){outtextfile<<",";}
                outtextfile<<(hist->GetBinContent(i));
            }
            outtextfile<<"};"<<endl;
        }
        snprintf(currenthistname,sizeof(currenthistname),"%s",hist_name.c_str());
        hist = (TH1D*) file->Get(currenthistname);
        if(hist){
            if(hist_name==printhistname){
                cout<<"double "<<currenthistname<<"["<<hist->GetNbinsX()<<"]={";
                for(int i=1;i<=hist->GetNbinsX();i++){
                    if(i!=1){cout<<",";}
                    cout<<(hist->GetBinContent(i));
                }
                cout<<"};"<<endl;
            }
            
            outtextfile<<"double "<<currenthistname<<"["<<hist->GetNbinsX()<<"]={";
            for(int i=1;i<=hist->GetNbinsX();i++){
                if(i!=1){outtextfile<<",";}
                outtextfile<<(hist->GetBinContent(i));
            }
            outtextfile<<"};"<<endl;
        }
        
        snprintf(currenthistname,sizeof(currenthistname),"%s",hist_name.c_str());
        hist = (TH1D*) file->Get(currenthistname);
        if(hist){
            if(hist_name==printhistname){
                cout<<"double "<<currenthistname<<"_err["<<hist->GetNbinsX()<<"]={";
                for(int i=1;i<=hist->GetNbinsX();i++){
                    if(i!=1){cout<<",";}
                    cout<<(hist->GetBinError(i));
                }
                cout<<"};"<<endl;
            }
            outtextfile<<"double "<<currenthistname<<"_err["<<hist->GetNbinsX()<<"]={";
            for(int i=1;i<=hist->GetNbinsX();i++){
                if(i!=1){outtextfile<<",";}
                outtextfile<<(hist->GetBinError(i));
            }
            outtextfile<<"};"<<endl;
        }
        
        snprintf(currenthistname,sizeof(currenthistname),"%s_systematics",hist_name.c_str());
        hist = (TH1D*) file->Get(currenthistname);
        if(hist){
            if(hist_name==printhistname){
                cout<<"double "<<currenthistname<<"["<<hist->GetNbinsX()<<"]={";
                for(int i=1;i<=hist->GetNbinsX();i++){
                    if(i!=1){cout<<",";}
                    cout<<(hist->GetBinContent(i));
                }
                cout<<"};"<<endl<<endl;
            }
            outtextfile<<"double "<<currenthistname<<"["<<hist->GetNbinsX()<<"]={";
            for(int i=1;i<=hist->GetNbinsX();i++){
                if(i!=1){outtextfile<<",";}
                outtextfile<<(hist->GetBinContent(i));
            }
            outtextfile<<"};"<<endl<<endl;
        }
    }
}
