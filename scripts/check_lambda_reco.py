import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import argparse

num_ptbins = 30
num_ybins = 20

def main(data_files, output_file):
    files = {}
    num_events = {}
    for data_file in data_files:
        key = data_file.split('/')[1]   
        files[key] = uproot.open(data_file)
        num_events[key] = files[key]['centrality'].counts()[1:]
    
    fig_cen_pt, ax_cen_pt = plt.subplots(3,3, figsize=(15,15))
    for i in range(9):
        for key in files.keys():
            ratio = np.zeros(num_ptbins)
            for j in range(num_ptbins):
                total = np.sum([np.sum(files[key][f'hLambdaM_cen_y_pt_{i}_{k}_{j}'].counts()) for k in range(num_ybins)])
                ratio[j] = total/num_events[key][i]
            ptbin_width = 4./num_ptbins
            ptbin_centers = np.linspace(ptbin_width/2., 4.-ptbin_width/2., num_ptbins)
            ax_cen_pt[i//3, i%3].plot(ptbin_centers, ratio, label=key)
            ax_cen_pt[i//3, i%3].set_xlabel(r'$p_T$ (GeV/c)')
            ax_cen_pt[i//3, i%3].set_ylabel(r'$N_{\Lambda}$ / $N_{event}$')
            ax_cen_pt[i//3, i%3].legend()

    fig_cen_y, ax_cen_y = plt.subplots(3,3, figsize=(15,15))
    for i in range(9):
        for key in files.keys():
            ratio = np.zeros(num_ybins)
            for j in range(num_ybins):
                total = np.sum([np.sum(files[key][f'hLambdaM_cen_y_pt_{i}_{j}_{k}'].counts()) for k in range(num_ptbins)])
                ratio[j] = total/num_events[key][i]
            ybin_width = 2./num_ybins
            ybin_centers = np.linspace(-1.+ybin_width/2., 1.-ybin_width/2., num_ybins)
            ax_cen_y[i//3, i%3].plot(ybin_centers, ratio, label=key)
            ax_cen_y[i//3, i%3].set_xlabel(r'$y$')
            ax_cen_y[i//3, i%3].set_ylabel(r'$N_{\Lambda}$ / $N_{event}$')
            ax_cen_y[i//3, i%3].legend()
        
    figs = [fig_cen_pt, fig_cen_y]
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_file)
    for fig in figs:
        pdf.savefig(fig)
    pdf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_files', type=str, help='Data files', nargs='+', required=True)
    parser.add_argument('--output_file', type=str, help='Output file', required=True)
    args = parser.parse_args()
    main(args.data_files, args.output_file)