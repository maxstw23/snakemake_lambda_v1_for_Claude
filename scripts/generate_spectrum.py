import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from uncertainties import unumpy, ufloat
import numpy as np
# import pickle
import os 
import argparse
import yaml


def main(input_path, output_path):
    # energies = np.array([27, 19.6, 17.3, 14.6, 11.5, 9.2, 7.7])
    energies = np.array([19.6]) #, 11.5, 9.2, 7.7])
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(ncols=4, nrows=2, hspace=0, wspace=0)
    ax = gs.subplots(sharex='col', sharey='row')
    ax = ax.flatten()
    for i, energy in enumerate(energies):
        for cen in range(1, 10):
            if energy == 7.7 or energy == 9.2 or energy == 11.5 or energy == 19.6:
                dNdy = unumpy.uarray(np.zeros(20), np.zeros(20))
                y = np.linspace(-0.95, 0.95, 20)
            else:
                dNdy = unumpy.uarray(np.zeros(10), np.zeros(10))
                y = np.linspace(-0.9, 0.9, 10)
            for ybin in range(0, len(dNdy)):
                energy_str = f'{energy}GeV'.replace('.', 'p')
                if energy == 27.0:
                    energy_str = '27GeV'
                cen_str = f'cen{cen}'
                y_str = f'y{ybin*0.1-1:.1f}{(ybin+1)*0.1-1:.1f}'
                replaced_energy_str = input_path.split('_')[-4]
                replaced_cen_str = input_path.split('_')[-2]
                replaced_y_str = input_path.split('_')[-1].split('_')[-1].split('.yaml')[0]
                file_path = input_path.replace(replaced_energy_str, energy_str).replace(replaced_cen_str, cen_str).replace(replaced_y_str, y_str)
                with open(file_path, 'r') as f:
                    data_dict = yaml.load(f, Loader=yaml.CLoader)
                dNdy[ybin] = ufloat(data_dict['S'], data_dict['S_error'])
            ax[i].errorbar(y, unumpy.nominal_values(dNdy), yerr=unumpy.std_devs(dNdy), fmt='*')
        ax[i].annotate(f'{energy}GeV', xy=(0.05, 0.9), xycoords='axes fraction')
        ax[i].set_xlabel(r'$y$')
        ax[i].set_ylabel(r'$dN/dy$')
    plt.savefig(output_path, format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png')
    plt.savefig(output_path.replace('.pdf', '.eps'), format='eps')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a spectrum plot')
    parser.add_argument('--input_path', type=str, required=True, nargs='+', help='Path to the input file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
    args = parser.parse_args()
    main(args.input_path[0], args.output_path)
