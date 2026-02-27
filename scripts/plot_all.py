import numpy as np
from matplotlib import pyplot as plt
from uncertainties import unumpy
import pandas as pd
import argparse


def find_csv(energy, paths):
    for path in paths:
        if energy in path:
            return path
    return None


def main(lambda_v1, proton_v1, kaon_v1, energies):
    energies_float = [float(energy.replace('GeV', '').replace('p', '.')) for energy in energies]

    lambda_v1_1040 = []
    proton_v1_1040 = []
    kaon_v1_1040 = []
    lambda_v1_5080 = []
    proton_v1_5080 = []
    kaon_v1_5080 = []

    for energy in energies:
        lambda_path = find_csv(energy, lambda_v1)
        proton_path = find_csv(energy, proton_v1)
        kaon_path = find_csv(energy, kaon_v1)
        lambda_df = pd.read_csv(lambda_path)
        proton_df = pd.read_csv(proton_path)
        kaon_df = pd.read_csv(kaon_path)
        
        delta_lambda = unumpy.uarray(lambda_df['delta_dv1dy'], lambda_df['delta_dv1dy_err'])
        delta_proton = unumpy.uarray(proton_df['delta_v1'], proton_df['delta_v1_err'])
        delta_kaon = unumpy.uarray(kaon_df['delta_v1'], kaon_df['delta_v1_err'])

        # 10-40% centrality
        lambda_v1_1040.append(np.mean(delta_lambda[4:7]))
        proton_v1_1040.append(np.mean(delta_proton[4:7]))
        kaon_v1_1040.append(np.mean(delta_kaon[4:7]))
        
        # 50-80% centrality
        lambda_v1_5080.append(np.mean(delta_lambda[:3]))
        proton_v1_5080.append(np.mean(delta_proton[:3]))
        kaon_v1_5080.append(np.mean(delta_kaon[:3]))
    
    lambda_v1_1040 = np.array(lambda_v1_1040)
    proton_v1_1040 = np.array(proton_v1_1040)
    kaon_v1_1040 = np.array(kaon_v1_1040)
    lambda_v1_5080 = np.array(lambda_v1_5080)
    proton_v1_5080 = np.array(proton_v1_5080)
    kaon_v1_5080 = np.array(kaon_v1_5080)

    combo_1040 = proton_v1_1040 - kaon_v1_1040
    combo_5080 = proton_v1_5080 - kaon_v1_5080

    fig_1040, ax_1040 = plt.subplots(figsize=(8, 6))
    fig_5080, ax_5080 = plt.subplots(figsize=(8, 6))
    ax_1040.errorbar(energies_float, unumpy.nominal_values(lambda_v1_1040), yerr=unumpy.std_devs(lambda_v1_1040), label=r'$\Lambda^0-\bar{\Lambda}^0$', fmt='o', color='C0')
    ax_1040.errorbar(energies_float, unumpy.nominal_values(combo_1040), yerr=unumpy.std_devs(combo_1040), label=r'$(p-\bar{p})-(K^+-K^-)$', fmt='o', color='C1')
    ax_1040.errorbar(energies_float, unumpy.nominal_values(proton_v1_1040), yerr=unumpy.std_devs(proton_v1_1040), label=r'$p-\bar{p}$', fmt='o', color='C2')
    ax_5080.errorbar(energies_float, unumpy.nominal_values(lambda_v1_5080), yerr=unumpy.std_devs(lambda_v1_5080), label=r'$\Lambda^0-\bar{\Lambda}^0$', fmt='o', color='C0')
    ax_5080.errorbar(energies_float, unumpy.nominal_values(combo_5080), yerr=unumpy.std_devs(combo_5080), label=r'$(p-\bar{p})-(K^+-K^-)$', fmt='o', color='C1')
    ax_5080.errorbar(energies_float, unumpy.nominal_values(proton_v1_5080), yerr=unumpy.std_devs(proton_v1_5080), label=r'$p-\bar{p}$', fmt='o', color='C2')

    ax_1040.set_xlabel('Energy (GeV)', fontsize=16)
    ax_1040.set_ylabel(r'$\Delta dv_1/dy$', fontsize=16)
    ax_1040.annotate('10-40%', xy=(0.3, 0.9), xycoords='axes fraction', fontsize=20)
    ax_1040.hlines(0, *ax_1040.get_xlim(), linestyles='dashed', color='black')
    ax_1040.legend()

    ax_5080.set_xlabel('Energy (GeV)', fontsize=16)
    ax_5080.set_ylabel(r'$\Delta dv_1/dy$', fontsize=16)
    ax_5080.annotate('50-80%', xy=(0.3, 0.9), xycoords='axes fraction', fontsize=20)
    ax_5080.hlines(0, *ax_5080.get_xlim(), linestyles='dashed', color='black')
    ax_5080.legend()
    
    # log scale x-axis
    ax_1040.set_xscale('log')
    ax_5080.set_xscale('log')

    fig_1040.savefig('plots/dv1dy_1040.pdf')
    fig_5080.savefig('plots/dv1dy_5080.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_v1', type=str, nargs='+', help='paths to the csv files containing the lambda values', required=True)
    parser.add_argument('--proton_v1', type=str, nargs='+', help='paths to the csv files containing the proton values', required=True)
    parser.add_argument('--kaon_v1', type=str, nargs='+', help='paths to the csv files containing the kaon values', required=True)
    parser.add_argument('--energies', type=str, nargs='+', help='energies of the collisions', required=True)
    args = parser.parse_args()
    main(args.lambda_v1, args.proton_v1, args.kaon_v1, args.energies)
