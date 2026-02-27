import numpy as np
from matplotlib import pyplot as plt
from uncertainties import unumpy, ufloat
import pandas as pd
from scipy.optimize import curve_fit
import argparse
import yaml


def find_csv(energy, paths):
    for path in paths:
        if energy in path:
            return path
    return None

plot_config = {
    'combo1':
    {
        'marker': '*',
        'color': 'C0',
        'markersize': 10,
        'zorder': 3,
        'ls': 'none',
        'capsize': 2
    },
    'combo2':
    {
        'marker': 's',
        'color': 'C1',
        'markersize': 8,
        'ls': 'none',
        'capsize': 2
    }
}

def main(lambda_v1_paths, xi_v1_paths, proton_v1_paths, kaon_v1_paths, pion_v1_paths, dv1dy_coal, energies, sys_tag):
    energies_float = np.array([float(energy.replace('GeV', '').replace('p', '.')) for energy in energies])

    cent = {'1040': [4, 5, 6], '5080': [0, 1, 2]}
    v1s = {}
    for i, energy in enumerate(energies):
        lambda_path = find_csv(energy, lambda_v1_paths)
        xi_path = find_csv(energy, xi_v1_paths)
        proton_path = find_csv(energy, proton_v1_paths)
        kaon_path = find_csv(energy, kaon_v1_paths)
        pion_path = find_csv(energy, pion_v1_paths)
        lambda_df = pd.read_csv(lambda_path)
        xi_df = pd.read_csv(xi_path)
        proton_df = pd.read_csv(proton_path)
        kaon_df = pd.read_csv(kaon_path)
        pion_df = pd.read_csv(pion_path)
        
        lambda_v1 = unumpy.uarray(lambda_df['dv1dy_lambda'], lambda_df['dv1dy_lambda_err'])
        xi_v1 = unumpy.uarray(xi_df['dv1dy_xi'], xi_df['dv1dy_xi_err'])
        proton_v1 = unumpy.uarray(proton_df['v1_p'], proton_df['v1_p_err'])
        kplus_v1 = unumpy.uarray(kaon_df['v1_p'], kaon_df['v1_p_err'])
        piplus_v1 = unumpy.uarray(pion_df['v1_p'], pion_df['v1_p_err'])
        antilambda_v1 = unumpy.uarray(lambda_df['dv1dy_lambdabar'], lambda_df['dv1dy_lambdabar_err'])
        antixi_v1 = unumpy.uarray(xi_df['dv1dy_xibar'], xi_df['dv1dy_xibar_err'])
        antiproton_v1 = unumpy.uarray(proton_df['v1_n'], proton_df['v1_n_err'])
        kminus_v1 = unumpy.uarray(kaon_df['v1_n'], kaon_df['v1_n_err'])
        piminus_v1 = unumpy.uarray(pion_df['v1_n'], pion_df['v1_n_err'])

        merged_lambda_v1 = {key: ufloat(0,0) for key in cent.keys()}
        # merged_xi_v1 = {key: ufloat(0,0) for key in cent.keys()}
        merged_proton_v1 = {key: ufloat(0,0) for key in cent.keys()}
        merged_kplus_v1 = {key: ufloat(0,0) for key in cent.keys()}
        merged_piplus_v1 = {key: ufloat(0,0) for key in cent.keys()}
        merged_antilambda_v1 = {key: ufloat(0,0) for key in cent.keys()}
        # merged_antixi_v1 = {key: ufloat(0,0) for key in cent.keys()}
        merged_antiproton_v1 = {key: ufloat(0,0) for key in cent.keys()}
        merged_kminus_v1 = {key: ufloat(0,0) for key in cent.keys()}
        merged_piminus_v1 = {key: ufloat(0,0) for key in cent.keys()}

        for top_down, (cent_label, cent_ranges) in enumerate(cent.items()):
            merged_lambda_v1[cent_label] = np.mean(lambda_v1[cent_ranges])
            # merged_xi_v1[cent_label].append(np.mean(xi_v1[cent_ranges]))
            merged_proton_v1[cent_label] = np.mean(proton_v1[cent_ranges])
            merged_kplus_v1[cent_label] = np.mean(kplus_v1[cent_ranges])
            merged_piplus_v1[cent_label] = np.mean(piplus_v1[cent_ranges])
            # merged_antixi_v1[cent_label].append(np.mean(antixi_v1[cent_ranges]))
            merged_antilambda_v1[cent_label] = np.mean(antilambda_v1[cent_ranges])
            merged_antiproton_v1[cent_label] = np.mean(antiproton_v1[cent_ranges])
            merged_kminus_v1[cent_label] = np.mean(kminus_v1[cent_ranges])
            merged_piminus_v1[cent_label] = np.mean(piminus_v1[cent_ranges])
                                                 

            if energy not in v1s.keys():
                v1s[energy] = {}
            if cent_label not in v1s[energy].keys():
                v1s[energy][cent_label] = {}
            v1s[energy][cent_label]['lambda'] = merged_lambda_v1[cent_label]
            # v1s[energy][cent_label]['xi'] = merged_xi_v1[cent_label]
            v1s[energy][cent_label]['proton'] = merged_proton_v1[cent_label]
            v1s[energy][cent_label]['kplus'] = merged_kplus_v1[cent_label]
            v1s[energy][cent_label]['piplus'] = merged_piplus_v1[cent_label]
            v1s[energy][cent_label]['antilambda'] = merged_antilambda_v1[cent_label]
            # v1s[energy][cent_label]['antixi'] = merged_antixi_v1[cent_label]
            v1s[energy][cent_label]['antiproton'] = merged_antiproton_v1[cent_label]
            v1s[energy][cent_label]['kminus'] = merged_kminus_v1[cent_label]
            v1s[energy][cent_label]['piminus'] = merged_piminus_v1[cent_label]

    # change the dict to be an array in energies, i.e., v1s[cent_label][particle]['lambda'] = unumpy.uarray(values, errors)
    # where values are the v1 values for the different energies, and errors are the errors for the v1 values
    v1s_energy = {}
    for cent_label in cent.keys():
        v1s_energy[cent_label] = {}
        for particle in ['lambda', 'proton', 'kplus', 'piplus', 'antilambda', 'antiproton', 'kminus', 'piminus']: #, 'xi', 'antixi']:
            v1s_energy[cent_label][particle] = np.array([v1s[energy][cent_label][particle] for energy in energies])

    # first comparison: antilambda vs. kminus + 1/3 antiproton (for pair-produced particles)
    fig1, ax1 = plt.subplots(1, 1, figsize=(16, 9))
    # comp_a = v1s_energy['1040']['antilambda']
    # get lambda from the coal file
    dv1dy_lambdabar_merged = unumpy.uarray(np.zeros(len(energies)), np.zeros(len(energies)))
    for i, file in enumerate(dv1dy_coal):
        with open(file, 'r') as f:
            data_dict = yaml.load(f, Loader=yaml.CLoader)
            dv1dy_lambdabar_merged[i] = ufloat(data_dict['dv1dy_lambdabar_1040']['value'], data_dict['dv1dy_lambdabar_1040']['error'])
    comp_a = dv1dy_lambdabar_merged
    print(comp_a)
    comp_b = v1s_energy['1040']['kminus'] + 1/3 * v1s_energy['1040']['antiproton']
    ax1.errorbar(energies_float/1.01, unumpy.nominal_values(comp_a), yerr=unumpy.std_devs(comp_a), label=r'$\bar{\Lambda}$', **plot_config['combo1'])
    ax1.errorbar(energies_float*1.01, unumpy.nominal_values(comp_b), yerr=unumpy.std_devs(comp_b), label=r'$K^-$ + 1/3 $\bar{p}$', **plot_config['combo2'])
    ax1.set_xlabel('Energy [GeV]')
    ax1.set_ylabel(r'$v_1$')
    ax1.set_xscale('log')
    ax1.legend()
    fig1.savefig(f'plots/sys_tag_{sys_tag}/comparison_1040_1.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_v1', type=str, nargs='+', help='paths to the csv files containing the lambda values', required=True)
    parser.add_argument('--xi_v1', type=str, nargs='+', help='paths to the csv files containing the xi values', required=True)
    parser.add_argument('--proton_v1', type=str, nargs='+', help='paths to the csv files containing the proton values', required=True)
    parser.add_argument('--kaon_v1', type=str, nargs='+', help='paths to the csv files containing the kaon values', required=True)
    parser.add_argument('--pion_v1', type=str, nargs='+', help='paths to the csv files containing the pion values', required=True)
    parser.add_argument('--input_dv1dy_coal', type=str, nargs='+')
    parser.add_argument('--energies', type=str, nargs='+', help='energies of the collisions', required=True)
    parser.add_argument('--sys_tag', type=str, help='systematic tag', required=True)
    args = parser.parse_args()
    print(args.lambda_v1)
    main(args.lambda_v1, args.xi_v1, args.proton_v1, args.kaon_v1, args.pion_v1, args.input_dv1dy_coal, args.energies, args.sys_tag)
