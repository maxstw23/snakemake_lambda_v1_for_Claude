import numpy as np
from matplotlib import pyplot as plt
from uncertainties import unumpy
import pandas as pd
from scipy.optimize import curve_fit
import argparse


def find_csv(energy, paths):
    for path in paths:
        if energy in path:
            return path
    return None


def plane_fit(x, c0, c1, c2):
    delta_u, delta_s = x
    return c0 + c1 * delta_u + c2 * delta_s


def main(lambda_v1_paths, xi_v1_paths, proton_v1_paths, kaon_v1_paths, pion_v1_paths, energies, sys_tag):
    energies_float = [float(energy.replace('GeV', '').replace('p', '.')) for energy in energies]

    cent = {'1040': [4, 5, 6], '4080': [0, 1, 2, 3]}
    cent_xi = {'1040': {'7p7GeV': [1], '9p2GeV': [2,3,4], '11p5GeV': [3,4,5], '14p6GeV': [3,4,5], '17p3GeV': [3,4,5], '19p6GeV': [3,4,5], '27GeV': [3,4,5]},
               '4080': {'7p7GeV': [0], '9p2GeV': [0,1], '11p5GeV': [0,1,2], '14p6GeV': [0,1,2], '17p3GeV': [0,1,2], '19p6GeV': [0,1,2], '27GeV': [0,1,2]}}

    fig_diff = plt.figure(figsize=(15, 12))
    gs_diff = fig_diff.add_gridspec(2, 3, hspace=0, wspace=0)
    ax_diff = gs_diff.subplots(sharex='col', sharey='row')
    diff_type = [r'$\Delta N_{\Delta u}$', r'$\Delta N_{\Delta d}$', r'$\Delta N_{\Delta s}$']
    combo_diff_num = {1: [0,0,0], 2: [-1,0,1], 3: [0,-1,1], 4: [0,0,0]} # delta_u, delta_d, delta_s
    combo_markers = {1: 'o', 2: 's', 3: 'D', 4: '^'}

    combos = {}
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
        
        delta_lambda = unumpy.uarray(lambda_df['delta_dv1dy'], lambda_df['delta_dv1dy_err'])
        delta_xi = unumpy.uarray(xi_df['delta_dv1dy'], xi_df['delta_dv1dy_err'])
        delta_proton = unumpy.uarray(proton_df['delta_v1'], proton_df['delta_v1_err'])
        delta_kaon = unumpy.uarray(kaon_df['delta_v1'], kaon_df['delta_v1_err'])
        delta_pion = unumpy.uarray(pion_df['delta_v1'], pion_df['delta_v1_err'])
        
        lambda_v1 = {key: [] for key in cent.keys()}
        xi_v1 = {key: [] for key in cent.keys()}
        proton_v1 = {key: [] for key in cent.keys()}
        kaon_v1 = {key: [] for key in cent.keys()}
        pion_v1 = {key: [] for key in cent.keys()}

        for top_down, (cent_label, cent_ranges) in enumerate(cent.items()):
            lambda_v1[cent_label].append(np.mean(delta_lambda[cent_ranges]))
            xi_v1[cent_label].append(np.mean(delta_xi[cent_xi[cent_label][energy]]))
            proton_v1[cent_label].append(np.mean(delta_proton[cent_ranges]))
            kaon_v1[cent_label].append(np.mean(delta_kaon[cent_ranges]))
            pion_v1[cent_label].append(np.mean(delta_pion[cent_ranges]))

            lambda_v1[cent_label] = np.array(lambda_v1[cent_label])
            xi_v1[cent_label] = np.array(xi_v1[cent_label])
            proton_v1[cent_label] = np.array(proton_v1[cent_label])
            kaon_v1[cent_label] = np.array(kaon_v1[cent_label])
            pion_v1[cent_label] = np.array(pion_v1[cent_label])
        
            if energy not in combos.keys():
                combos[energy] = {}
            if cent_label not in combos[energy].keys():
                combos[energy][cent_label] = {}
            combos[energy][cent_label][1] = lambda_v1[cent_label] - proton_v1[cent_label] + kaon_v1[cent_label]
            combos[energy][cent_label][2] = lambda_v1[cent_label] - proton_v1[cent_label] # -1 delta u
            combos[energy][cent_label][3] = lambda_v1[cent_label] - proton_v1[cent_label] + pion_v1[cent_label] # -1 delta d
            combos[energy][cent_label][4] = xi_v1[cent_label] - lambda_v1[cent_label] + kaon_v1[cent_label]
                    
    for i, cent_label in enumerate(cent.keys()):
        for j, delta in enumerate(diff_type):
            ax_diff[i][j].set_xlabel(delta, fontsize=16)
            ax_diff[i][j].set_xlim(-1.5, 1.5)
            if i == 0:
                ax_diff[i][j].set_ylim(-0.015, 0.015)
            else:
                ax_diff[i][j].set_ylim(-0.1, 0.1)
            if i == 1:
                ax_diff[i][j].set_xticks([-1, 0, 1])
            ax_diff[i][j].hlines(0, -1.5, 1.5, linestyles='dashed', color='black')

            lowest = {}
            for k, energy in enumerate(energies):
                shift_for_energy = 0.06 / (len(energies)-1) * k - 0.03
                for combo in combos[energy][cent_label].keys():
                    x = combo_diff_num[combo][j]
                    y = unumpy.nominal_values(combos[energy][cent_label][combo])
                    yerr = unumpy.std_devs(combos[energy][cent_label][combo])

                    if combo not in lowest.keys():
                        lowest[combo] = y - yerr
                    elif y - yerr < lowest[combo]:
                        lowest[combo] = y - yerr
                    shift_for_combo = 0.3 / (len(combos[energy][cent_label].keys())-1) * (combo-1) - 0.15
                    kw = {'label': f'{energy}', 'fmt': combo_markers[combo]} if combo == 1 else {'fmt': combo_markers[combo]}
                    ax_diff[i][j].errorbar(x+shift_for_energy+shift_for_combo, y, yerr=yerr, **kw, color=f'C{k}')
            
            # draw arrows pointing to the lowest point in each combo from below that says which combo it is
            cur_height = ax_diff[i][j].get_ylim()[1] - ax_diff[i][j].get_ylim()[0]
            for combo in combos[energy][cent_label].keys():
                shift_for_combo = 0.2 / (len(combos[energy][cent_label].keys())-1) * (combo-1) - 0.1
                x = combo_diff_num[combo][j] + shift_for_combo
                y = lowest[combo]
                arrow_height = 0.1 * cur_height
                ax_diff[i][j].annotate(f'Combo {combo}', (x, y), xytext=(x, y-arrow_height), 
                                       horizontalalignment='center', arrowprops=dict(arrowstyle='->', lw=1), fontsize=10)
    
    ax_diff[0][0].set_ylabel(r'$\Delta v_1/dy$', fontsize=16)
    ax_diff[0][0].legend(fontsize=16)
    ax_diff[0][0].annotate('10-40%', xy=(0.3, 0.9), xycoords='axes fraction', fontsize=20)
    ax_diff[1][0].annotate('50-80%', xy=(0.3, 0.9), xycoords='axes fraction', fontsize=20)
    # handlers = []
    # for combo, marker in combo_markers.items():
    #     handlers.append(ax_diff[1][0].errorbar(-2, 0, fmt=marker, label=f'Combo {combo}', color='black'))
    # ax_diff[1][0].legend(handles=handlers, fontsize=16)
    fig_diff.tight_layout()
    fig_diff.savefig(f'plots/sys_tag_{sys_tag}/other_coal.pdf')

    # Plane fit using Least Squares
    C0 = np.zeros_like(energies_float)
    C_delta_u = np.zeros_like(energies_float)
    C_delta_s = np.zeros_like(energies_float)
    C0_err = np.zeros_like(energies_float)
    C_delta_u_err = np.zeros_like(energies_float)
    C_delta_s_err = np.zeros_like(energies_float)
    fig_plane = plt.figure(figsize=(15, 12))
    for i, energy in enumerate(energies):
        ax_plane = fig_plane.add_subplot(2, 4, i+1, projection='3d')
        cent_label = '1040'
        delta_u = np.array([delta_q[0] for _, delta_q in combo_diff_num.items()])
        delta_d = np.array([delta_q[1] for _, delta_q in combo_diff_num.items()])
        delta_s =  np.array([delta_q[2] for _, delta_q in combo_diff_num.items()])
        delta_v1 =  np.array([unumpy.nominal_values(delta)[0] for _, delta in combos[energy][cent_label].items()])
        delta_v1_err =  np.array([unumpy.std_devs(delta)[0] for _, delta in combos[energy][cent_label].items()])
        
        popt, pcov = curve_fit(plane_fit, [delta_u, delta_s], delta_v1, sigma=delta_v1_err)
        C0[i] = popt[0]
        C_delta_u[i] = popt[1]
        C_delta_s[i] = popt[2]
        C0_err[i] = np.sqrt(pcov[0,0])
        C_delta_u_err[i] = np.sqrt(pcov[1,1])
        C_delta_s_err[i] = np.sqrt(pcov[2,2])

         ### QA check
        print(f'Energy: {energy}')
        print(f'delta_u: {delta_u}')
        print(f'delta_d: {delta_d}')
        print(f'delta_u+delta_d: {delta_u+delta_d}')
        print(f'delta_s: {delta_s}')
        print(f'delta_v1: {delta_v1}')
        print(f'delta_v1_err: {delta_v1_err}')
        print(f'popt: {popt}')
        print(f'pcov: {np.diag(pcov**0.5)}') 
        print('---------------------------------')

        ax_plane.scatter(delta_u, delta_s, delta_v1, label=energy)
        ax_plane.plot_trisurf(delta_u, delta_s, plane_fit([delta_u, delta_s], *popt), alpha=0.5)       

        ax_plane.set_xlabel(r'$\Delta u$', fontsize=16)
        ax_plane.set_ylabel(r'$\Delta s$', fontsize=16)
        ax_plane.set_zlabel(r'$\Delta v_1/dy$', fontsize=16)   

    fig_coeff, ax_coeff = plt.subplots(3, 1, figsize=(5, 12))
    ax_coeff[0].errorbar(energies_float, C0, yerr=C0_err, fmt='o')
    ax_coeff[0].set_ylabel(r'$C_0$', fontsize=16)
    ax_coeff[1].errorbar(energies_float, C_delta_u, yerr=C_delta_u_err, fmt='o')
    ax_coeff[1].set_ylabel(r'$C_{\Delta ud}$', fontsize=16)
    ax_coeff[2].errorbar(energies_float, C_delta_s, yerr=C_delta_s_err, fmt='o')
    ax_coeff[2].set_ylabel(r'$C_{\Delta s}$', fontsize=16)
    ax_coeff[2].set_xlabel('Energy (GeV)', fontsize=16)
    fig_coeff.tight_layout()

    # plt.show()
    fig_plane.savefig(f'plots/sys_tag_{sys_tag}/plane_fit.pdf')
    fig_coeff.savefig(f'plots/sys_tag_{sys_tag}/plane_coeff.pdf')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_v1', type=str, nargs='+', help='paths to the csv files containing the lambda values', required=True)
    parser.add_argument('--xi_v1', type=str, nargs='+', help='paths to the csv files containing the xi values', required=True)
    parser.add_argument('--proton_v1', type=str, nargs='+', help='paths to the csv files containing the proton values', required=True)
    parser.add_argument('--kaon_v1', type=str, nargs='+', help='paths to the csv files containing the kaon values', required=True)
    parser.add_argument('--pion_v1', type=str, nargs='+', help='paths to the csv files containing the pion values', required=True)
    parser.add_argument('--energies', type=str, nargs='+', help='energies of the collisions', required=True)
    parser.add_argument('--sys_tag', type=str, help='systematic tag', required=True)
    args = parser.parse_args()
    print(args.lambda_v1)
    main(args.lambda_v1, args.xi_v1, args.proton_v1, args.kaon_v1, args.pion_v1, args.energies, args.sys_tag)
