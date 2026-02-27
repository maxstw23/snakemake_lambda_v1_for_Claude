import numpy as np
from matplotlib import pyplot as plt
import uproot
# import pickle
import yaml
import platform
import scipy.odr as odr
from scipy.optimize import curve_fit
from iminuit import cost, Minuit
from uncertainties import unumpy
import pandas as pd
import argparse
from matplotlib.patches import Rectangle


def find_csv(particle, flow, paths):
    for path in paths:
        if f'fit_{particle}_{flow}' in path:
            return path
    return None


def find_csv_piKp(particle, paths):
    for path in paths:
        if particle in path:
            return path
    return None


def main(paths, paths_piKp, path_lambda, fres, output, method, **kwargs):
    print(f'paths: {paths}')
    if method == 1:
        fun = func_wrapper
        range = 'full'
    # elif method == 'curve_fit':
    #     fun = func # need to switch argument, so obselete for now
    elif method == 3:
        fun = func_wrapper_3rd
        range = 'full'

    if kwargs["sys_tag"] == '6':
        method = 3 # use 3rd order polynomial for fitting as a systematic check
        fun = func_wrapper_3rd
    if kwargs["sys_tag"] == '5':
        range = 'half' # use only the positive y range for fitting as a systematic check

    # centrality merging
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.CLoader)

    # centrality bins
    centralities = np.array([75, 65, 55, 45, 35, 25, 15, 7.5, 2.5])
    centralities_bins = np.array([80, 70, 60, 50, 40, 30, 20, 10, 5, 0])

    # prefix for output path
    prefix = 'special_' if float(kwargs["sys_tag"]) >= 5 else ""
    fres_root = uproot.open(fres)
    cen_correspondence = {5: '10-40%', 1: '40-80%'}
    hres = unumpy.uarray(fres_root['hEPDEP_ew_cos_1'].values(), fres_root['hEPDEP_ew_cos_1'].errors())
    # mask out centralities where the resolution^2 is negative
    # Also mask out the 70-80% at 7.7 GeV
    cen_mask = unumpy.nominal_values(hres) > 0
    # cen_mask[0] = False
    print(f'cen_mask: {cen_mask}')
    hres_ct = fres_root['hEPDEP_ew_cos_1'].counts()
    resolution = np.where(cen_mask, abs(hres) ** 0.5, 0)
    fig_res, ax_res = plt.subplots(figsize=(8, 6))

    # 0.42 - 1.8
    xi_v1_df = pd.read_csv(find_csv('Xi', 'v1', paths), header=[0, 1], index_col=0)
    xibar_v1_df = pd.read_csv(find_csv('Xibar', 'v1', paths), header=[0, 1], index_col=0)
    # xi_a1_df = pd.read_csv(find_csv('Xi', 'a1', paths), header=[0, 1], index_col=0)
    # xibar_a1_df = pd.read_csv(find_csv('Xibar', 'a1', paths), header=[0, 1], index_col=0)

    xi_v1 = {}
    xibar_v1 = {}
    # xi_a1 = {}
    # xibar_a1 = {}
    # merge the centrality bins
    merge_guide = config['merged_xi_cen_bins'][kwargs['energy']]
    # however, we need to adjust the bins to merge as sometimes there are nan values for some centralities
    # if there is a nan value for v1 (not a1), we should remove this centrality from the guide
    for cen in xi_v1_df.columns.levels[0]:
        if np.isnan(xi_v1_df.loc[:, (cen, 'values')].astype(float)).any():
            for merged_bins in merge_guide:
                if float(cen) in merged_bins:
                    merged_bins.remove(float(cen))
        if np.isnan(xibar_v1_df.loc[:, (cen, 'values')].astype(float)).any():
            for merged_bins in merge_guide:
                if float(cen) in merged_bins:
                    merged_bins.remove(float(cen))
    print(f'updated merge_guide: {merge_guide}')
    new_centralities = []
    new_centralities_bins = []
    new_cen_mask = []
    for cen_index, merged_bins in enumerate(merge_guide):
        if cen_index == 0:
            new_centralities_bins.append(centralities_bins[merged_bins[0]-1])
        merged_xi_v1 = unumpy.uarray(np.zeros(xi_v1_df.shape[0]), np.zeros(xi_v1_df.shape[0]))
        merged_xibar_v1 = merged_xi_v1.copy()
        merged_xi_a1 = merged_xi_v1.copy()
        merged_xibar_a1 = merged_xi_v1.copy()
        # also shift centrality bins when merging
        new_centralities.append((centralities[merged_bins[0]-1] + centralities[merged_bins[-1]-1])*0.5)
        new_centralities_bins.append(centralities_bins[merged_bins[-1]])
        new_cen_mask.append(np.any(cen_mask[[cen - 1 for cen in merged_bins]]))
        for cen in merged_bins:
            merged_xi_v1 += unumpy.uarray(xi_v1_df.loc[:, (str(cen), 'values')], xi_v1_df.loc[:, (str(cen), 'errors')])
            merged_xibar_v1 += unumpy.uarray(xibar_v1_df.loc[:, (str(cen), 'values')], xibar_v1_df.loc[:, (str(cen), 'errors')])
            # merged_xi_a1 += unumpy.uarray(xi_a1_df.loc[:, (str(cen), 'values')], xi_a1_df.loc[:, (str(cen), 'errors')])
            # merged_xibar_a1 += unumpy.uarray(xibar_a1_df.loc[:, (str(cen), 'values')], xibar_a1_df.loc[:, (str(cen), 'errors')])
        xi_v1[cen_index+1] = {'values': unumpy.nominal_values(merged_xi_v1), 'errors': unumpy.std_devs(merged_xi_v1)}
        xibar_v1[cen_index+1] = {'values': unumpy.nominal_values(merged_xibar_v1), 'errors': unumpy.std_devs(merged_xibar_v1)}
        # xi_a1[cen_index+1] = {'values': unumpy.nominal_values(merged_xi_a1), 'errors': unumpy.std_devs(merged_xi_a1)}
        # xibar_a1[cen_index+1] = {'values': unumpy.nominal_values(merged_xibar_a1), 'errors': unumpy.std_devs(merged_xibar_a1)}
    centralities = np.array(new_centralities)
    centralities_bins = np.array(new_centralities_bins)
    cen_mask = np.array(new_cen_mask)
    print(f'centralities: {centralities}')
    print(f'centralities_bins: {centralities_bins}')
    print(f'cen_mask: {cen_mask}')

    # for cen in xi_v1_df.columns.levels[0]:
    #     xi_v1[int(cen)] = {s: xi_v1_df.loc[:, (cen, s)].values for s in ['values', 'errors']}
    #     xibar_v1[int(cen)] = {s: xibar_v1_df.loc[:, (cen, s)].values for s in ['values', 'errors']}
    #     xi_a1[int(cen)] = {s: xi_a1_df.loc[:, (cen, s)].values for s in ['values', 'errors']}
    #     xibar_a1[int(cen)] = {s: xibar_a1_df.loc[:, (cen, s)].values for s in ['values', 'errors']}
    
    # Plotting
    num_ybin = len(xi_v1[1]['values'])
    ybin_edges = np.linspace(-1., 1., num_ybin + 1)
    ybin = 0.5 * (ybin_edges[:-1] + ybin_edges[1:])      
    ybin_err = np.diff(ybin_edges) / 2
    ybin_unumpy = unumpy.uarray(ybin, ybin_err)
    # ax_res.errorbar(centralities[cen_mask], unumpy.nominal_values(resolution[cen_mask]), xerr=0, yerr=unumpy.std_devs(resolution[cen_mask]),
    #                 fmt='d', color='k', capsize=2, ms=8)
    # # ax_res.set_ylim(0, 0.5)
    # ax_res.set_xlabel('Centrality %', fontsize=18, loc='right')
    # ax_res.set_ylabel(r'Resolution', fontsize=18, loc='top')

    ### v1
    fig_1, ax_1 = plt.subplots(3, 3, figsize=(16, 9))
    # xi
    dv1dy_xi = np.zeros(len(centralities))
    dv1dy_xi_err = np.zeros(len(centralities))
    for cen, data in xi_v1.items():
        if cen_mask[cen - 1] == False:
            continue
        num_ybin = len(data['values'])
        ybin_edges = np.linspace(-1., 1., num_ybin + 1)
        ybin = 0.5 * (ybin_edges[:-1] + ybin_edges[1:])      
        ybin_err = np.diff(ybin_edges) / 2
        ybin_unumpy = unumpy.uarray(ybin, ybin_err)

        bool_nan = np.isnan(data['values'].astype(float))
        v1 = data['values'][np.invert(bool_nan)]
        v1_err = data['errors'][np.invert(bool_nan)]
        v1_unumpy = unumpy.uarray(v1, v1_err)
        v1_final = v1_unumpy / resolution[cen - 1]
        ybin_good = ybin_unumpy[np.invert(bool_nan)]
        ax_1[(cen - 1) // 3, (cen - 1) % 3].errorbar(unumpy.nominal_values(ybin_good), unumpy.nominal_values(v1_final),
                                                     xerr=0, yerr=unumpy.std_devs(v1_final),
                                                     fmt='o', color='C0', capsize=2, label=r'$\Xi$'
                                                     )
        ax_1[(cen - 1) // 3, (cen - 1) % 3].hlines(0., -1., 1., color='k', linestyle='--')
        ax_1[(cen - 1) // 3, (cen - 1) % 3].set_xlabel(r'$y$')
        ax_1[(cen - 1) // 3, (cen - 1) % 3].set_ylabel(r'$v_1$')
        ax_1[(cen - 1) // 3, (cen - 1) % 3].annotate(f'{centralities_bins[cen]}-{centralities_bins[cen - 1]}%',
                                                     xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)

        # fitting
        print(f'fitting {cen} Xi v1')
        popt, dv1dy_xi_err[cen - 1] = fit(ybin_good, v1_final, method=method, range=range)
        dv1dy_xi[cen - 1] = popt[0]
        x_fit = np.linspace(-1, 1, 101)
        ax_1[(cen - 1) // 3, (cen - 1) % 3].plot(x_fit, fun(popt, x_fit), '-', color='C0')
        ax_1[(cen - 1) // 3, (cen - 1) % 3].legend()

    # xibar
    dv1dy_xibar = np.zeros(len(centralities))
    dv1dy_xibar_err = np.zeros(len(centralities))
    for cen, data in xibar_v1.items():
        if cen_mask[cen - 1] == False:
            continue
        num_ybin = len(data['values'])
        ybin_edges = np.linspace(-1., 1., num_ybin + 1)
        ybin = 0.5 * (ybin_edges[:-1] + ybin_edges[1:])      
        ybin_err = np.diff(ybin_edges) / 2
        ybin_unumpy = unumpy.uarray(ybin, ybin_err)

        bool_nan = np.isnan(data['values'].astype(float))
        v1 = data['values'][np.invert(bool_nan)]
        v1_err = data['errors'][np.invert(bool_nan)]
        v1_unumpy = unumpy.uarray(v1, v1_err)
        v1_final = v1_unumpy / resolution[cen - 1]
        ybin_good = ybin_unumpy[np.invert(bool_nan)]
        ax_1[(cen - 1) // 3, (cen - 1) % 3].errorbar(unumpy.nominal_values(ybin_good), unumpy.nominal_values(v1_final),
                                                     xerr=0, yerr=unumpy.std_devs(v1_final),
                                                     fmt='o', color='C1', capsize=2, label=r'$\bar{\Xi}$'
                                                     )
        ax_1[(cen - 1) // 3, (cen - 1) % 3].set_xlabel(r'$y$')
        ax_1[(cen - 1) // 3, (cen - 1) % 3].set_ylabel(r'$v_1$')
        ax_1[(cen - 1) // 3, (cen - 1) % 3].set_ylim(-0.1, 0.1)

        # fitting
        print(f'fitting {cen} Xibar v1')
        popt, dv1dy_xibar_err[cen - 1] = fit(ybin_good, v1_final, method=method, range=range)
        dv1dy_xibar[cen - 1] = popt[0]
        x_fit = np.linspace(-1, 1, 101)
        ax_1[(cen - 1) // 3, (cen - 1) % 3].plot(x_fit, fun(popt, x_fit), '-', color='C1')
        ax_1[(cen - 1) // 3, (cen - 1) % 3].legend()

    # slopes
    fig_3, ax_3 = plt.subplots(1, 1, figsize=(8, 6))
    ax_3.errorbar(centralities[cen_mask], dv1dy_xi[cen_mask], yerr=dv1dy_xi_err[cen_mask], fmt='o', color='black', capsize=2,
                  label=r'$\Xi$')
    ax_3.errorbar(centralities[cen_mask], dv1dy_xibar[cen_mask], yerr=dv1dy_xibar_err[cen_mask], fmt='o', color='red', capsize=2,
                  label=r'$\bar{\Xi}$')
    ax_3.set_xlabel('Centrality (%)')
    ax_3.set_ylabel(r'$dv_1/dy$')
    ax_3.legend()

    # # xi a1
    # fig_2, ax_2 = plt.subplots(3, 3, figsize=(16, 9))
    # da1dy_xi = np.zeros(len(centralities))
    # da1dy_xi_err = np.zeros(len(centralities))
    # for cen, data in xi_a1.items():
    #     if cen_mask[cen - 1] == False:
    #         continue
    #     num_ybin = len(data['values'])
    #     ybin_edges = np.linspace(-1., 1., num_ybin + 1)
    #     ybin = 0.5 * (ybin_edges[:-1] + ybin_edges[1:])      
    #     ybin_err = np.diff(ybin_edges) / 2
    #     ybin_unumpy = unumpy.uarray(ybin, ybin_err)

    #     bool_nan = np.isnan(data['values'].astype(float))
    #     a1 = data['values'][np.invert(bool_nan)]
    #     a1_err = data['errors'][np.invert(bool_nan)]
    #     a1_unumpy = unumpy.uarray(a1, a1_err)
    #     a1_final = a1_unumpy / resolution[cen - 1]
    #     ybin_good = ybin_unumpy[np.invert(bool_nan)]
    #     ax_2[(cen - 1) // 3, (cen - 1) % 3].errorbar(unumpy.nominal_values(ybin_good), unumpy.nominal_values(a1_final),
    #                                                  xerr=0, yerr=unumpy.std_devs(a1_final),
    #                                                  fmt='o', color='C0', capsize=2, label=r'$\Xi$'
    #                                                  )
    #     ax_2[(cen - 1) // 3, (cen - 1) % 3].hlines(0., -1., 1., color='k', linestyle='--')
    #     ax_2[(cen - 1) // 3, (cen - 1) % 3].set_xlabel(r'$y$')
    #     ax_2[(cen - 1) // 3, (cen - 1) % 3].set_ylabel(r'$a_1$')
    #     ax_2[(cen - 1) // 3, (cen - 1) % 3].annotate(f'{centralities_bins[cen]}-{centralities_bins[cen - 1]}%',
    #                                                  xy=(0.05, 0.85), xycoords='axes fraction', fontsize=20)

    #     # fitting
    #     print(f'fitting {cen} Xi a1')
    #     popt, da1dy_xi_err[cen - 1] = fit(ybin_good, a1_final, method=method, range=range)
    #     da1dy_xi[cen - 1] = popt[0]
    #     x_fit = np.linspace(-1, 1, 101)
    #     ax_2[(cen - 1) // 3, (cen - 1) % 3].plot(x_fit, fun(popt, x_fit), '-', color='C0')
    #     ax_2[(cen - 1) // 3, (cen - 1) % 3].legend()

    # # xibar a1
    # da1dy_xibar = np.zeros(len(centralities))
    # da1dy_xibar_err = np.zeros(len(centralities))
    # for cen, data in xibar_a1.items():
    #     if cen_mask[cen - 1] == False:
    #         continue
    #     num_ybin = len(data['values'])
    #     ybin_edges = np.linspace(-1., 1., num_ybin + 1)
    #     ybin = 0.5 * (ybin_edges[:-1] + ybin_edges[1:])      
    #     ybin_err = np.diff(ybin_edges) / 2
    #     ybin_unumpy = unumpy.uarray(ybin, ybin_err)
        
    #     bool_nan = np.isnan(data['values'].astype(float))
    #     a1 = data['values'][np.invert(bool_nan)]
    #     a1_err = data['errors'][np.invert(bool_nan)]
    #     a1_unumpy = unumpy.uarray(a1, a1_err)
    #     a1_final = a1_unumpy / resolution[cen - 1]
    #     ybin_good = ybin_unumpy[np.invert(bool_nan)]
    #     ax_2[(cen - 1) // 3, (cen - 1) % 3].errorbar(unumpy.nominal_values(ybin_good), unumpy.nominal_values(a1_final),
    #                                                  xerr=0, yerr=unumpy.std_devs(a1_final),
    #                                                  fmt='o', color='C1', capsize=2, label=r'$\bar{\Xi}$'
    #                                                  )
    #     ax_2[(cen - 1) // 3, (cen - 1) % 3].set_xlabel(r'$y$')
    #     ax_2[(cen - 1) // 3, (cen - 1) % 3].set_ylabel(r'$a_1$')
    #     ax_2[(cen - 1) // 3, (cen - 1) % 3].set_ylim(-0.1, 0.1)

    #     # fitting
    #     print(f'fitting {cen} Xibar a1')
    #     popt, da1dy_xibar_err[cen - 1] = fit(ybin_good, a1_final, method=method, range=range)
    #     da1dy_xibar[cen - 1] = popt[0] 
    #     x_fit = np.linspace(-1, 1, 101)
    #     ax_2[(cen - 1) // 3, (cen - 1) % 3].plot(x_fit, fun(popt, x_fit), '-', color='C1')
    #     ax_2[(cen - 1) // 3, (cen - 1) % 3].legend()


    # delta v1 and a1 slope
    delta_dv1dy = dv1dy_xi - dv1dy_xibar
    delta_dv1dy_err = np.sqrt(dv1dy_xi_err ** 2 + dv1dy_xibar_err ** 2)
    # delta_da1dy = da1dy_xi - da1dy_xibar
    # delta_da1dy_err = np.sqrt(da1dy_xi_err ** 2 + da1dy_xibar_err ** 2)
    fig_final, ax_final = plt.subplots(1,1, figsize=(8, 6))
    ax_final.errorbar(centralities[cen_mask] - 0.5, delta_dv1dy[cen_mask], fmt='o', yerr=delta_dv1dy_err[cen_mask], ls='none', label=r'$v_1$')
    # ax_final.errorbar(centralities[cen_mask] + 0.5, delta_da1dy[cen_mask], fmt='o', yerr=delta_da1dy_err[cen_mask], ls='none', label=r'$a_1$')
    ax_final.hlines(y=0, xmin=0, xmax=80, linestyle='--', color='black')
    ax_final.set_xlabel('Centrality (%)', loc='right', fontsize=18)
    ax_final.set_ylabel(r'$\Delta dv_1/dy$' + r'($\Delta da_1/dy$)', loc='top', fontsize=18)
    if 'yrange' in kwargs.keys():
        ax_final.set_ylim(*kwargs['yrange'])
    else:
        ax_final.set_ylim(-0.2, 0.15)
    ax_final.annotate(r'$\Xi^{-}-\bar{\Xi}^{+}$', xy=(0.1, 0.2), xycoords='axes fraction', fontsize=25)
    ax_final.annotate(kwargs['energy'].replace('p', '.'), xy=(0.1, 0.1), xycoords='axes fraction', fontsize=25)
    ax_final.legend(loc='best', fontsize=12)
    plt.tight_layout()

    # compare with coal prediction
    lambda_df = pd.read_csv(path_lambda)
    kaon_df = pd.read_csv(find_csv_piKp('kaons', paths_piKp))
    proton_df = pd.read_csv(find_csv_piKp('protons', paths_piKp))
    delta_lambda = unumpy.uarray(lambda_df['delta_dv1dy'], lambda_df['delta_dv1dy_err'])
    delta_kaon = unumpy.uarray(kaon_df['delta_v1'], kaon_df['delta_v1_err'])
    delta_proton = unumpy.uarray(proton_df['delta_v1'], proton_df['delta_v1_err'])
    # merge the centrality bins
    merged_delta_lambda = unumpy.uarray(np.zeros(len(centralities)), np.zeros(len(centralities)))
    merged_delta_kaon = unumpy.uarray(np.zeros(len(centralities)), np.zeros(len(centralities)))
    merged_delta_proton = unumpy.uarray(np.zeros(len(centralities)), np.zeros(len(centralities)))
    for cen, merged_bins in enumerate(merge_guide):
        for cen_bin in merged_bins:
            merged_delta_lambda[cen] += delta_lambda[cen_bin - 1]
            merged_delta_kaon[cen] += delta_kaon[cen_bin - 1]
            merged_delta_proton[cen] += delta_proton[cen_bin - 1]
    delta_lambda = merged_delta_lambda
    delta_kaon = merged_delta_kaon
    delta_proton = merged_delta_proton
    combo = delta_lambda - delta_kaon
    combo2 = delta_proton - 2 * delta_kaon
    fig_coal, ax_coal = plt.subplots(1, 1, figsize=(8, 6))
    ax_coal.errorbar(centralities[cen_mask]+0.5, delta_dv1dy[cen_mask], yerr=delta_dv1dy_err[cen_mask], fmt='o', label=r'$\Xi^{-}-\bar{\Xi}^{+}$')
    ax_coal.errorbar(centralities[cen_mask]-0.5, unumpy.nominal_values(combo[cen_mask]), yerr=unumpy.std_devs(combo[cen_mask]), fmt='o', label=r'$\Lambda^{0}-K^{+}$')
    ax_coal.hlines(y=0, xmin=0, xmax=80, linestyle='--', color='black')
    ax_coal.set_xlabel('Centrality (%)', loc='right', fontsize=18)
    ax_coal.set_ylabel(r'$\Delta dv_1/dy$', loc='top', fontsize=18)
    ax_coal.legend(loc='best', fontsize=12)
    plt.tight_layout()

    # saving the output
    data_dict = {'x': centralities[cen_mask], 'y': delta_dv1dy[cen_mask], 'yerr': delta_dv1dy_err[cen_mask],
                 'combo': unumpy.nominal_values(combo[cen_mask]), 'combo_err': unumpy.std_devs(combo[cen_mask]),
                 'combo2': unumpy.nominal_values(combo2[cen_mask]), 'combo2_err': unumpy.std_devs(combo2[cen_mask])}
    with open(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/paper_yaml/dv1dy_coal_{kwargs["energy"]}_xi.yaml', 'w') as file:
        yaml.dump(data_dict, file, default_flow_style=False)
                
    ### convert the output to panda and csv
    data = pd.DataFrame({'centrality': centralities, 
                         'dv1dy_xi': dv1dy_xi, 'dv1dy_xi_err': dv1dy_xi_err,
                         'dv1dy_xibar': dv1dy_xibar, 'dv1dy_xibar_err': dv1dy_xibar_err,
                         'delta_dv1dy': delta_dv1dy, 'delta_dv1dy_err': delta_dv1dy_err})
                         # 'da1dy_xi': da1dy_xi, 'da1dy_xi_err': da1dy_xi_err,
                         # 'da1dy_xibar': da1dy_xibar, 'da1dy_xibar_err': da1dy_xibar_err,
                         #Q 'delta_da1dy': delta_da1dy, 'delta_da1dy_err': delta_da1dy_err})
    data.to_csv(output, index=False)

    plt.tight_layout()
    # plt.show()
    fig_1.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/v1_cen_{kwargs["energy"]}_xi.pdf')
    # fig_2.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/a1_cen_{kwargs["energy"]}_xi.pdf')
    fig_final.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/dv1a1dy_{kwargs["energy"]}_xi.pdf')
    fig_coal.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/dv1dy_coal_{kwargs["energy"]}_xi.pdf')

    # also png
    fig_1.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/v1_cen_{kwargs["energy"]}_xi.png')
    # fig_2.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/a1_cen_{kwargs["energy"]}_xi.png')
    fig_final.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/dv1a1dy_{kwargs["energy"]}_xi.png')
    fig_coal.savefig(f'plots/{prefix}sys_tag_{kwargs["sys_tag"]}/dv1dy_coal_{kwargs["energy"]}_xi.png')


def func_wrapper(params, x):
    return params[0] * x


def fit_odr(x, y):
    model = odr.Model(func_wrapper)
    data = odr.RealData(unumpy.nominal_values(x), unumpy.nominal_values(y),
                        sx=unumpy.std_devs(x), sy=unumpy.std_devs(y))
    odr_instance = odr.ODR(data, model, beta0=[0.], maxit=100)
    odr_result = odr_instance.run()
    return odr_result.beta, odr_result.sd_beta[0]


def func_wrapper_3rd(params, x):
    return params[0] * x + params[1] * x ** 3


def fit_odr_3rd(x, y):
    model = odr.Model(func_wrapper_3rd)
    data = odr.RealData(unumpy.nominal_values(x), unumpy.nominal_values(y),
                        sx=unumpy.std_devs(x), sy=unumpy.std_devs(y))
    odr_instance = odr.ODR(data, model, beta0=[0., 1.], maxit=100)
    odr_result = odr_instance.run()
    return odr_result.beta, odr_result.sd_beta[0]


def func(x, a):
    return a * x


def func_3rd(x, a, b):
    return a * x + b * x ** 3


def fit_iminuit(x, y):
    c = cost.LeastSquares(unumpy.nominal_values(x), unumpy.nominal_values(y), unumpy.std_devs(y), func)
    m = Minuit(c, a=0)
    m.simplex()
    print(m.migrad())
    if not m.valid:
        raise RuntimeError('Fit did not converge')
    return [m.values['a']], m.errors['a']


def fit_iminuit_3rd(x, y):
    c = cost.LeastSquares(unumpy.nominal_values(x), unumpy.nominal_values(y), unumpy.std_devs(y), func_3rd)
    m = Minuit(c, a=0, b=0.1)
    m.migrad()
    if not m.valid:
        raise RuntimeError('Fit did not converge')
    return [m.values['a'],m.values['b']], m.errors['a']


def fit_curve_fit(x, y):
    popt, pcov = curve_fit(func, xdata=unumpy.nominal_values(x), ydata=unumpy.nominal_values(y),
                           p0=[0.], sigma=unumpy.std_devs(y))
    return popt, np.sqrt(pcov[0][0])


def fit_curve_fit_3rd(x, y):
    popt, pcov = curve_fit(func_3rd, xdata=unumpy.nominal_values(x), ydata=unumpy.nominal_values(y),
                           p0=[0, 0.1], sigma=unumpy.std_devs(y))
    # check if fit converges, if not throw an error
    if np.isinf(pcov).any():
        raise RuntimeError('Fit did not converge')    
    return popt, np.sqrt(pcov[0][0])


def fit(x, y, method, range):
    fit = fit_iminuit
    if method == 3:
        fit = fit_iminuit_3rd
    if range == 'half': # positive half
        mask = x > 0
        x = x[mask]
        y = y[mask]
    return fit(x, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, nargs='+', help='Paths to the four csv files containing the v1 and a1 val', required=True)
    parser.add_argument('--fres', type=str, help='Path to the file containing the resolution histograms', required=True)
    parser.add_argument('--paths_piKp', type=str, nargs='+', help='Paths to the csv files containing the v1 and a1 val of pi, K and p', required=True)
    parser.add_argument('--path_lambda', type=str, help='Path to the csv file containing the v1 and a1 val of lambda', required=True)
    parser.add_argument('--output', type=str, help='Path to the output txt file', required=True)
    parser.add_argument('--sys_tag', type=str, help='Systematic tag. "0" is default cut', default=0)
    parser.add_argument('--method', type=int, help='Method to use for fitting', default=1)
    parser.add_argument('--energy', type=str, help='Energy of the collision, as a string', default='7p7GeV')
    parser.add_argument('--yrange', type=float, nargs=2, help='Y range for the final plot', default=None)
    args = parser.parse_args()
    # select only non-required arguments and group them in a dictionnary
    kwd = {k: v for k, v in vars(args).items() if k not in ['paths', 'paths_piKp', 'path_lambda', 'fres', 'output']}
    main(paths=args.paths, paths_piKp=args.paths_piKp, path_lambda=args.path_lambda, fres=args.fres, output=args.output, **kwd)
