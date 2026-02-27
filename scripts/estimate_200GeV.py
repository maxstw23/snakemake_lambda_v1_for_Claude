import numpy as np
from uncertainties import unumpy

def main():
    dv1dy_proton_5080 = np.array([-0.00580853,-0.00598084,-0.00516893]) # ,-0.00335092])
    dv1dy_proton_5080_err = np.array([0.00107524,0.000494364,0.000298284]) # ,0.000219541])
    dv1dy_antiproton_5080 = np.array([-0.0027339,-0.00328934,-0.00368366]) # ,-0.00458922])
    dv1dy_antiproton_5080_err = np.array([0.00136472,0.000646522,0.000389539]) # ,0.000316338])
    dv1dy_proton_5080_merged = np.sum(dv1dy_proton_5080 / dv1dy_proton_5080_err**2) / np.sum(1./dv1dy_proton_5080_err**2)
    dv1dy_proton_5080_merged_err = 1./np.sqrt(np.sum(1./dv1dy_proton_5080_err**2))
    dv1dy_antiproton_5080_merged = np.sum(dv1dy_antiproton_5080 / dv1dy_antiproton_5080_err**2) / np.sum(1./dv1dy_antiproton_5080_err**2)
    dv1dy_antiproton_5080_merged_err = 1./np.sqrt(np.sum(1./dv1dy_antiproton_5080_err**2))
    delta_proton_5080 = dv1dy_proton_5080_merged - dv1dy_antiproton_5080_merged
    delta_proton_5080_err = np.sqrt(dv1dy_proton_5080_merged_err**2 + dv1dy_antiproton_5080_merged_err**2)

    dv1dy_kplus_5080 = np.array([-0.00476589,-0.00419217,-0.00376621]) # ,-0.00336885])
    dv1dy_kplus_5080_err = np.array([0.000801743,0.000380646,0.000225057]) # ,0.000158374])
    dv1dy_kminus_5080 = np.array([-0.0037113,-0.00318118,-0.00363695]) # ,-0.00389323])
    dv1dy_kminus_5080_err = np.array([0.000992968,0.000439613,0.000254668]) # ,0.000225585])
    dv1dy_kplus_5080_merged = np.sum(dv1dy_kplus_5080 / dv1dy_kplus_5080_err**2) / np.sum(1./dv1dy_kplus_5080_err**2)
    dv1dy_kplus_5080_merged_err = 1./np.sqrt(np.sum(1./dv1dy_kplus_5080_err**2))
    dv1dy_kminus_5080_merged = np.sum(dv1dy_kminus_5080 / dv1dy_kminus_5080_err**2) / np.sum(1./dv1dy_kminus_5080_err**2)
    dv1dy_kminus_5080_merged_err = 1./np.sqrt(np.sum(1./dv1dy_kminus_5080_err**2))
    delta_kplus_5080 = dv1dy_kplus_5080_merged - dv1dy_kminus_5080_merged
    delta_kplus_5080_err = np.sqrt(dv1dy_kplus_5080_merged_err**2 + dv1dy_kminus_5080_merged_err**2)

    dv1dy_piplus_5080 = np.array([-0.00421209,-0.00402106,-0.00366061])
    dv1dy_piplus_5080_err = np.array([0.000233335,0.000111523,6.79151e-05])
    dv1dy_piminus_5080 = np.array([-0.00462907,-0.00435163,-0.00386902])
    dv1dy_piminus_5080_err = np.array([0.000280556,0.000123905,7.60611e-05])
    dv1dy_piplus_5080_merged = np.sum(dv1dy_piplus_5080 / dv1dy_piplus_5080_err**2) / np.sum(1./dv1dy_piplus_5080_err**2)
    dv1dy_piplus_5080_merged_err = 1./np.sqrt(np.sum(1./dv1dy_piplus_5080_err**2))
    dv1dy_piminus_5080_merged = np.sum(dv1dy_piminus_5080 / dv1dy_piminus_5080_err**2) / np.sum(1./dv1dy_piminus_5080_err**2)
    dv1dy_piminus_5080_merged_err = 1./np.sqrt(np.sum(1./dv1dy_piminus_5080_err**2))
    delta_pion_5080 = dv1dy_piplus_5080_merged - dv1dy_piminus_5080_merged
    delta_pion_5080_err = np.sqrt(dv1dy_piplus_5080_merged_err**2 + dv1dy_piminus_5080_merged_err**2)

    combo = delta_proton_5080 - delta_kplus_5080
    combo_err = np.sqrt(delta_proton_5080_err**2 + delta_kplus_5080_err**2)
    print(f'Delta dv1dy for protons in 50-80% centrality: {delta_proton_5080:.6f}' + rf'$\pm${delta_proton_5080_err:.6f}')
    print(f'Delta dv1dy for kaons in 50-80% centrality: {delta_kplus_5080:.6f}' + rf'$\pm${delta_kplus_5080_err:.6f}')
    print(f'Delta dv1dy for pions in 50-80% centrality: {delta_pion_5080:.6f}' + rf'$\pm${delta_pion_5080_err:.6f}')
    print(f'Delta dv1dy for protons - kaons in 50-80% centrality: {combo:.6f}' + rf'$\pm${combo_err:.6f}')
    # significance
    print(f"Significance for delta proton - kaon in 50-80% centrality: {abs(combo) / combo_err:.2f}")

if __name__ == "__main__":
    main()