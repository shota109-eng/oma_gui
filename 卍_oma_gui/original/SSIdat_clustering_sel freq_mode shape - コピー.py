import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from datetime import datetime
from pyoma2.algorithms import SSIdat
from pyoma2.setup import SingleSetup
from pyoma2.functions import gen

#====================================================================================================================
# setting
#====================================================================================================================

# import data from csv file path
folder_path =r'C:\Users\yokoe\Downloads'
file_names = [
    'acc1_06.03_141650 (120s).csv',
    'acc2_06.03_141650 (120s).csv',
    'acc3_06.03_141650 (120s).csv',
]
usecols = [1, 2, 3]
start_time = None

# import geometry file from csv file path
geo = r"C:\Users\yokoe\Downloads\HC_vault1_1.2.3.xlsx"

# save the file
folder = r"D:\case_study\belltower\result\06.03_141650 (120s)\SSIdat_clustering_sel freq_mode shape"
new_folder_name = "SSIdat-clustering"
save_folder = os.path.join(folder, folder)
os.makedirs(save_folder, exist_ok=True)

# select output method, choose "h"(high) or "l"(low)
mpl = 'h'
pyvista = 'l'

# setup SSI algorithm parameter
fs = 200
freqlim = (0,50) # display frequency range
# setup SSIdat, SSIcov, p=LSCF parameter
br = 40 # br: Number of block datas in the Hankel matrix
ordmax = 120 # ordmax: Maximum model order for the analysis
ordmin = 0 # ordmin: Minimum model order for the analysis
step = 1 # Step size for iterating through model order_out
# sc Soft criteria for the SSI analysis
# hc Hard criteria for the SSI analysis
calc_unc=True # calc_unc: Whether to calculate uncertainty
nb = 100 # Number of bootstrap samples to use for uncertainty calculations
hide_poles = False

# Hierarchical clustering
percentile = 80
min_cluster_elements = 10

# mode shape paremeter
sel_freq = None # list [2.81, 2.55] or None: The frequency of the mode to be extracted (if None, not mpe)
rtol = 0.01 # Relative tolerance for comparing identified frequencies with the selected ones
sel_freqlim = (0,10) # tuple (1,10) or None: The frequency range of the mode to be extracted (if None, not mpe from plot)
order_in = None # ordde_in: Specified model order(s) for which the modal parameters are to be extracted

# setup Detrend, filter and decimate parameter
q = 0 # the decimation factor (if 0, not decimation)
Wn = (0.1) # the critical frequency or frequencies
order = 0 # the order of the filter (if 0, not filtering)
btype = "highpass" # choose filter type from "lowpass", "highpass", "bandpass"or "bandstop"


#====================================================================================================================
# run module
#====================================================================================================================

# create single setup
if start_time:
    start_time_obj = datetime.strptime(start_time, "%H:%M:%S")
    df_data = []
    for name in file_names:
        folder_path = os.path.join(folder_path, name)
        df = pd.read_csv(folder_path, usecols=usecols, header=0)
        time_col = df.iloc[:, 3].astype(str)
        match_index = None
        for idx, t in time_col.items():
            try:
                t_obj = datetime.strptime(t.strip(), "%H:%M:%S")
                if t_obj >= start_time_obj:
                    match_index = idx
                    break
            except:
                continue
        if match_index is not None:
            df_cut = df.iloc[match_index:, :3]
            df_cut = df_cut.reset_index(drop=True)
            df_data.append(df_cut)
    comb_data = pd.concat(df_data, axis=1)
    if comb_data.isnull().any(axis=1).any():
        nan_start_index = comb_data.isnull().any(axis=1).idxmax()
        combined_data = comb_data.iloc[:nan_start_index]
    else:
        combined_data = comb_data
    ss = SingleSetup(combined_data, fs=fs)
else:
    df_data = []
    for name in file_names:
        file_name = os.path.join(folder_path, name)
        df = pd.read_csv(file_name, usecols=usecols)
        df_clean = df.dropna(how='all')
        df_data.append(df_clean)
    comb_data = pd.concat(df_data, axis=1)
    if comb_data.isnull().any(axis=1).any():
        nan_start_index = comb_data.isnull().any(axis=1).idxmax()
        combined_data = comb_data.iloc[:nan_start_index]
    else:
        combined_data = comb_data
    ss = SingleSetup(combined_data, fs=fs)

# Detrend, freq_ilter or decimate
ss.detrend_data()
if q:
    ss.decimate_data(q=q)

if order:
    ss.freq_ilter_data(Wn=Wn, order=order, btype=btype)

# Initialise the algorithms
ssidat = SSIdat(name="SSIdat", br=br, ordmax=ordmax, calc_unc=calc_unc)
ss.add_algorithms(ssidat)
ss.run_by_name("SSIdat")
algoRes = ssidat.result


#====================================================================================================================
# output the result
#====================================================================================================================

# output result (figs and parameters) ===================================================================
fig, ax = ssidat.plot_stab(freqlim=freqlim, hide_poles=hide_poles)
output_ssidat_fig = os.path.join(save_folder, "SSIdat_stab_"+os.path.splitext(os.path.basename(folder_path))[0]+".png")
plt.savefig(output_ssidat_fig, dpi=300, bbox_inches='tight')
fig, ax = ssidat.plot_freqvsdamp(freqlim=freqlim, hide_poles=hide_poles)
output_ssidat_fig = os.path.join(save_folder, "SSIdat_freq-damp_"+os.path.splitext(os.path.basename(folder_path))[0]+".png")
plt.savefig(output_ssidat_fig, dpi=300, bbox_inches='tight')


y_label_twin = 'Model Order'
Fn = ssidat.result.Fn_poles
Xi = ssidat.result.Xi_poles
Phi = ssidat.result.Phi_poles
Lab = ssidat.result.Lab
step = ssidat.run_params.step
Fn_std = ssidat.result.Fn_poles_std
Fns_stab = np.where(Lab == 1, Fn, np.nan)
Fns_unstab = np.where(Lab == 0, Fn, np.nan)
order_out = ssidat.result.order_out

#====================================================================================================================
# Hierarchical clustering
#====================================================================================================================

# only stable poles
def extract_valid_poles(Fn, Phi,Lab):
    poles_list = []
    num_poles, num_models = Fn.shape

    for indx in range(num_poles):
        for n in range(num_models):
            if Lab[indx, n] == 1:
                freq = Fn[indx, n]
                if not np.isnan(freq):
                    phi = Phi[indx, n, :]  # shape: (3,)
                    poles_list.append((indx, n, freq, phi))
    
    return poles_list

"""
# unstable and stable poles
def extract_valid_poles(Fn, Phi):
    poles_list = []
    num_poles, num_poles = Fn.shape

    for n in range(num_poles):
        for i in range(num_poles):
            freq = Fn[n, i]
            if not np.isnan(freq):
                phi = Phi[n, i, :]  # shape: (3,)
                poles_list.append((n, i, freq, phi))
    
    return poles_list
"""  

def compute_min_distances(poles_list):
    result_list = []
    grouped = {}

    for indx, n, freq, phi in poles_list: # group poles every model order
        grouped.setdefault(n, []).append({'indx': indx, 'freq': freq, 'phi': phi})

    for nk in sorted(grouped.keys()): 
        if nk == 0:
            continue

        poles_k = grouped[nk]
        poles_km1 = grouped.get(nk - 1, [])
        if not poles_km1:
            continue

        for pi in poles_k:
            freq_i, phi_i = pi['freq'], pi['phi']
            min_d = np.inf
            min_match = None

            for pj in poles_km1:
                freq_j, phi_j = pj['freq'], pj['phi']
                mac = gen.MAC(phi_i, phi_j)
                dist_ij = abs(freq_i - freq_j) / freq_i + (1 - mac) 
                if dist_ij < min_d:
                    min_d = dist_ij
                    min_match = {'nk': nk, 'indx': pi['indx'], 'freq_k': freq_i,
                                'nk_1': nk - 1, 'j': pj['indx'], 'freq_km1': freq_j,
                                'mac': mac, 'distance': dist_ij}
            result_list.append(min_match)

    return result_list

def compute_distance_matrix(poles_list):
    N = len(poles_list)
    distance_matrix = np.zeros((N, N))

    for i in range(N):
        _, _, freq_i, phi_i = poles_list[i]
        for j in range(i + 1, N):
            _, _, freq_j, phi_j = poles_list[j]
            mac = gen.MAC(phi_i, phi_j)
            dist = abs(freq_i - freq_j) / freq_i + (1 - mac)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix

def plot_Hclus(Z, threshold, zoom_ylim=None):
    fig = plt.figure(figsize=(12, 5))
    dendrogram(Z, no_labels=True, color_threshold=threshold)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.2f}')
    plt.title('Hierarchical Clustering (Single Linkage)')
    plt.xlabel('Element labels')
    plt.ylabel('Distance')
    if zoom_ylim is not None:
        plt.ylim(zoom_ylim)
    plt.grid(True)
    plt.tight_layout()
    return fig


#====================================================================================================================
# output
#====================================================================================================================

if hide_poles:
    x = Fns_stab.flatten(order="F")
    y = np.array([i // len(Fns_stab) for i in range(len(x))]) * step
    x1 = None
    y1 = None
else:
    x = Fns_stab.flatten(order="f")
    y = np.array([i // len(Fns_stab) for i in range(len(x))]) * step
    x1 = Fns_unstab.flatten(order="f")
    y1 = np.array([i // len(Fns_unstab) for i in range(len(x))]) * step

# stabilization csv file ======================================================
outnm_stab=("SSIdat_stab_"+os.path.splitext(os.path.basename(folder_path))[0]+".csv")
data_stab = []
if hide_poles:
    if Fn_std is not None:
        xerr = Fn_std.flatten(order="f")
    else:
        xerr = None

    for xi, yi, xerr_i in zip(x, y, xerr if xerr is not None else [None] * len(x)):
        if not np.isnan(xi):
            data_stab.append([xi, yi, None, xerr_i])

else:
    if Fn_std is not None:
        xerr = abs(Fn_std).flatten(order="f")
    else:
        xerr = None

    for i, (xi, yi) in enumerate(zip(x, y)):
        if not np.isnan(xi):
            err = xerr[i] if i < len(xerr) else None
            data_stab.append([xi, yi, None, err])

    for i, (xi1, yi1) in enumerate(zip(x1, y1)):
        if not np.isnan(xi1):
            err = xerr[len(x) + i] if len(xerr) > len(x) + i else None
            data_stab.append([xi1, None, yi1, err])

df_poles = pd.DataFrame(data_stab, columns=['Frequency', 'Stable', 'Unstable', 'Error',])
output_path_stab = os.path.join(save_folder, outnm_stab)
df_poles.to_csv(output_path_stab, index=False)

# frequency-damping csv file  ===============================================================
outnm_FnXi=("SSIdat_freq-damp_"+os.path.splitext(os.path.basename(folder_path))[0]+".csv")
data_FnXi = []
a = np.where(Lab == 1, Fn, np.nan) # stable
a_flat = a.flatten(order="f")
aa = np.where(Lab == 1, Xi, np.nan)
aa_flat = aa.flatten(order="f")
b = np.where(Lab == 0, Fn, np.nan) # unstable
b_flat = b.flatten(order="f")
bb = np.where(Lab == 0, Xi, np.nan)
bb_flat = bb.flatten(order="f")

if hide_poles:
    if Fn_std is not None:
        f_err = Fn_std.flatten(order="f") 
    else:
        f_err = None

    for f_st, d_st, f_err in zip(a_flat, aa_flat, f_err if f_err is not None else [None]*len(a_flat)):
        if not np.isnan(f_st):
            data_FnXi.append([f_st, d_st, None, f_err])

else:
    if Fn_std is not None:
        f_err = abs(Fn_std).flatten(order="f")
    else:
        f_err =None

    for i, (f_st, d_st) in enumerate(zip(a_flat, aa_flat)):
        if not np.isnan(f_st):
            err = f_err[i] if f_err is not None and i < len(f_err) else None
            data_FnXi.append([f_st, d_st, None, err])

    for i, (f_unst, d_unst) in enumerate(zip(b_flat, bb_flat)):
        if not np.isnan(f_unst):
            err = f_err[len(a_flat) + i] if f_err is not None and (len(a_flat) + i) < len(f_err) else None
            data_FnXi.append([f_unst, None, d_unst, err])

df_freq_damp_poles = pd.DataFrame(data_FnXi, columns=['Frequency', 'Stable', 'Unstable', 'Error'])
output_path_FnXi = os.path.join(save_folder, outnm_FnXi)
df_freq_damp_poles.to_csv(output_path_FnXi, index=False)


#====================================================================================================================
# run Hierarchical clustering
#====================================================================================================================

poles_list = extract_valid_poles(Fn, Phi, Lab)  

min_distances = compute_min_distances(poles_list)
d_vec = np.array([entry['distance'] for entry in min_distances])
d_percent = np.percentile(d_vec, q=percentile)

distance_matrix = compute_distance_matrix(poles_list)
condensed_dist = squareform(distance_matrix)
Z = linkage(condensed_dist, method='single')
clusters = fcluster(Z, t=d_percent, criterion='distance')  

fig = plot_Hclus(Z, d_percent) 
output_Hclus_fig = os.path.join(save_folder, "Hclus_" + os.path.splitext(os.path.basename(folder_path))[0] + ".png")
fig.savefig(output_Hclus_fig, dpi=300, bbox_inches='tight')
plt.close(fig) 

zoom_max = d_percent * 1.2
fig_zoom = plot_Hclus(Z, d_percent, zoom_ylim=(0, zoom_max))
output_zoom_fig = os.path.join(save_folder, "Hclus_zoom_" + os.path.splitext(os.path.basename(folder_path))[0] + ".png")
fig_zoom.savefig(output_zoom_fig, dpi=300, bbox_inches='tight')
plt.close(fig_zoom)

# output Hierarchical clustering csv file
outnm_Hclus = ("Hclus_"+os.path.splitext(os.path.basename(folder_path))[0]+".csv")
data = []
Hclus = []
for cluster_id, pole_info in zip(clusters, poles_list):
    indx, n, freq, phi = pole_info
    Hclus.append([cluster_id, n, indx, freq])
    df_Hclus_info = pd.DataFrame(Hclus, columns=['ClusterID', 'ModelIndex', 'PoleIndex', 'Frequency'])

freq_info = df_Hclus_info.groupby('ClusterID')['Frequency'].agg(MedFrequency='median', AvgFrequency='mean', MinFrequency='min', MaxFrequency='max').reset_index()
df_freq_info = freq_info.rename(columns={'ClusterID': 'Cluster'})
cluster_N = df_Hclus_info['ClusterID'].value_counts().to_dict()
df_Hclus_info['ClusterN'] = df_freq_info['Cluster'].map(cluster_N)

Hclus = pd.concat([df_Hclus_info, df_freq_info], axis=1)
data.append(Hclus)

df_Hclus = pd.concat(data,  axis=1)
output_path_Hclus = os.path.join(save_folder, outnm_Hclus)
df_Hclus.to_csv(output_path_Hclus, index=False)

# median frequency
med_freq_list = df_Hclus.loc[df_Hclus['ClusterN'] >= min_cluster_elements, 'MedFrequency'].tolist()
print(med_freq_list)

#====================================================================================================================
# Select modes to extract
#====================================================================================================================

sel_freq = med_freq_list
ss.mpe("SSIdat", sel_freq=sel_freq, rtol=rtol)
mode_nr = len(sel_freq)
plt.close()

# output mode shape =========================================================

ss.def_geo2_by_file(geo)

if mpl == 'h':
    _, _ = ss.plot_geo2_mpl(scaleF=2)
    for i in range(1, mode_nr + 1):
        fig_shape, _ = ss.plot_mode_geo2_mpl(algo_res=algoRes, mode_nr=i, view="3D", scaleF=3)
        freq = algoRes.Fn[i - 1]
        Phi_filename = f"{freq:.2f}Hz.png"
        fig_shape.savefig(os.path.join(save_folder, Phi_filename), dpi=300)

if pyvista == 'h':
    _ = ss.plot_geo2(scaleF=2)
    for i in range(1, mode_nr + 1):
        _ = ss.plot_mode_geo2(algo_res=algoRes, mode_nr=i, scaleF=3)
        _ = ss.anim_mode_geo2(algo_res=algoRes, mode_nr=i, scaleF=3)


# MPC, MPD =========================================================
Phi = algoRes.Phi

def compute_mpc(phi: np.ndarray) -> float:
    real = np.real(phi)
    imag = np.imag(phi)

    Sxx = real.T @ real
    Syy = imag.T @ imag
    Sxy = real.T @ imag

    numerator = (Sxx - Syy)**2 + 4 * (Sxy**2)
    denominator = (Sxx + Syy)**2

    return numerator / denominator

def compute_mpd(phi: np.ndarray) -> float:
    real_phi = np.real(phi).reshape(-1, 1)
    imag_phi = np.imag(phi).reshape(-1, 1)

    # Construct matrix [Re(phi), Im(phi)]
    Phi_mat = np.hstack((real_phi, imag_phi))

    # Perform SVD: Phi_mat = U * S * Vh
    _, _, Vh = np.linalg.svd(Phi_mat, full_matrices=False)
    V = Vh.T
    V12 = V[0, 1]
    V22 = V[1, 1]

    # Compute MPD
    num = 0.0
    denom = 0.0
    for o in range(len(phi)):
        phi_o = phi[o]
        abs_phi_o = np.abs(phi_o)
        if abs_phi_o == 0:
            continue  # avoid division by zero

        weight = abs_phi_o
        numerator = np.real(phi_o) * V22 - np.imag(phi_o) * V12
        denominator = np.sqrt(V12**2 + V22**2) * abs_phi_o

        # Ensure the argument of arccos is in [-1, 1]
        cos_val = np.clip(numerator / denominator, -1.0, 1.0)
        angle = np.arccos(cos_val)

        num += weight * angle
        denom += weight

    return float(num / denom / np.pi) if denom != 0 else 0.0

mpc_list = [compute_mpc(Phi[:, i]) for i in range(Phi.shape[1])]
mpc_array = np.array(mpc_list)
print("MPC values:", mpc_array)
mpd_list = [compute_mpd(Phi[:, i]) for i in range(Phi.shape[1])]
print("MPD values:", mpd_list)

# output mode shape csv file =========================================================
outnm_phi = ("SSIdat_shape_"+os.path.splitext(os.path.basename(folder_path))[0]+".csv")
df_geo = pd.read_excel(geo, sheet_name='sensors names', header=None)
sensor_names = df_geo.iloc[1, 1:].dropna().tolist()
Fn = algoRes.Fn
Phi = algoRes.Phi.real

header = [''] + [f"mode{i+1}" for i in range(len(Fn))]
row_freq = ['freq(Hz)'] + list(np.round(Fn, 4))
row_mpc = ['MPC'] + list(np.round(mpc_list, 6))
row_mpd = ['MPD'] + list(np.round(mpd_list, 6))

data_phi = []
for i, name in enumerate(sensor_names):
    row = [name] + list(np.round(Phi[i], 6))  
    data_phi.append(row)

df_phi = pd.DataFrame([header, row_freq, row_mpc, row_mpd] + data_phi)
output_path_phi = os.path.join(save_folder, outnm_phi)
df_phi.to_csv(output_path_phi, index=False, header=False)


# save parameter text file=========================================================
output_path_param = os.path.join(save_folder, "parameter_"+os.path.splitext(os.path.basename(folder_path))[0]+".txt")
with open(output_path_param, 'w') as f:
    indent = "  "
    f.write("・SSI parameter\n") 
    f.write(f"{indent}fs = {fs}\n")
    f.write(f"{indent}br = {br}\n")
    f.write(f"{indent}ordmax = {ordmax}\n")
    f.write(f"{indent}ordmin = {ordmin}\n")
    f.write(f"{indent}step = {step}\n")
    f.write(f"{indent}calc_unc = {calc_unc}\n")
    f.write(f"{indent}nb = {nb}\n")
    f.write(f"{indent}hide_poles = {hide_poles}\n")
    f.write("・Detrending, decimation, filtering parameter\n")
    f.write(f"{indent}q = {q}\n")
    f.write(f"{indent}Wn = {Wn}\n")
    f.write(f"{indent}order = {order}\n")
    f.write(f"{indent}btype = {btype}\n")
    f.write(f"{indent}freqlim = {freqlim}\n")
    f.write("・clustering parameter\n")
    f.write(f"{indent}percentile = {percentile}\n")

plt.show()
