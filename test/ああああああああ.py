def _post_process(self, params, ss, efdd, ssidat):
    # make directory to save the results
    efdd_save_folder = params['save_dir'] + '/EFDD'
    ssi_save_folder  = params['save_dir'] + '/SSI (auto)'
    os.makedirs(efdd_save_folder, exist_ok=True)
    os.makedirs(ssi_save_folder , exist_ok=True)

    # post-process
    self.save_stab_and_damp_diagram(params, ssidat, ssi_save_folder)

    clusters, min_distances, d_percent, Z = self.hierarchical_clustering()
    self.save_clustering_info(params, ssi_save_folder, min_distances, d_percent, Z)

    large_clusters = self.delete_small_clusters(params, clusters)
    self.save_cluster(params, ssi_save_folder, clusters, large_clusters, outlier=False)

    large_clusters_without_outliers = self.cutoff_outliers_from_clusters(large_clusters)
    self.save_cluster(params, ssi_save_folder, clusters, large_clusters_without_outliers, outlier=True)

    self.mpe_from_clusters(params, ssi_save_folder, large_clusters_without_outliers, ss, ssidat, efdd)

def save_stab_and_damp_diagram(self, params, ssidat, ssi_save_folder):
    p = params

    # figure ===================================================================
    # stabilization
    fig, ax = ssidat.plot_stab(freqlim=p['freqlim'], hide_poles=p['hide_poles'])
    output_ssidat_fig = ssi_save_folder + "/" + "SSIdat_stab_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".png"
    plt.savefig(output_ssidat_fig, dpi=300, bbox_inches='tight')
    # damping
    fig, ax = ssidat.plot_freqvsdamp(freqlim=p['freqlim'], hide_poles=p['hide_poles'])
    output_ssidat_fig = ssi_save_folder + "/" + "SSIdat_freq-damp_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".png"
    plt.savefig(output_ssidat_fig, dpi=300, bbox_inches='tight')

    # csv ======================================================
    # stabilization
    df_stab_poles = self.stab_poles_to_df(p['hide_poles'], ssidat.result, ssidat)
    outnm_stab=("SSIdat_stab_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".csv")
    output_path_stab = ssi_save_folder + "/" + outnm_stab
    df_stab_poles.to_csv(output_path_stab, index=False)
    # damping
    df_freq_damp_poles = self.freq_damp_poles_to_df(p['hide_poles'], ssidat.result)
    outnm_FnXi=("SSIdat_freq-damp_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".csv")
    output_path_FnXi = ssi_save_folder + "/" + outnm_FnXi
    df_freq_damp_poles.to_csv(output_path_FnXi, index=False)

def hierarchical_clustering(self, params, ssidat):
    df_poles = self.poles_to_df(ssidat.result)
    valid_poles_list = self.extract_valid_poles(ssidat.result)
    min_distances, d_percent, Z, clusters = self.single_linkage_on_poles_list(valid_poles_list, params['percentile'])

    c = pd.DataFrame({'cluster_id': clusters})
    df_clustered_poles = pd.concat([c, df_poles], axis=1)

    return df_clustered_poles, min_distances, d_percent, Z

def save_clustering_info(self, params, ssi_save_folder, min_distances, d_percent, Z):
    p = params

    # compute minimam distance and output to csv
    df_min_distances = self.min_distances_with_percentiles_to_df(min_distances)
    outnm = "SSIdat_min_distances_" + os.path.splitext(os.path.basename(p['data_folder']))[0] + ".csv"
    output_path = ssi_save_folder + "/" + outnm
    df_min_distances.to_csv(output_path, index=False, encoding="utf-8-sig")

    # output dendrogram
    fig = self.plot_Hclus(Z, d_percent)
    output_Hclus_fig = ssi_save_folder + "/" + "Hclus_" + os.path.splitext(os.path.basename(p['data_folder']))[0] + ".png"
    fig.savefig(output_Hclus_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)

    zoom_max = d_percent * 1.2
    fig_zoom = self.plot_Hclus(Z, d_percent, zoom_ylim=(0, zoom_max))
    output_zoom_fig = ssi_save_folder + "/" + "Hclus_zoom_" + os.path.splitext(os.path.basename(p['data_folder']))[0] + ".png"
    fig_zoom.savefig(output_zoom_fig, dpi=300, bbox_inches='tight')
    plt.close(fig_zoom)

def delete_small_clusters(self, params, clusters):
    min_cluster_elements = params['min_cluster_elements']

    counts = clusters['cluster_id'].value_counts()
    mask = counts >= min_cluster_elements
    large_cluster_id = counts[mask].index.tolist()

    large_clusters = clusters.query('state in @large_cluster_id')

    return large_clusters

def cutoff_outliers_from_clusters(self, clusters):
    c_id = clusters['cluster'].unique()

    cluster_without_outlier = pd.DataFrame(columns=clusters.columns)
    for n in c_id:
        df = clusters[clusters['cluster'] == n]
        while True:
            Fn = df['Fn']
            q = Fn.quantile([0.25, 0.75])
            q1 = q.iloc[0]
            q3 = q.iloc[1]
            iqr = q3 - q1

            upper = q3 + 1.5 * iqr
            lower = q1 - 1.5 * iqr

            mask = (Fn < upper) & (Fn > lower)

            if all(mask):
                break
            else:
                df = df[mask]

        cluster_without_outlier = pd.concat([cluster_without_outlier, df])

    return cluster_without_outlier

def save_cluster(self, params, ssi_save_folder, clusters, large_clusters, outlier=False):
    df_Hclus = self.make_df_Hclus(clusters, large_clusters)

    if outlier:
        outnm_Hclus = ("Hclus_outlier_"+os.path.splitext(os.path.basename(params['data_folder']))[0]+".csv")
    else:
        outnm_Hclus = ("Hclus_"+os.path.splitext(os.path.basename(params['data_folder']))[0]+".csv")
    output_path_Hclus = ssi_save_folder + "/" + outnm_Hclus
    df_Hclus.to_csv(output_path_Hclus, index=False)

def make_df_Hclus(self, clusters, large_clusters):
    # df_left
    df_left = clusters['cluster_id', 'row', 'col', 'Fn', 'Xi']
    df_left.set_axis(['ClusterID', 'ModelIndex', 'PoleIndex', 'Frequency', 'Damping'], axis='columns')

    # df_center
    count_of_cluster_elements = large_clusters['cluster_id'].value_counts().sort_index()
    df_center = count_of_cluster_elements.reset_index()
    df_center.columns = ['ClusterNo.', 'Elements']

    # df_right
    df_right = self.make_df_right_of_Hclus()

    # concat the above dfs
    df_Hclus = pd.concat([
        df_left,
        pd.DataFrame(columns=[' ']),
        df_center,
        pd.DataFrame(columns=[' ']),
        df_right
        ], axis=1)

    return df_Hclus

def make_df_right_of_Hclus(self, large_clusters, df_center):
    df_upper = self.make_df_right_upper_of_Hclus(large_clusters, df_center)
    df_lower = self.make_df_right_lower_of_Hclus(large_clusters, df_center)

    df_right = pd.concat([
        df_upper,
        pd.DataFrame([[np.nan]], columns=['ClusterNo.']),
        df_lower
    ]).reset_index(drop=True)
    df_right.columns = [' '] + [f"mode{i}" for i in range(1, len(df_right.columns))]

    return df_right

def make_df_right_upper_of_Hclus(self, large_clusters, df_center):
    df_cluster_info_on_freq = large_clusters.groupby('cluster_id')['Fn'].agg(
        CV_freq=lambda x: x.std(ddof=0) / x.mean(),
        MedFrequency='median',
        AvgFrequency='mean',
        MinFrequency='min',
        MaxFrequency='max'
    ).reset_index().rename(columns={'cluster_id': 'ClusterNo.'})

    df_cluster_info_on_damp = large_clusters.groupby('cluster_id')['Xi'].agg(
        CV_damp=lambda x: x.std(ddof=0) / x.mean(),
        MedDamping='median',
        AvgDamping='mean'
    ).reset_index().rename(columns={'cluster_id': 'ClusterNo.'})

    # merge df_count_of_cluster_elements, df_cluster_info_on_freq, df_cluster_info_on_damp
    df_clusters_info = pd.merge(
        df_cluster_info_on_freq,
        df_cluster_info_on_damp,
        on='ClusterNo.'
    )
    df_clusters_info = pd.merge(
        df_center,
        df_clusters_info,
        on='ClusterNo.'
    )
    df_clusters_info = df_clusters_info.sort_values('MedFrequency')

    df_clusters_info_T = df_clusters_info.T.reset_index()
    df_clusters_info_T.columns = ['ClusterNo.'] + df_clusters_info['ClusterNo.'].to_list()

    return df_clusters_info_T

def make_df_right_lower_of_Hclus(self, large_clusters, df_center):
    df_freqs_in_large_clusters = pd.DataFrame()

    for n in df_center['ClusterNo.']:
        df_cluster = large_clusters[
            large_clusters['ClusterID'] == n
        ].reset_index(drop=True)

        df_freqs_in_large_clusters = pd.concat([df_freqs_in_large_clusters, df_cluster['Frequency']], axis=1)

    df_freqs_in_large_clusters.columns = df_center['ClusterNo.']  # 他のdfと連結しやすくする

    return df_freqs_in_large_clusters

def mpe_from_clusters(self, params, ssi_save_folder, clusters, ss, ssidat, efdd):
    p = params
    for med_or_mean in ['med', 'mean']:
        mpe_folder = ssi_save_folder + '/' + med_or_mean
        os.makedirs(mpe_folder, exist_ok=True)

        # extract modal parameters from each cluster
        df_mp = self.modal_params_of_med_or_mean_values_of_xi_in_clusters_to_df(clusters, med_or_mean, p['min_cluster_elements'])

        # save the mpe result
        result = ss.algorithms['SSIdat'].result
        result.Fn      = df_mp['Fn'    ].to_numpy()
        result.Fn_std  = df_mp['Fn_std'].to_numpy()
        result.Phi     = np.array(df_mp['Phi'    ].to_list()).T
        result.Phi_std = np.array(df_mp['Phi_std'].to_list()).T
        result.Xi      = df_mp['Xi'    ].to_numpy()
        result.Xi_std  = df_mp['Xi_std'].to_numpy()

        mode_nr = len(df_mp)

        # output mode shape
        self.save_mode_shape(ss, p['geo_path'], mode_nr, ssidat.result, mpe_folder, p['mpe_with_mpl'], p['mpe_with_pv'])

        # create a dataframe of the mpe result
        df_valid = self.phi_to_df(
            df_mp,
            clusters,
            efdd.result,
            p['geo_path']
        )

        # output the dataframe
        outnm_valid = ("ssidat_validation_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".csv")
        output_path_valid = mpe_folder + "/" + outnm_valid
        df_valid.to_csv(output_path_valid, index=True, header=True)

def modal_params_of_med_or_mean_values_of_xi_in_clusters_to_df(self, clusters, med_or_mean, min_cluster_elements):
    uniques = clusters['cluster_id'].unique()

    # indexes to extraxt modal parameters
    idx_mp = []
    for u in uniques:
        df = clusters[clusters['cluster_id'] == u].copy()
        df = df.sort.sort_values(['Xi'])

        if med_or_mean == 'med':
            l = len(df)
            if l % 2 == 0:
                idx = df.iloc[(len(df) // 2) - 1].name
            else:
                idx = df.iloc[len(df) // 2].name

        elif med_or_mean == 'mean':
            s = df['Xi'] - df['Xi'].mean()
            idx = s.abs().idxmin()

        idx_mp.append(idx)

    # extract modal parameters
    df_mp = clusters.loc[idx_mp, :].sort_values('Fn')

    return df_mp

def phi_to_df(self, df_mp, clusters, efdd_result, geo):
    Fn          = df_mp['Fn'        ].to_numpy()
    Phi         = np.array(df_mp['Phi'    ].to_list()).T
    Xi          = df_mp['Xi'        ].to_numpy()
    cluster_num = df_mp['ClusterNo.'].to_numpy()

    elements    = clusters['cluster_id'].value_counts().reindex(index=cluster_num).to_numpy()
    CV_freq, CV_damp = self.compute_CV()
    mpc_list, mpd_list, mcf_list = self.MPC_MPD_MCF(Phi)
    FR_list = self.compute_FR(efdd_result, Fn)

    if not geo == '':
        df_geo = pd.read_excel(geo, sheet_name='sensors names', header=None)
        sensor_names = df_geo.iloc[1, 1:].dropna().tolist()
    else:
        sensor_names = [f'Ch.{i}' for i in range(1, Phi.shape[0]+1)]

    df_phi = pd.DataFrame(columns=[f"mode{i+1}" for i in range(len(Fn))])
    df_phi.loc['freq(Hz)']   = np.round(Fn, 4)
    df_phi.loc['damp(%)']    = np.round(Xi, 4)
    df_phi.loc['ClusterNo.'] = cluster_num
    df_phi.loc['Elements']   = elements
    df_phi.loc['CV_freq']    = np.round(CV_freq, 4)
    df_phi.loc['CV_damp']    = np.round(CV_damp, 4)
    df_phi.loc['1-CV_freq']  = 1 - df_phi.loc['CV_freq']
    df_phi.loc['1-CV_damp']  = 1 - df_phi.loc['CV_damp']
    df_phi.loc['MPC']        = np.round(mpc_list, 6)
    df_phi.loc['MPD']        = np.round(mpd_list, 6)
    df_phi.loc['1 - MPD']    = 1 - df_phi.loc['MPD']
    df_phi.loc['MCF']        = np.round(mcf_list, 6)
    df_phi.loc['FR']         = np.round(FR_list, 6)
    df_phi.loc[' ']          = np.nan
    for i, name in enumerate(sensor_names):
        df_phi.loc[name] = list(np.round(Phi.real[i], 6))

    return df_phi

def compute_CV(self, clusters, cluster_num):
    CV = clusters.groupby('cluster_id')['Fn', 'Xi'].agg(
        lambda x: x.std(ddof=0) / x.mean()
    )

    CV = CV.reindex(index=cluster_num)

    CV_freq = CV['Fn'].to_numpy()
    CV_damp = CV['Xi'].to_numpy()

    return CV_freq, CV_damp

def MPC_MPD_MCF(self, Phi):

    mpc_list         = [gen.MPC(Phi[:, i]) for i in range(Phi.shape[1])]
    mpd_list         = [gen.MPD(Phi[:, i]) for i in range(Phi.shape[1])]
    mcf_list         = [gen.MCF(Phi[:, i])[0] for i in range(Phi.shape[1])]

    return mpc_list, mpd_list, mcf_list

