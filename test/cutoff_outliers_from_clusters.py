import pandas as pd

def cutoff_outliers_from_clusters(self, cluster):
    c_id = cluster['cluster'].unique()

    cluster_without_outlier = pd.DataFrame(columns=cluster.columns)
    for n in c_id:
        df = cluster[cluster['cluster'] == n]
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

a = [-10, 1, 4, 5, 6, 9, 20]
b = list(range(len(a)))
df = pd.DataFrame({'cluster': [1]*len(a), 'Fn': a, 'hoge': b})

self = 'fuga'
df_valid = cutoff_outliers_from_clusters(self, df)

print(df_valid)