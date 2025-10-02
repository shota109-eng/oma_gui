import pandas as pd
import os

# CSV 読み込み（例: time[s], value）
df = pd.read_csv(r"d:\shaking_table_test\2024_July\csv\1903-1932\data.csv")

# パラメータ
threshold = 0.3  # 閾値
min_samples = int(60 / 0.005)  # 60秒間に必要なサンプル数 = 1200

df["value"] = df["value"] - df["value"].mean()

# 閾値判定
mask = (df["value"].abs() < threshold)

# マスクの変化点を見つける
df["group"] = (mask != mask.shift()).cumsum()

# 保存先フォルダ
output_dir = r"D:\shaking_table_test\2024_July\csv\1903-1932"
os.makedirs(output_dir, exist_ok=True)

results = []
idx = 1
for g, sub in df.groupby("group"):
    if mask[sub.index[0]]:  # True の区間だけ
        if len(sub) >= min_samples:
            # 区間情報を記録
            results.append({
                "file": f"interval_{idx}.csv",
                "start_time": sub["time"].iloc[0],
                "end_time": sub["time"].iloc[-1],
                "duration": len(sub) * 0.005
            })
            # 区間データを保存
            sub.to_csv(os.path.join(output_dir, f"interval_{idx}.csv"), index=False)
            idx += 1

# 区間一覧をまとめて保存
intervals = pd.DataFrame(results)
intervals.to_csv(os.path.join(output_dir, "interval_summary.csv"), index=False)

print("保存完了:", intervals)