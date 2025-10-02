import pandas as pd
from pathlib import Path
from collections import defaultdict

def collect_csv_files(base_folders):
    """
    フォルダ配下（サブフォルダ含む）のすべてのCSVファイルを収集
    戻り値: {ファイル名: [ファイルパス, ...]} の辞書
    """
    file_dict = defaultdict(list)
    for base in base_folders:
        for path in Path(base).rglob("*.csv"):
            file_dict[path.name].append(path)
    return file_dict


def compare_csv_files(file_dict):
    """
    同名CSVファイルが完全一致するか判定
    """
    results = {}

    for filename, paths in file_dict.items():
        if len(paths) < 2:
            # 比較対象が1つしかない場合はスキップ
            continue

        dfs = []
        for path in paths:
            try:
                df = pd.read_csv(path)
                dfs.append(df)
            except Exception as e:
                print(f"読み込み失敗: {path}, {e}")
                dfs = []
                break

        if dfs and all(dfs[0].equals(df) for df in dfs[1:]):
            results[filename] = "一致"
        else:
            results[filename] = "不一致"

    return results


# --- 使用例 ---
folders = [
    r"D:\case_study\belltower\result\all_3, 4, 5, 6\EFDD_SSIdat_clustering_sel freq_mode shape_FR\SSI (auto)",
    r"D:\case_study\belltower\result\all_3, 4, 5, 6\oma_gui\v0.23\set_up_2\SSI (auto)"
]

file_dict = collect_csv_files(folders)
result = compare_csv_files(file_dict)

for k, v in result.items():
    print(f"{k}: {v}")
