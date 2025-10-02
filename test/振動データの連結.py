import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog

# Tkinter のルートウィンドウを非表示
root = tk.Tk()
root.withdraw()

# 複数のCSVファイルを選択
file_paths = filedialog.askopenfilenames(
    title="CSVファイルを選択してください",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

for name in file_paths:
    print(name)

# 出力フォルダを選択
out_dir = filedialog.askdirectory(title="保存先フォルダを選択してください")

df = pd.DataFrame()

for csv_file in file_paths:
    # CSVを読み込み（ヘッダなし）
    df_new = pd.read_csv(csv_file, header=1)

    df = pd.concat([df, df_new])

print(len(df))

# 出力ファイル名（入力ファイル名を利用）
out_file = os.path.join(out_dir, "data.csv")
df.to_csv(out_file)
