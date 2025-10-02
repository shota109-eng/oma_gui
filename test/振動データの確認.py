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

# 出力フォルダを選択
out_dir = filedialog.askdirectory(title="保存先フォルダを選択してください")

for csv_file in file_paths:
    # CSVを読み込み（ヘッダなし）
    df = pd.read_csv(csv_file, header=2)

    # 4行目から A列(0列目) と B列(1列目) を取り出す
    time = df.iloc[0:, 0].astype(float).values
    response = df.iloc[0:, 1].astype(float).values

    # プロット
    plt.figure(figsize=(8, 4))
    plt.plot(time, response, label="Response")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Time History of Vibration")
    plt.legend()
    plt.grid(True)

    # 出力ファイル名（入力ファイル名を利用）
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    out_file = os.path.join(out_dir, base_name + "_plot.png")

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()  # 表示せずに閉じる
