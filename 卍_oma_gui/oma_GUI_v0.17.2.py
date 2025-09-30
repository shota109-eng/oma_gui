"""oma_GUI_vo.8 の
    else:
            if Fn_std is not None:
                xerr = abs(Fn_std).flatten(order="f")
            else:
                xerr = None

    で else: xerr = None を else: xerr = [] に変更
    """

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import os
import shlex
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.interpolate import griddata

from pyoma2.algorithms import FDD, EFDD, FSDD, SSIdat, SSIcov
from pyoma2.setup import SingleSetup
from pyoma2.functions import gen, ssi

class OmaApp:
    def __init__(self, master):
        self.master = master
        master.title("OMA App")

        self.set_up_dict = {}

        self.construct_widgets()

    def construct_widgets(self):
        master = self.master
        #=========================================================
        # Construct tabs
        #=========================================================
        # Create Notebook Widget
        note = ttk.Notebook(master)
        # Create tab
        self.tab_data = tk.Frame(note)
        self.tab_geo = tk.Frame(note)
        self.tab_param = tk.Frame(note)
        self.tab_run = tk.Frame(note)
        self.tab_result = tk.Frame(note)
        self.tab_sel_freq = tk.Frame(note)
        self.tab_mpe = tk.Frame(note)
        self.tab_SSI = tk.Frame(note)
        self.tab_SSIres = tk.Frame(note)
        self.tab_SSIgeo = tk.Frame(note)
        # Add tab
        note.add(self.tab_data, text='Import Data')
        note.add(self.tab_geo, text='Geometry')
        note.add(self.tab_param, text='Parameter')
        note.add(self.tab_run, text='Run')
        note.add(self.tab_result, text='Result')
        # note.add(self.tab_sel_freq, text='Preprocessing')
        note.add(self.tab_mpe, text='MPE')
        # note.add(self.tab_SSI, text='SSI')
        # note.add(self.tab_SSIres, text='SSI_result')
        # note.add(self.tab_SSIgeo, text='SSI_geometry')

        # Assign Widgets of Designer to py code
        #=========================================================
        # Import Data Frame
        #=========================================================
        # import data from csv file path
        # create widgets
        self.data_frm0 = tk.Frame(self.tab_data)
        self.data_lab0_0_0 = tk.Label(self.data_frm0, text='importing data folder')
        self.data_lab0_1_0 = tk.Label(self.data_frm0, text='importing data files')
        self.data_lab0_2_0 = tk.Label(self.data_frm0, text='use cols')
        self.data_lab0_3_0 = tk.Label(self.data_frm0, text='start time')
        self.data_lab0_4_0 = tk.Label(self.data_frm0, text='save folder')
        self.data_etr0_0_1 = tk.Entry(self.data_frm0)
        self.data_etr0_1_1 = tk.Entry(self.data_frm0)
        self.data_etr0_2_1 = tk.Entry(self.data_frm0)
        self.data_etr0_3_1 = tk.Entry(self.data_frm0)
        self.data_etr0_4_1 = tk.Entry(self.data_frm0)
        self.data_btn0_0_2 = tk.Button(self.data_frm0, text='Brows', command=self.brows_importing_data_folder)
        self.data_btn0_1_2 = tk.Button(self.data_frm0, text='Brows', command=self.brows_importing_data_files)
        self.data_btn0_4_2 = tk.Button(self.data_frm0, text='Brows', command=self.brows_save_folder)

        # layout widgets
        self.data_lab0_0_0.grid(row=0, column=0)
        self.data_lab0_1_0.grid(row=1, column=0)
        self.data_lab0_2_0.grid(row=2, column=0)
        self.data_lab0_3_0.grid(row=3, column=0)
        self.data_lab0_4_0.grid(row=4, column=0)
        self.data_etr0_0_1.grid(row=0, column=1, sticky=tk.W+tk.E)
        self.data_etr0_1_1.grid(row=1, column=1, sticky=tk.W+tk.E)
        self.data_etr0_2_1.grid(row=2, column=1)
        self.data_etr0_3_1.grid(row=3, column=1)
        self.data_etr0_4_1.grid(row=4, column=1, sticky=tk.W+tk.E)
        self.data_btn0_0_2.grid(row=0, column=2)
        self.data_btn0_1_2.grid(row=1, column=2)
        self.data_btn0_4_2.grid(row=4, column=2)
        self.data_frm0.grid_columnconfigure(1, weight=1)
        self.data_frm0.pack(fill=tk.X)

        # initialize widgets of Import data
        self.data_etr0_2_1.insert(0, '1, 2, 3')
        self.data_etr0_3_1.insert(0, '')

        # title label
        self.titleLabel = tk.Label(self.tab_data, text="Input Data", font=('Helvetica', '35'))
        self.titleLabel.pack(anchor='center', expand=True)

        #=========================================================
        # Geometry Frame
        #=========================================================
        # create widgets
        self.geo_frm0 = tk.Frame(self.tab_geo)
        self.geo_lab0_0_0 = tk.Label(self.geo_frm0, text='load geometry file')
        self.geo_etr0_0_1 = tk.Entry(self.geo_frm0)
        self.geo_btn0_0_2 = tk.Button(self.geo_frm0, text='Brows', command=self.brows_geometry_file)

        # layout widgets
        self.geo_lab0_0_0.grid(row=0, column=0)
        self.geo_etr0_0_1.grid(row=0, column=1, sticky=tk.W+tk.E)
        self.geo_btn0_0_2.grid(row=0, column=2)
        self.geo_frm0.grid_columnconfigure(1, weight=1)
        self.geo_frm0.pack(fill=tk.X)

        # title label
        self.titleLabel = tk.Label(self.tab_geo, text="Geometry", font=('Helvetica', '35'))
        self.titleLabel.pack(anchor='center', expand=True)

        #=========================================================
        # Parameter Frame
        #=========================================================
        # Boolean variables for check buttons
        self.var_to_calc_unc = tk.BooleanVar()
        self.var_to_hide_poles = tk.BooleanVar()
        self.var_to_run_fdd = tk.BooleanVar()
        self.var_to_run_ssi = tk.BooleanVar()
        self.var_to_mpe_with_mpl = tk.BooleanVar()
        self.var_to_mpe_with_pv = tk.BooleanVar()
        self.var_of_clustering_method = tk.StringVar()

        # create widgets
        self.param_lfm0 = ttk.Labelframe(self.tab_param, text='Set up')
        self.param_lfm0_0 = ttk.Labelframe(self.param_lfm0, text='basic')
        self.param_lfm0_1 = ttk.Labelframe(self.param_lfm0, text='Detrend, filter and decimate')
        self.param_lab0_0_0_0 = tk.Label(self.param_lfm0_0, text='sampling frequency')
        self.param_lab0_0_1_0 = tk.Label(self.param_lfm0_0, text='freqlim')
        self.param_etr0_0_0_1 = tk.Entry(self.param_lfm0_0)
        self.param_etr0_0_1_1 = tk.Entry(self.param_lfm0_0)
        self.param_lab0_1_0_0 = tk.Label(self.param_lfm0_1, text='q')
        self.param_lab0_1_1_0 = tk.Label(self.param_lfm0_1, text='Wn')
        self.param_lab0_1_2_0 = tk.Label(self.param_lfm0_1, text='order')
        self.param_lab0_1_3_0 = tk.Label(self.param_lfm0_1, text='btype')
        self.param_etr0_1_0_1 = tk.Entry(self.param_lfm0_1)
        self.param_etr0_1_1_1 = tk.Entry(self.param_lfm0_1)
        self.param_etr0_1_2_1 = tk.Entry(self.param_lfm0_1)
        self.param_cmb0_1_3_1 = ttk.Combobox(self.param_lfm0_1, state='readonly', values=['lowpass', 'highpass', 'bandpass', 'bandstop'])

        self.param_lfm1 = ttk.Labelframe(self.tab_param, text='mpe')
        self.param_lfm1_0 = ttk.Labelframe(self.param_lfm1, text='basic')
        self.param_lfm1_1 = ttk.Labelframe(self.param_lfm1, text='visualize mode shape')
        self.param_lab1_0_0_0 = tk.Label(self.param_lfm1_0, text='rtol')
        self.param_lab1_0_1_0 = tk.Label(self.param_lfm1_0, text='sel_freq')
        self.param_lab1_0_2_0 = tk.Label(self.param_lfm1_0, text='order_in')
        self.param_lab1_0_3_0 = tk.Label(self.param_lfm1_0, text='sel_freqlim')
        self.param_etr1_0_0_1 = tk.Entry(self.param_lfm1_0)
        self.param_etr1_0_1_1 = tk.Entry(self.param_lfm1_0)
        self.param_etr1_0_2_1 = tk.Entry(self.param_lfm1_0)
        self.param_etr1_0_3_1 = tk.Entry(self.param_lfm1_0)
        self.param_lab1_1_0_0 = tk.Label(self.param_lfm1_1, text='matplotlib')
        self.param_lab1_1_1_0 = tk.Label(self.param_lfm1_1, text='pyvista')
        self.param_chb1_1_0_1 = tk.Checkbutton(self.param_lfm1_1, variable=self.var_to_mpe_with_mpl)
        self.param_chb1_1_1_1 = tk.Checkbutton(self.param_lfm1_1, variable=self.var_to_mpe_with_pv)

        self.param_lfm2 = ttk.Labelframe(self.tab_param, text='FDD')
        self.param_chb2_1 = tk.Checkbutton(self.param_lfm2, variable=self.var_to_run_fdd, text='run fdd')
        self.param_lfm2_0 = ttk.Labelframe(self.param_lfm2, text='basic')
        self.param_lab2_0_0_0 = tk.Label(self.param_lfm2_0, text='method')
        self.param_lab2_0_1_0 = tk.Label(self.param_lfm2_0, text='nxseg')
        self.param_lab2_0_2_0 = tk.Label(self.param_lfm2_0, text='method_SD')
        self.param_lab2_0_3_0 = tk.Label(self.param_lfm2_0, text='pov')
        self.param_lab2_0_4_0 = tk.Label(self.param_lfm2_0, text='DF')
        self.param_lab2_0_5_0 = tk.Label(self.param_lfm2_0, text='DF1')
        self.param_lab2_0_6_0 = tk.Label(self.param_lfm2_0, text='DF2')
        self.param_lab2_0_7_0 = tk.Label(self.param_lfm2_0, text='cm')
        self.param_lab2_0_8_0 = tk.Label(self.param_lfm2_0, text='MAClim')
        self.param_lab2_0_9_0 = tk.Label(self.param_lfm2_0, text='sppk')
        self.param_lab2_0_10_0 = tk.Label(self.param_lfm2_0, text='npmax')
        self.param_cmb2_0_0_1 = ttk.Combobox(self.param_lfm2_0, state='readonly', values=['EFDD'])
        self.param_etr2_0_1_1 = tk.Entry(self.param_lfm2_0)
        self.param_cmb2_0_2_1 = ttk.Combobox(self.param_lfm2_0, state='readonly', values=['per', 'cor'])
        self.param_etr2_0_3_1 = tk.Entry(self.param_lfm2_0)
        self.param_etr2_0_4_1 = tk.Entry(self.param_lfm2_0)
        self.param_etr2_0_5_1 = tk.Entry(self.param_lfm2_0)
        self.param_etr2_0_6_1 = tk.Entry(self.param_lfm2_0)
        self.param_etr2_0_7_1 = tk.Entry(self.param_lfm2_0)
        self.param_etr2_0_8_1 = tk.Entry(self.param_lfm2_0)
        self.param_etr2_0_9_1 = tk.Entry(self.param_lfm2_0)
        self.param_etr2_0_10_1 = tk.Entry(self.param_lfm2_0)

        self.param_lfm3 = ttk.Labelframe(self.tab_param, text='SSI')
        self.param_chb3_2 = tk.Checkbutton(self.param_lfm3, variable=self.var_to_run_ssi, text='run ssi')
        self.param_lfm3_0 = ttk.Labelframe(self.param_lfm3, text='basic')
        self.param_lfm3_1 = ttk.Labelframe(self.param_lfm3, text='hierarchical clustering')
        self.param_lfm3_2 = ttk.Labelframe(self.param_lfm3, text='find optimal br')
        self.param_lab3_0_0_0 = tk.Label(self.param_lfm3_0, text='method')
        self.param_lab3_0_1_0 = tk.Label(self.param_lfm3_0, text='br')
        self.param_lab3_0_2_0 = tk.Label(self.param_lfm3_0, text='ord max')
        self.param_lab3_0_3_0 = tk.Label(self.param_lfm3_0, text='calc_unc')
        self.param_lab3_0_4_0 = tk.Label(self.param_lfm3_0, text='nb')
        self.param_lab3_0_5_0 = tk.Label(self.param_lfm3_0, text='hide poles')
        self.param_cmb3_0_0_1 = ttk.Combobox(self.param_lfm3_0, state='readonly', values=['SSIdat'])
        self.param_etr3_0_1_1 = tk.Entry(self.param_lfm3_0)
        self.param_etr3_0_2_1 = tk.Entry(self.param_lfm3_0)
        self.param_chb3_0_3_1 = tk.Checkbutton(self.param_lfm3_0, variable=self.var_to_calc_unc)
        self.param_etr3_0_4_1 = tk.Entry(self.param_lfm3_0)
        self.param_chb3_0_5_1 = tk.Checkbutton(self.param_lfm3_0, variable=self.var_to_hide_poles)
        self.param_lab3_1_0_0 = tk.Label(self.param_lfm3_1, text='min distance')
        self.param_lab3_1_1_0 = tk.Label(self.param_lfm3_1, text='percentile')
        self.param_lab3_1_2_0 = tk.Label(self.param_lfm3_1, text='mean')
        self.param_lab3_1_3_0 = tk.Label(self.param_lfm3_1, text='min cluster elements')
        self.param_rbn3_1_0_1 = tk.Radiobutton(self.param_lfm3_1, value='min_distance', var=self.var_of_clustering_method)
        self.param_etr3_1_1_1 = tk.Entry(self.param_lfm3_1)
        self.param_rbn3_1_2_1 = tk.Radiobutton(self.param_lfm3_1, value='mean', var=self.var_of_clustering_method)
        self.param_etr3_1_3_1 = tk.Entry(self.param_lfm3_1)
        self.param_lab3_2_0_0 = tk.Label(self.param_lfm3_2, text='br min')
        self.param_lab3_2_1_0 = tk.Label(self.param_lfm3_2, text='br max')
        self.param_lab3_2_2_0 = tk.Label(self.param_lfm3_2, text='br step')
        self.param_lab3_2_3_0 = tk.Label(self.param_lfm3_2, text='ord min')
        self.param_lab3_2_4_0 = tk.Label(self.param_lfm3_2, text='ord max')
        self.param_lab3_2_5_0 = tk.Label(self.param_lfm3_2, text='ord step')
        self.param_lab3_2_6_0 = tk.Label(self.param_lfm3_2, text='threshold')
        self.param_etr3_2_0_1 = tk.Entry(self.param_lfm3_2)
        self.param_etr3_2_1_1 = tk.Entry(self.param_lfm3_2)
        self.param_etr3_2_2_1 = tk.Entry(self.param_lfm3_2)
        self.param_etr3_2_3_1 = tk.Entry(self.param_lfm3_2)
        self.param_etr3_2_4_1 = tk.Entry(self.param_lfm3_2)
        self.param_etr3_2_5_1 = tk.Entry(self.param_lfm3_2)
        self.param_etr3_2_6_1 = tk.Entry(self.param_lfm3_2)

        self.param_btn4 = tk.Button(self.tab_param, text='Save', command=self.define_set_up, font=('Helvetica', '35'))
        self.param_tbl5 = ttk.Treeview(self.tab_param, show="tree")
        self.param_tbl5.heading("#0", text='set-up name')

        # layout widgets
        self.param_lab0_0_0_0.grid(row=0, column=0)
        self.param_lab0_0_1_0.grid(row=1, column=0)
        self.param_etr0_0_0_1.grid(row=0, column=1)
        self.param_etr0_0_1_1.grid(row=1, column=1)
        self.param_lab0_1_0_0.grid(row=0, column=0)
        self.param_lab0_1_1_0.grid(row=1, column=0)
        self.param_lab0_1_2_0.grid(row=2, column=0)
        self.param_lab0_1_3_0.grid(row=3, column=0)
        self.param_etr0_1_0_1.grid(row=0, column=1)
        self.param_etr0_1_1_1.grid(row=1, column=1)
        self.param_etr0_1_2_1.grid(row=2, column=1)
        self.param_cmb0_1_3_1.grid(row=3, column=1)
        self.param_lfm0_0.pack()
        self.param_lfm0_1.pack()
        self.param_lfm0.grid(row=0, column=0)

        self.param_lab1_0_0_0.grid(row=0, column=0)
        # self.param_lab1_0_1_0.grid(row=1, column=0)
        # self.param_lab1_0_2_0.grid(row=2, column=0)
        # self.param_lab1_0_3_0.grid(row=3, column=0)
        self.param_etr1_0_0_1.grid(row=0, column=1)
        # self.param_etr1_0_1_1.grid(row=1, column=1)
        # self.param_etr1_0_2_1.grid(row=2, column=1)
        # self.param_etr1_0_3_1.grid(row=3, column=1)
        self.param_lab1_1_0_0.grid(row=0, column=0)
        self.param_lab1_1_1_0.grid(row=1, column=0)
        self.param_chb1_1_0_1.grid(row=0, column=1)
        self.param_chb1_1_1_1.grid(row=1, column=1)
        self.param_lfm1_0.pack()
        self.param_lfm1_1.pack()
        self.param_lfm1.grid(row=1, column=0)

        self.param_lab2_0_0_0.grid(row=0, column=0)
        self.param_lab2_0_1_0.grid(row=1, column=0)
        self.param_lab2_0_2_0.grid(row=2, column=0)
        self.param_lab2_0_3_0.grid(row=3, column=0)
        self.param_lab2_0_4_0.grid(row=4, column=0)
        self.param_lab2_0_5_0.grid(row=5, column=0)
        self.param_lab2_0_6_0.grid(row=6, column=0)
        self.param_lab2_0_7_0.grid(row=7, column=0)
        self.param_lab2_0_8_0.grid(row=8, column=0)
        self.param_lab2_0_9_0.grid(row=9, column=0)
        self.param_lab2_0_10_0.grid(row=10, column=0)
        self.param_cmb2_0_0_1.grid(row=0, column=1)
        self.param_etr2_0_1_1.grid(row=1, column=1)
        self.param_cmb2_0_2_1.grid(row=2, column=1)
        self.param_etr2_0_3_1.grid(row=3, column=1)
        self.param_etr2_0_4_1.grid(row=4, column=1)
        self.param_etr2_0_5_1.grid(row=5, column=1)
        self.param_etr2_0_6_1.grid(row=6, column=1)
        self.param_etr2_0_7_1.grid(row=7, column=1)
        self.param_etr2_0_8_1.grid(row=8, column=1)
        self.param_etr2_0_9_1.grid(row=9, column=1)
        self.param_etr2_0_10_1.grid(row=10, column=1)
        self.param_chb2_1.pack()
        self.param_lfm2_0.pack()
        self.param_lfm2.grid(row=0, column=1, rowspan=2)

        self.param_lab3_0_0_0.grid(row=0, column=0)
        self.param_lab3_0_1_0.grid(row=1, column=0)
        self.param_lab3_0_2_0.grid(row=2, column=0)
        self.param_lab3_0_3_0.grid(row=3, column=0)
        self.param_lab3_0_4_0.grid(row=4, column=0)
        self.param_lab3_0_5_0.grid(row=5, column=0)
        self.param_cmb3_0_0_1.grid(row=0, column=1)
        self.param_etr3_0_1_1.grid(row=1, column=1)
        self.param_etr3_0_2_1.grid(row=2, column=1)
        self.param_chb3_0_3_1.grid(row=3, column=1)
        self.param_etr3_0_4_1.grid(row=4, column=1)
        self.param_chb3_0_5_1.grid(row=5, column=1)
        self.param_lab3_1_0_0.grid(row=0, column=0)
        self.param_lab3_1_1_0.grid(row=1, column=0)
        self.param_lab3_1_2_0.grid(row=2, column=0)
        self.param_lab3_1_3_0.grid(row=3, column=0)
        self.param_rbn3_1_0_1.grid(row=0, column=1)
        self.param_etr3_1_1_1.grid(row=1, column=1)
        self.param_rbn3_1_2_1.grid(row=2, column=1)
        self.param_etr3_1_3_1.grid(row=3, column=1)
        self.param_lab3_2_0_0.grid(row=0, column=0)
        self.param_lab3_2_1_0.grid(row=1, column=0)
        self.param_lab3_2_2_0.grid(row=2, column=0)
        self.param_lab3_2_3_0.grid(row=3, column=0)
        self.param_lab3_2_4_0.grid(row=4, column=0)
        self.param_lab3_2_5_0.grid(row=5, column=0)
        # self.param_lab3_2_6_0.grid(row=6, column=0)
        self.param_etr3_2_0_1.grid(row=0, column=1)
        self.param_etr3_2_1_1.grid(row=1, column=1)
        self.param_etr3_2_2_1.grid(row=2, column=1)
        self.param_etr3_2_3_1.grid(row=3, column=1)
        self.param_etr3_2_4_1.grid(row=4, column=1)
        self.param_etr3_2_5_1.grid(row=5, column=1)
        # self.param_etr3_2_6_1.grid(row=6, column=1)
        self.param_chb3_2.pack()
        self.param_lfm3_0.pack()
        self.param_lfm3_1.pack()
        self.param_lfm3_2.pack()
        self.param_lfm3.grid(row=0, column=2, rowspan=2)

        self.param_btn4.grid(row=2, column=0, columnspan=3)
        self.param_tbl5.grid(row=3, column=0, columnspan=3)

        self.initialize_widgets_of_params()

        #=========================================================
        # Run Frame
        #=========================================================
        # create widgets
        self.run_lfm0 = ttk.Labelframe(self.tab_run, text='select set-up to run')
        self.run_frm1 = tk.Frame(self.tab_run)
        self.run_btn0 = tk.Button(self.run_frm1, text='Find', command=self.find_op_i, font=('Helvetica', '35'))
        self.run_btn1 = tk.Button(self.run_frm1, text='Run', command=self.run_button, font=('Helvetica', '35'))

        # layout widgets
        self.run_btn0.pack(anchor='center')
        self.run_btn1.pack(anchor='center')
        self.run_lfm0.pack()
        self.run_frm1.pack()

        #=========================================================
        # Select frequency Frame
        #=========================================================
        # create widgets
        self.selfq_frame = 1
        self.frm3 = tk.Frame(self.tab_sel_freq)
        self.note3_0 = ttk.Notebook(self.frm3)
        tab3_0_0 = tk.Frame(self.note3_0)

        # layout widgets
        self.frm3.pack()
        # title label
        self.titleLabel = tk.Label(self.tab_sel_freq, text="Preprocessing", font=('Helvetica', '35'))
        self.titleLabel.pack(anchor='center', expand=True)

        #=========================================================
        # Result Frame
        #=========================================================
        # create widgets
        self.res_note = ttk.Notebook(self.tab_result)

        # layout widgets
        self.res_note.pack(expand=True)

        #=========================================================
        # MPE Frame
        #=========================================================
        # create widgets
        self.mpe_lfm0 = ttk.Labelframe(self.tab_mpe, text='Select frequencies for mpe')



        # layout widgets
        self.mpe_lfm0.pack()



        #=========================================================
        # SSI Frame
        #=========================================================
        self.frm3 = tk.Frame(self.tab_SSI)
        self.frm3.pack()
        # title label
        self.titleLabel = tk.Label(self.tab_SSI, text="SSI", font=('Helvetica', '35'))
        self.titleLabel.pack(anchor='center', expand=True)

        #=========================================================
        # SSI_result Frame
        #=========================================================
        self.frm5 = tk.Frame(self.tab_SSIres)
        self.frm5.pack()
        # title label
        self.titleLabel = tk.Label(self.tab_SSIres, text="SSI_result", font=('Helvetica', '35'))
        self.titleLabel.pack(anchor='center', expand=True)

        #=========================================================
        # SSI_geometry Frame
        #=========================================================
        self.frm6 = tk.Frame(self.tab_SSIgeo)
        self.frm6.pack()
        # title label
        self.titleLabel = tk.Label(self.tab_SSIgeo, text="SSI_geometry", font=('Helvetica', '35'))
        self.titleLabel.pack(anchor='center', expand=True)

        # Locate tab
        note.pack(expand=True, fill='both') #, padx=10, pady=10)

    def initialize_widgets_of_params(self):
        # initialize the parameters
        self.param_etr0_0_0_1.insert(0, '200')
        self.param_etr0_0_1_1.insert(0, '0, 60')
        self.param_etr0_1_0_1.insert(0, '0')
        self.param_etr0_1_1_1.insert(0, '0.1')
        self.param_etr0_1_2_1.insert(0, '0')
        self.param_cmb0_1_3_1.set("highpass")

        self.param_etr1_0_0_1.insert(0, '0.01')
        self.param_etr1_0_1_1.insert(0, '')
        self.param_etr1_0_2_1.insert(0, 'find_min')
        self.param_etr1_0_3_1.insert(0, '0, 10')
        self.var_to_mpe_with_mpl.set(True)
        self.var_to_mpe_with_pv.set(False)

        self.var_to_run_fdd.set(True)
        self.param_cmb2_0_0_1.set('EFDD')
        self.param_etr2_0_1_1.insert(0, '4096')
        self.param_cmb2_0_2_1.set('cor')
        self.param_etr2_0_3_1.insert(0, '0.5')
        self.param_etr2_0_4_1.insert(0, '0.1')
        self.param_etr2_0_5_1.insert(0, '0.1')
        self.param_etr2_0_6_1.insert(0, '1.0')
        self.param_etr2_0_7_1.insert(0, '1')
        self.param_etr2_0_8_1.insert(0, '0.85')
        self.param_etr2_0_9_1.insert(0, '3')
        self.param_etr2_0_10_1.insert(0, '20')

        self.var_to_run_ssi.set(True)
        self.param_cmb3_0_0_1.set('SSIdat')
        self.param_etr3_0_1_1.insert(0, '100')
        self.param_etr3_0_2_1.insert(0, '60')
        self.var_to_calc_unc.set(True)
        self.param_etr3_0_4_1.insert(0, '100')
        self.var_to_hide_poles.set(False)
        self.var_of_clustering_method.set('min_distance')
        self.param_etr3_1_1_1.insert(0, '80')
        self.param_etr3_1_3_1.insert(0, '10')
        self.param_etr3_2_0_1.insert(0, '10')
        self.param_etr3_2_1_1.insert(0, '')
        self.param_etr3_2_2_1.insert(0, '10')
        self.param_etr3_2_3_1.insert(0, '10')
        self.param_etr3_2_4_1.insert(0, '100')
        self.param_etr3_2_5_1.insert(0, '10')
        self.param_etr3_2_6_1.insert(0, '0.01')

    def brows_importing_data_folder(self):
        dir = filedialog.askdirectory()
        self.data_etr0_0_1.delete(0, tk.END)
        self.data_etr0_0_1.insert(0, dir)

    def brows_importing_data_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("CSV Files", "*.csv")], initialdir=self.data_etr0_0_1.get())
        fpaths = ''
        for path in paths:
            fpaths += f'"{os.path.basename(path)}" '
        self.data_etr0_1_1.delete(0, tk.END)
        self.data_etr0_1_1.insert(0, fpaths)

    def brows_save_folder(self):
        dir = filedialog.askdirectory()
        self.data_etr0_4_1.delete(0, tk.END)
        self.data_etr0_4_1.insert(0, dir)

    def brows_geometry_file(self):
        path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        self.geo_etr0_0_1.delete(0, tk.END)
        self.geo_etr0_0_1.insert(0, path)

    def get_parameters(self):
        # pre-get some parameters difficult to read
        freqlim = self.param_etr0_0_1_1.get()
        if freqlim == '':
            freqlim = None
        else:
            freqlim = tuple([float(x) for x in freqlim.split(',')])

        sel_freq = self.param_etr1_0_1_1.get()
        if sel_freq == '':
            sel_freq == None
        else:
            sel_freq = list([float(x) for x in sel_freq.split(',')])

        sel_freqlim = self.param_etr1_0_3_1.get()
        if sel_freqlim == '':
            sel_freqlim = None
        else:
            sel_freqlim = tuple([float(x) for x in sel_freqlim.split(',')])

        order_in = self.param_etr1_0_2_1.get()
        if not order_in == 'find_min':
            order_in = list([int(x) for x in order_in.split(',')])

        start_time = self.data_etr0_3_1.get()
        if start_time == '':
            start_time = None

        br_max = self.param_etr3_2_1_1.get()
        if br_max == '':
            br_max = np.inf
        else:
            br_max = int(br_max)

        # get parameters
        param_dict = {
                    'data_folder'           : self.data_etr0_0_1.get(),
                    'data_files'            : shlex.split(self.data_etr0_1_1.get()),
                    'usecols'               : [int(x) for x in self.data_etr0_2_1.get().split(',')],
                    'start_time'            : start_time,
                    'save_dir'              : self.data_etr0_4_1.get(),
                    'geo_path'              : self.geo_etr0_0_1.get(),

                    'freqlim'               : freqlim,
                    'fs'                    : float(self.param_etr0_0_0_1.get()),
                    'q'                     : int(self.param_etr0_1_0_1.get()),
                    'Wn'                    : tuple([float(x) for x in self.param_etr0_1_1_1.get().split(',')]),
                    'order'                 : int(self.param_etr0_1_2_1.get()),
                    'btype'                 : self.param_cmb0_1_3_1.get(),

                    'rtol'                  : float(self.param_etr1_0_0_1.get()),
                    # 'sel_freq'              : sel_freq,
                    # 'order_in'              : order_in,
                    # 'sel_freqlim'           : sel_freqlim,
                    'mpe_with_mpl'          : self.var_to_mpe_with_mpl.get(),
                    'mpe_with_pv'           : self.var_to_mpe_with_pv.get(),

                    'run_fdd'               : self.var_to_run_fdd.get(),
                    'fdd_method'            : self.param_cmb2_0_0_1.get(),
                    'nxseg'                 : float(self.param_etr2_0_1_1.get()),
                    'method_SD'             : self.param_cmb2_0_2_1.get(),
                    'pov'                   : float(self.param_etr2_0_3_1.get()),
                    'DF'                    : float(self.param_etr2_0_4_1.get()),
                    'DF1'                   : float(self.param_etr2_0_5_1.get()),
                    'DF2'                   : float(self.param_etr2_0_6_1.get()),
                    'cm'                    : int(self.param_etr2_0_7_1.get()),
                    'MAClim'                : float(self.param_etr2_0_8_1.get()),
                    'sppk'                  : int(self.param_etr2_0_9_1.get()),
                    'npmax'                 : int(self.param_etr2_0_10_1.get()),

                    'run_ssi'               : self.var_to_run_ssi.get(),
                    'ssi_method'            : self.param_cmb3_0_0_1.get(),
                    'br'                    : int(self.param_etr3_0_1_1.get()),
                    'ord_max_run'           : int(self.param_etr3_0_2_1.get()),
                    'calc_unc'              : self.var_to_calc_unc.get(),
                    'nb'                    : int(self.param_etr3_0_4_1.get()),
                    'hide_poles'            : self.var_to_hide_poles.get(),
                    'clustering_method'     : self.var_of_clustering_method.get(),
                    'percentile'            : float(self.param_etr3_1_1_1.get()),
                    'min_cluster_elements'  : int(self.param_etr3_1_3_1.get()),
                    'br_min'                : int(self.param_etr3_2_0_1.get()),
                    'br_max'                : br_max,
                    'br_step'               : int(self.param_etr3_2_2_1.get()),
                    'ord_min_find'          : int(self.param_etr3_2_3_1.get()),
                    'ord_max_find'          : int(self.param_etr3_2_4_1.get()),
                    'ord_step_find'         : int(self.param_etr3_2_5_1.get()),
                    # 'threshold'             : float(self.param_etr3_2_6_1.get())
                    }

        return param_dict

    def define_set_up(self):
        for i in range(1, len(self.set_up_dict) + 2):
            name = f'set_up_{i}'
            if name not in self.set_up_dict.keys():
                break
        self.set_up_dict[name] = {}

        # construct a widget to run analysis by current parameter set
        self.set_up_dict[name]['var_to_run'] = tk.BooleanVar()
        self.set_up_dict[name]['chb_to_run'] = tk.Checkbutton(
            self.run_lfm0,
            text=name,
            variable=self.set_up_dict[name]['var_to_run']
            )
        self.set_up_dict[name]['chb_to_run'].pack()
        self.set_up_dict[name]['var_to_run'].set(True)

        # get parameters
        self.set_up_dict[name]['params'] = self.get_parameters()
        self.param_tbl5.insert('', 'end', text=name)

    # ==========================================================================
    # functions to run button
    # ==========================================================================
    def run_button(self):
        for key in self.set_up_dict.keys():
            if self.set_up_dict[key]['var_to_run'].get():
                params = self.set_up_dict[key]['params']

                for k in params.keys():
                    print(f"{k}: {params[k]}")
                self._save_params(params)

                ss, fdd, efdd, fsdd, ssidat, ssicov = self._pre_process(params)

                self._post_process(params, ss, efdd, ssidat)

                self.set_up_dict[key]['var_to_run'].set(False)

        messagebox.showinfo(title='Run', message='Done!')

    def _save_params(self, params):
        p = params
        os.makedirs(p['save_dir'], exist_ok=True)
        output_path_param = p['save_dir'] + '/' + "parameter_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".txt"
        with open(output_path_param, 'w') as f:
            indent = "  "
            f.write("・EFDD parameter\n")
            f.write(f"{indent}nxseg = {p['nxseg']}\n")
            f.write(f"{indent}method_SD = {p['method_SD']}\n")
            f.write(f"{indent}pov = {p['pov']}\n")
            f.write(f"{indent}DF = {p['DF']}\n")
            f.write(f"{indent}DF1 = {p['DF1']}\n")
            f.write(f"{indent}DF2 = {p['DF2']}\n")
            f.write(f"{indent}cm = {p['cm']}\n")
            f.write(f"{indent}MAClim = {p['MAClim']}\n")
            f.write(f"{indent}sppk = {p['sppk']}\n")
            f.write(f"{indent}npmax = {p['npmax']}\n")
            f.write("・SSI parameter\n")
            f.write(f"{indent}fs = p['fs']\n")
            f.write(f"{indent}br = p['br']\n")
            f.write(f"{indent}ordmax = p['ord_max_run']\n")
            f.write(f"{indent}calc_unc = p['calc_unc']\n")
            f.write(f"{indent}nb = p['nb']\n")
            f.write(f"{indent}hide_poles = p['hide_poles']\n")
            f.write("・Detrending, decimation, filtering parameter\n")
            f.write(f"{indent}q = p['q']\n")
            f.write(f"{indent}Wn = p['Wn']\n")
            f.write(f"{indent}order = p['order']\n")
            f.write(f"{indent}btype = p['btype']\n")
            f.write(f"{indent}freqlim = p['freqlim']\n")
            f.write("・clustering parameter\n")
            f.write(f"{indent}percentile = p['percentile']\n")

    def _pre_process(self, params):
        # short cut of params
        p = params

        # Create single set_up
        ss = self.create_single_setup(p['start_time'], p['data_files'], p['data_folder'], p['usecols'], p['fs'])

        # Detrend, freq_ilter or decimate
        ss.detrend_data()
        if p['q']:
            ss.decimate_data(q=p['q'])
        if p['order']:
            ss.freq_filter_data(Wn=p['Wn'], order=p['order'], btype=p['btype'])

        # Initialise the algorithms
        fdd = FDD(name="FDD", nxseg=p['nxseg'], method_SD=p['method_SD'], pov=p['pov'])
        efdd = EFDD(name="EFDD", nxseg=p['nxseg'], method_SD=p['method_SD'], pov=p['pov'], DF1=p['DF1'], DF2=p['DF2'], cm=p['cm'], MAClim=p['MAClim'], sppk=p['sppk'], npmax=p['npmax'],)
        fsdd = FSDD(name="FSDD", nxseg=p['nxseg'], method_SD=p['method_SD'], pov=p['pov'])
        ssidat = SSIdat(name="SSIdat", br=p['br'], ordmax=p['ord_max_run'], calc_unc=p['calc_unc'])
        ssicov = SSIcov(name="SSIcov", br=p['br'], ordmax=p['ord_max_run'], calc_unc=p['calc_unc'])

        # Add algorithms to the single setup class
        ss.add_algorithms(fdd, efdd, fsdd, ssidat, ssicov)

        # Run selected method
        if p['run_fdd']:
            ss.run_by_name(p['fdd_method'])
        if p['run_ssi']:
            ss.run_by_name(p['ssi_method'])

        return ss, fdd, efdd, fsdd, ssidat, ssicov

    def _post_process(self, params, ss, efdd, ssidat):
        """Automatically extract modal parameters

        Args:
            params (dictionary): _description_
            ss (pyoma2 SingleSetup class): _description_
            efdd (pyoma2 EFDD class): _description_
            ssidat (pyoma2 SSIdat class): _description_
        """
        # make directory to save the results
        efdd_save_folder = params['save_dir'] + '/EFDD'
        ssi_save_folder  = params['save_dir'] + '/SSI (auto)'
        os.makedirs(efdd_save_folder, exist_ok=True)
        os.makedirs(ssi_save_folder , exist_ok=True)

        # post-process
        # save stabilization and damping diagram
        self.save_stab_and_damp_diagram(params, ssidat, ssi_save_folder)

        # clustering
        clusters, min_distances, hclus_threshold, Z = self.hierarchical_clustering(params, ssidat)
        self.save_clustering_info(params, ssi_save_folder, min_distances, hclus_threshold, Z)

        large_clusters = self.delete_small_clusters(params, clusters)
        self.save_cluster(params, ssi_save_folder, clusters, large_clusters, outlier=False)

        large_clusters_without_outliers = self.cutoff_outliers_from_clusters(large_clusters)
        self.save_cluster(params, ssi_save_folder, clusters, large_clusters_without_outliers, outlier=True)

        # mpe
        self.mpe_from_clusters(params, ssi_save_folder, large_clusters_without_outliers, ss, ssidat, efdd)

    def create_single_setup(self, start_time, file_names, folder_path, usecols, fs):
        if start_time:
            start_time_obj = datetime.strptime(start_time, "%H:%M:%S")
            df_data = []
            for name in file_names:
                file_name = os.path.join(folder_path, name)
                df = pd.read_csv(file_name, usecols=usecols, header=0)
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

        return ss

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

    def stab_poles_to_df(self, hide_poles, ssidat_result, ssidat):
        Fn = ssidat_result.Fn_poles
        Lab = ssidat_result.Lab
        step = ssidat.run_params.step
        Fn_std = ssidat_result.Fn_poles_std
        Fns_stab = np.where(Lab == 1, Fn, np.nan)
        Fns_unstab = np.where(Lab == 0, Fn, np.nan)

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
                xerr = []

            for i, (xi, yi) in enumerate(zip(x, y)):
                if not np.isnan(xi):
                    err = xerr[i] if i < len(xerr) else None
                    data_stab.append([xi, yi, None, err])

            for i, (xi1, yi1) in enumerate(zip(x1, y1)):
                if not np.isnan(xi1):
                    err = xerr[len(x) + i] if len(xerr) > len(x) + i else None
                    data_stab.append([xi1, None, yi1, err])

        df_stab_poles = pd.DataFrame(data_stab, columns=['Frequency', 'Stable', 'Unstable', 'Error',])

        return df_stab_poles

    def freq_damp_poles_to_df(self, hide_poles, ssidat_result):
        Fn = ssidat_result.Fn_poles
        Xi = ssidat_result.Xi_poles
        Lab = ssidat_result.Lab
        Fn_std = ssidat_result.Fn_poles_std

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
                f_err =[]

            for i, (f_st, d_st) in enumerate(zip(a_flat, aa_flat)):
                if not np.isnan(f_st):
                    err = f_err[i] if f_err is not [] and i < len(f_err) else None
                    data_FnXi.append([f_st, d_st, None, err])

            for i, (f_unst, d_unst) in enumerate(zip(b_flat, bb_flat)):
                if not np.isnan(f_unst):
                    err = f_err[len(a_flat) + i] if f_err is not [] and (len(a_flat) + i) < len(f_err) else None
                    data_FnXi.append([f_unst, None, d_unst, err])

        df_freq_damp_poles = pd.DataFrame(data_FnXi, columns=['Frequency', 'Stable', 'Unstable', 'Error'])

        return df_freq_damp_poles

    def hierarchical_clustering(self, params, ssidat):
        df_poles = self.poles_to_df(ssidat.result)
        valid_poles_list = self.extract_valid_poles(ssidat.result)
        min_distances, threshold, Z, clusters = self.single_linkage_on_poles_list(params, valid_poles_list, df_poles)

        c = pd.DataFrame({'cluster_id': clusters})
        df_clustered_poles = pd.concat([c, df_poles], axis=1)

        return df_clustered_poles, min_distances, threshold, Z

    def poles_to_df(self, ssidat_result):
        Lab     = ssidat_result.Lab
        Fn      = ssidat_result.Fn_poles
        Phi     = ssidat_result.Phi_poles
        Xi      = ssidat_result.Xi_poles
        Fn_std  = ssidat_result.Fn_poles_std
        Phi_std = ssidat_result.Phi_poles_std
        Xi_std  = ssidat_result.Xi_poles_std

        coords = np.where(Lab == 1)

        Fn  = Fn[coords]
        Phi = list(Phi[coords])
        Xi  = Xi[coords]
        if Fn_std is not None:
            Fn_std  = Fn_std[coords]
            Phi_std = list(Phi_std[coords])
            Xi_std  = Xi_std[coords]
        else:
            Fn_std  = np.nan
            Phi_std = np.nan
            Xi_std  = np.nan

        df_poles = pd.DataFrame()
        df_poles['row']     = coords[0]
        df_poles['col']     = coords[1]
        df_poles['Fn']      = Fn
        df_poles['Phi']     = Phi
        df_poles['Xi']      = Xi
        df_poles['Fn_std']  = Fn_std
        df_poles['Phi_std'] = Phi_std
        df_poles['Xi_std']  = Xi_std

        return df_poles

    def extract_valid_poles(self, ssidat_result):
        Fn  = ssidat_result.Fn_poles
        Phi = ssidat_result.Phi_poles
        Xi  = ssidat_result.Xi_poles
        Lab = ssidat_result.Lab

        poles_list = []
        num_poles, num_models = Fn.shape

        for indx in range(num_poles):
            for n in range(num_models):
                if Lab[indx, n] == 1:
                    freq = Fn[indx, n]
                    if not np.isnan(freq):
                        phi = Phi[indx, n, :]  # shape: (3,)
                        xi  = Xi[indx, n]
                        poles_list.append((indx, n, freq, phi, xi))

        return poles_list

    def compute_min_distances(self, poles_list):
        result_list = []
        grouped = {}

        for indx, n, freq, phi, _ in poles_list: # group poles every model order
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

    def valid_compute_min_distances(self, df_poles):
        arr_model_order = df_poles['col'].unique()
        arr_pole_index  = df_poles['row'].unique()

        df_min_distance = pd.DataFrame(
            np.nan,
            index  =arr_model_order,
            columns=arr_pole_index
        )

        for o in arr_model_order:
            if o == 0:
                continue

            arr_pole_index = df_poles.query('col == @o')['row'].unique()

            for i in arr_pole_index:
                pole = df_poles.query('col == @o & row == @i')
                freq_i = pole['Fn' ].iloc[0]
                phi_i  = pole['Phi'].iloc[0]

                other_poles = df_poles.query('col < @o')

                for j in range(len(other_poles)):
                    dist_i = np.array([])
                    s = other_poles.iloc[j, :]
                    freq_j = s['Fn' ]
                    phi_j  = s['Phi']

                    mac = gen.MAC(phi_i, phi_j)
                    dist_ij = abs(freq_i - freq_j) / freq_i + (1 - mac)

                    dist_i = np.hstack([dist_i, dist_ij])

                if len(other_poles) > 0:
                    print(min_dist_i)
                    min_dist_i = dist_i.min()

                    df_min_distance.loc[o, i] = min_dist_i

        return df_min_distance

    def compute_distance_matrix(self, poles_list):
        N = len(poles_list)
        distance_matrix = np.zeros((N, N))

        for i in range(N):
            _, _, freq_i, phi_i, _ = poles_list[i]
            for j in range(i + 1, N):
                _, _, freq_j, phi_j, _ = poles_list[j]
                mac = gen.MAC(phi_i, phi_j)
                dist = abs(freq_i - freq_j) / freq_i + (1 - mac)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix

    def single_linkage_on_poles_list(self, params, poles_list, df_poles):
        clustering_method = params['clustering_method']
        percentile        = params['percentile']

        distance_matrix = self.compute_distance_matrix(poles_list)
        condensed_dist = squareform(distance_matrix)
        Z = linkage(condensed_dist, method='single')

        min_distances = self.compute_min_distances(poles_list)
        d_vec = np.array([entry['distance'] for entry in min_distances])

        if clustering_method == 'min_distance':
            threshold = np.percentile(d_vec, q=percentile)
        elif clustering_method == 'mean':
            df_min_distance = self.valid_compute_min_distances(df_poles)
            d_vec = df_min_distance.values
            print(f'd_vec: {d_vec}')
            mean = np.nanmean(d_vec)
            std  = np.nanstd(d_vec)
            print(f'mean: {mean}')
            print(f'mean: {std}')
            threshold = mean + 2 * std

        clusters = fcluster(Z, t=threshold, criterion='distance')

        return min_distances, threshold, Z, clusters

    def save_clustering_info(self, params, ssi_save_folder, min_distances, threshold, Z):
        p = params

        # compute minimam distance and output to csv
        df_min_distances = self.min_distances_with_percentiles_to_df(min_distances)
        outnm = "SSIdat_min_distances_" + os.path.splitext(os.path.basename(p['data_folder']))[0] + ".csv"
        output_path = ssi_save_folder + "/" + outnm
        df_min_distances.to_csv(output_path, index=False, encoding="utf-8-sig")

        # output dendrogram
        fig = self.plot_Hclus(Z, threshold)
        output_Hclus_fig = ssi_save_folder + "/" + "Hclus_" + os.path.splitext(os.path.basename(p['data_folder']))[0] + ".png"
        fig.savefig(output_Hclus_fig, dpi=300, bbox_inches='tight')
        plt.close(fig)

        zoom_max = threshold * 1.2
        fig_zoom = self.plot_Hclus(Z, threshold, zoom_ylim=(0, zoom_max))
        output_zoom_fig = ssi_save_folder + "/" + "Hclus_zoom_" + os.path.splitext(os.path.basename(p['data_folder']))[0] + ".png"
        fig_zoom.savefig(output_zoom_fig, dpi=300, bbox_inches='tight')
        plt.close(fig_zoom)

    def plot_Hclus(self, Z, threshold, zoom_ylim=None):
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["lines.linewidth"] = 0.7

        fig = plt.figure(figsize=(2.75, 1.95))
        dendrogram(Z, no_labels=True, color_threshold=threshold)
        plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold = {threshold:.2f}')
        #plt.title('Hierarchical Clustering (Single Linkage)')
        plt.xlabel('Element labels')
        plt.ylabel('D$_i$$_,$$_j$')
        if zoom_ylim is not None:
            plt.ylim(zoom_ylim)
        plt.grid(True)
        plt.tight_layout()

        plt.rcdefaults()

        return fig

    def min_distances_with_percentiles_to_df(self, min_distances):
        d_vec = np.array([entry['distance'] for entry in min_distances])
        df_dist = pd.DataFrame(d_vec, columns=["distance"])

        percentiles = list(range(20, 100, 5))  # 50,55,...,95
        perc_values = {f"p{p}": [np.percentile(d_vec, p)] for p in percentiles}
        df_perc = pd.DataFrame(perc_values)

        df_out = pd.concat([df_dist, df_perc], axis=1)

        return df_out

    def delete_small_clusters(self, params, clusters):
        min_cluster_elements = params['min_cluster_elements']

        counts = clusters['cluster_id'].value_counts()
        mask = counts >= min_cluster_elements
        large_cluster_id = counts[mask].index.tolist()

        large_clusters = clusters.query('cluster_id in @large_cluster_id')

        return large_clusters

    def cutoff_outliers_from_clusters(self, clusters):
        c_id = clusters['cluster_id'].unique()

        cluster_without_outlier = pd.DataFrame(columns=clusters.columns)
        for n in c_id:
            df = clusters[clusters['cluster_id'] == n]
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
        df_left = clusters[['cluster_id', 'col', 'row', 'Fn', 'Xi']].copy()
        df_left = df_left.set_axis(['ClusterID', 'ModelIndex', 'PoleIndex', 'Frequency', 'Damping'], axis='columns', copy=False)

        # df_center
        count_of_cluster_elements = large_clusters['cluster_id'].value_counts().sort_index()
        df_center = count_of_cluster_elements.reset_index()
        df_center.columns = ['ClusterNo.', 'Elements']

        # df_right
        df_right = self.make_df_right_of_Hclus(large_clusters, df_center)

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
                large_clusters['cluster_id'] == n
            ].reset_index(drop=True)

            df_freqs_in_large_clusters = pd.concat([df_freqs_in_large_clusters, df_cluster['Fn']], axis=1)

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
            df = df.sort_values(['Xi'])

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
        cluster_num = df_mp['cluster_id'].to_numpy()

        elements    = clusters['cluster_id'].value_counts().reindex(index=cluster_num).to_numpy()
        CV_freq, CV_damp = self.compute_CV(clusters, cluster_num)
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
        df_phi.loc['1-MPD']      = 1 - df_phi.loc['MPD']
        df_phi.loc['MCF']        = np.round(mcf_list, 6)
        df_phi.loc['FR']         = np.round(FR_list, 6)
        df_phi.loc[' ']          = np.nan
        for i, name in enumerate(sensor_names):
            df_phi.loc[name] = list(np.round(Phi.real[i], 6))

        return df_phi

    def compute_CV(self, clusters, cluster_num):
        CV = clusters.groupby('cluster_id')[['Fn', 'Xi']].agg(
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

    def save_mode_shape(self, ss, geo, mode_nr, ssidat_result, save_folder, backend_mpl, backend_pv):
        if geo == '':
            return

        ss.def_geo2_by_file(geo)

        if  backend_mpl:
            _, _ = ss.plot_geo2_mpl(scaleF=2)
            for i in range(1, mode_nr + 1):
                fig_shape, _ = ss.plot_mode_geo2_mpl(algo_res=ssidat_result, mode_nr=i, view="3D", scaleF=3)
                freq = ssidat_result.Fn[i - 1]
                Phi_filename = f"{freq:.2f}Hz.png"
                fig_shape.savefig(os.path.join(save_folder, Phi_filename), dpi=300)

        if backend_pv:
            _ = ss.plot_geo2(scaleF=2)
            for i in range(1, mode_nr + 1):
                plot = ss.plot_mode_geo2(algo_res=ssidat_result, mode_nr=i, scaleF=3)
                anim = ss.anim_mode_geo2(algo_res=ssidat_result, mode_nr=i, scaleF=3)
                freq = ssidat_result.Fn[i - 1]
                plot_filename = f"{freq:.2f}Hz.gltf"
                anim_filename = f"anim_{freq:.2f}Hz.gltf"
                plot.export_gltf(os.path.join(save_folder, plot_filename))
                anim.export_gltf(os.path.join(save_folder, anim_filename))

    def idx_of_the_nearest(self, data, value):
        if type(value) == float:
            idx = np.argmin(np.abs(np.array(data) - value))
            return idx

        if type(value) == list:
            idx = [None]*len(value)
            for i in range(len(value)):
                idx[i] = np.argmin(np.abs(np.array(data) - value[i]))
            return idx

    def compute_FR(self, efdd_result, selected_freq):
        freq = efdd_result.freq
        S_val = efdd_result.S_val[0, 0, :]

        sel_freq_idx = self.idx_of_the_nearest(freq, list(selected_freq))

        sel_S_val = S_val[sel_freq_idx]
        max_S_val = S_val.max()

        FR = sel_S_val / max_S_val
        print('Selected frequency : ', selected_freq, '\n', 'FR : ', FR)

        return FR


    # ==========================================================================
    # functions to find optimal numbers of block rows
    # ==========================================================================
    def find_op_i(self):
        for key in self.set_up_dict.keys():
            if self.set_up_dict[key]['var_to_run'].get():
                params = self.set_up_dict[key]['params']

                for k in params.keys():
                    print(f"{k}: {params[k]}")

                self._find_op_i(params)

                self.set_up_dict[key]['var_to_run'].set(False)

        messagebox.showinfo(title='find', message='Done!')

    def _find_op_i(self, params):
        p = params
        find_i_save_folder  = p['save_dir'] + '/find'
        os.makedirs(find_i_save_folder , exist_ok=True)

        # Create single set_up
        ss = self.create_single_setup(p['start_time'], p['data_files'], p['data_folder'], p['usecols'], p['fs'])

        # Detrend, freq_ilter or decimate
        ss.detrend_data()
        if p['q']:
            ss.decimate_data(q=p['q'])
        if p['order']:
            ss.freq_filter_data(Wn=p['Wn'], order=p['order'], btype=p['btype'])

        #====================================================================================================================
        # find optimal i
        #====================================================================================================================
        op_i_list, i_max, df_kappa, df_delta_kappa, _ \
            = self.find_all_optimal_i(Y=ss.data.T, ord_min=p['ord_min_find'], ord_max=p['ord_max_find'], ord_step=p['ord_step_find'], \
                                method="dat", i_min=p['br_min'], i_max_lim=p['br_max'], i_step=p['br_step'], nb=p['nb'])

        #====================================================================================================================
        # output the results
        #====================================================================================================================
        # output csv of kappa
        outnm_kappa = ("kappa_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".csv")
        output_path_kappa = find_i_save_folder + '/' + outnm_kappa
        self.df_to_csv_including_idx_and_col_name(df_kappa, output_path_kappa)

        # output csv of delta_kappa
        outnm_delta_kappa = ("delta_kappa_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".csv")
        output_path_delta_kappa = find_i_save_folder + '/' + outnm_delta_kappa
        self.df_to_csv_including_idx_and_col_name(df_delta_kappa, output_path_delta_kappa)

        # output contour of kappa
        contour = self.df_to_contour(df_kappa, log10_z=True)
        plt.colorbar(contour, label='Contour\nlog$_1$$_0$(κ)')
        plt.xticks(ticks=range(p['ord_min_find'], p['ord_max_find'] + 1, p['ord_step_find']))
        plt.yticks(ticks=range(p['br_min'] , i_max        + 1, p['br_step'] ))
        plt.xlabel('Model order n')
        plt.ylabel('Block rows i')
        plt.title('Contour Plot of κ over (n, i)')
        plt.grid(True)
        plt.tight_layout()
        output_name = ("kappa_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".png")
        output_path = find_i_save_folder + '/' + output_name
        plt.savefig(output_path, dpi=300)
        plt.close()

        # optimal i
        delta_kappa_arr_of_op_i \
            = self.df_at_method_by_list(df_delta_kappa, op_i_list, df_delta_kappa.columns.to_list())
        data = {'model_order': df_delta_kappa.columns,
                'optimal_br' : op_i_list,
                'delta_kappa': delta_kappa_arr_of_op_i}
        df_optimal_i = pd.DataFrame(data)
        print("最適なハンケル行列数", '\n', df_optimal_i[['model_order', 'optimal_br']])

        # parameters
        param_rows = [
            ["i_min", p['br_min']],
            ["i_max", i_max],
            ["i_step", p['br_step']],
            ["ord_min", p['ord_min_find']],
            ["ord_max", p['ord_max_find']],
            ["ord_step", p['ord_step_find']]
        ]
        df_params = pd.DataFrame(param_rows, columns=[' ', ' '])

        # output optimal i and parameters
        df_i = pd.concat([df_optimal_i, pd.DataFrame(columns=[' ']), df_params], axis=1)
        outnm_find_i = ("find_i_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".csv")
        output_path_find_i = find_i_save_folder + '/' + outnm_find_i
        df_i.to_csv(output_path_find_i, index=False)

    def compute_condition_number(self, H, S=None, ord_k=None):
        if S is None:
            _, S, _ = np.linalg.svd(H, full_matrices=False)
        if ord_k is not None:
            S = S[:ord_k]
        if np.min(S) == 0:
            return np.inf
        return np.max(S) / np.min(S)

    def is_any_cols_of_dataframe_have_false_only(self, df):
        values_T = df.values.T

        is_cols_of_dataframe_have_false_only     = np.all(values_T == False, axis=1)
        is_any_cols_of_dataframe_have_false_only = np.any(is_cols_of_dataframe_have_false_only)

        return is_any_cols_of_dataframe_have_false_only

    def list_of_min_index_of_True_in_2D_array(self, arr_2D, return_array_along='row'):
        if return_array_along   == 'col':
            pass
        elif return_array_along == 'row':
            arr_2D = arr_2D.T

        n_row, n_col = arr_2D.shape

        idx_of_row            = np.arange(n_col)
        idx_matrix_of_mask_2D = np.tile(idx_of_row, (n_row, 1))

        masked_idx_matrix = np.where(arr_2D, idx_matrix_of_mask_2D, np.nan)

        min_idx_arr = np.nanmin(masked_idx_matrix, axis=1)
        min_idx_list = [min_idx.astype(int) if ~np.isnan(min_idx) else np.nan for min_idx in min_idx_arr]

        return min_idx_list

    def find_all_optimal_i(self, Y, ref_ind=None, ord_min=1, ord_max=100, ord_step=1, method="dat", i_min=0, i_max_lim=np.inf, i_step=10, nb=50, threshold=1e-2):
        Yref   = Y[ref_ind, :] if ref_ind is not None else Y
        n_list = list(range(ord_min, ord_max + 1, ord_step))
        length_n_list = len(n_list)

        br_i = i_min

        df_kappa                    = pd.DataFrame(columns=n_list)
        df_delta_kappa              = pd.DataFrame(columns=n_list)
        df_op_i_mask                = pd.DataFrame(columns=n_list)
        df_kappa.index.name         = 'block rows i'
        df_kappa.columns.name       = 'model order n'
        df_delta_kappa.index.name   = 'block rows i'
        df_delta_kappa.columns.name = 'model order n'
        df_op_i_mask.index.name     = 'block rows i'
        df_op_i_mask.columns.name   = 'model order n'

        if not i_step == 1:
            while True:
                # build a hankel matrix and compute its singular value decomposition
                print(f"br_i = {br_i}")
                try:
                    is_S1_calculated = False

                    H1, _ = ssi.build_hank(Y=Y, Yref=Yref, br=br_i  , method=method, calc_unc=False, nb=nb)
                    _, S1, _ = np.linalg.svd(H1, full_matrices=False)

                    is_S1_calculated = True

                    H2, _ = ssi.build_hank(Y=Y, Yref=Yref, br=br_i+1, method=method, calc_unc=False, nb=nb)
                    _, S2, _ = np.linalg.svd(H2, full_matrices=False)
                except Exception as e:
                    print(f"i={br_i} でエラー: {e}")
                    # save the result
                    if is_S1_calculated:
                        kappa_arr1 = np.array([float('nan')] * length_n_list)
                        for iter, n in enumerate(n_list):
                            try:
                                kappa_arr1[iter] = self.compute_condition_number([], S1, ord_k=n)
                            except Exception as e:
                                print(f"i={br_i}, ord_n={n} でエラー: {e}")
                        df_kappa.loc[br_i] = kappa_arr1
                    else:
                        df_kappa.loc[br_i] = float('nan')
                    df_delta_kappa.loc[br_i] = float('nan')
                    df_op_i_mask.loc[br_i]   = False

                    if br_i < i_max_lim:
                        br_i += i_step
                        continue
                    else:
                        i_max = br_i
                        break

                # if a hankel matrix and its svd are calculated, compute condition number, kappa
                kappa_arr1 = np.array([float('nan')] * length_n_list)
                kappa_arr2 = np.array([float('nan')] * length_n_list)
                for iter, n in enumerate(n_list):
                    try:
                        kappa_arr1[iter] = self.compute_condition_number([], S1, ord_k=n)
                    except Exception as e:
                        print(f"i={br_i}, ord_n={n} でエラー: {e}")
                    try:
                        kappa_arr2[iter] = self.compute_condition_number([], S2, ord_k=n)
                    except Exception as e:
                        print(f"i={br_i+1}, ord_n={n} でエラー: {e}")

                # compute delta kappa
                k1 = kappa_arr1
                k2 = kappa_arr2
                delta_kappa_arr = abs(k1 - k2) / k1

                op_i_mask = delta_kappa_arr <= threshold

                # save the result
                df_kappa.loc[br_i]       = kappa_arr1
                # df_kappa.loc[br_i+1]     = kappa_arr2
                df_delta_kappa.loc[br_i] = delta_kappa_arr
                df_op_i_mask.loc[br_i]   = op_i_mask

                if br_i < i_max_lim and self.is_any_cols_of_dataframe_have_false_only(df_op_i_mask):
                    br_i += i_step
                    continue
                else:
                    i_max = br_i
                    break

        elif i_step == 1:
            while True:
                # build hankel matrix
                print(f"br_i = {br_i}")
                try:
                    H, _ = ssi.build_hank(Y=Y, Yref=Yref, br=br_i, method=method, calc_unc=False, nb=nb)
                except Exception as e:
                    print(f"i={br_i} でエラー: {e}")

                    # save the result
                    df_kappa.loc[br_i]       = float('nan')
                    if not br_i  == i_min:
                        df_delta_kappa.loc[pre_br_i] = float('nan')
                        df_op_i_mask.loc[pre_br_i]   = False

                    if br_i < i_max_lim:
                        pre_br_i = br_i
                        br_i += i_step
                        pre_kappa_arr = np.array([float('nan')] * length_n_list)
                        continue
                    else:
                        i_max = br_i
                        break

                # if a hankel matrix is built, compute condition number, kappa
                kappa_arr = np.array([float('nan')] * length_n_list)
                _, S, _ = np.linalg.svd(H, full_matrices=False)
                for iter, n in enumerate(n_list):
                    try:
                        kappa_arr[iter] = self.compute_condition_number([], S, ord_k=n)
                    except Exception as e:
                        print(f"i={br_i}, ord_n={n} でエラー: {e}")

                # compute delta kappa
                if not br_i == i_min:
                    k1 = pre_kappa_arr
                    k2 = kappa_arr
                    delta_kappa_arr = abs(k1 - k2) / k1

                    op_i_mask = delta_kappa_arr <= threshold

                # save the result
                df_kappa.loc[br_i] = kappa_arr
                if not br_i  == i_min:
                    df_delta_kappa.loc[pre_br_i] = delta_kappa_arr
                    df_op_i_mask.loc[pre_br_i]   = op_i_mask

                if br_i < i_max_lim and self.is_any_cols_of_dataframe_have_false_only(df_op_i_mask):
                    pre_br_i = br_i
                    br_i += i_step
                    pre_kappa_arr = kappa_arr
                    continue
                else:
                    i_max = br_i
                    break

        op_i_idx = self.list_of_min_index_of_True_in_2D_array(df_op_i_mask.values)
        i_arr = df_delta_kappa.index.to_numpy()
        op_i_list  = [i_arr[idx] if ~np.isnan(idx) else np.nan for idx in op_i_idx]

        return op_i_list, i_max, df_kappa, df_delta_kappa, df_op_i_mask

    def df_to_csv_including_idx_and_col_name(self, df, file_path, **kwargs):
        idx = df.index.to_list()
        col = df.columns.to_list()

        new_idx = [' '] + [str(df.index.name)  ] + [' '] * (len(idx) - 1)
        new_col = [' '] + [str(df.columns.name)] + [' '] * (len(col) - 1)

        new_df_idx = pd.DataFrame([[i] for i in idx], index=idx, columns=['col'])
        new_df_col = pd.DataFrame([[''] + col], columns=['col'] + col)

        new_df = pd.concat([new_df_idx, df    ], axis=1)
        new_df = pd.concat([new_df_col, new_df], axis=0)

        new_df = new_df.set_axis(new_idx, axis='index'  )
        new_df = new_df.set_axis(new_col, axis='columns')

        new_df.to_csv(file_path, **kwargs)

    def df_to_contour(self, df, number_of_x_meshgrid=101, number_of_y_meshgrid=101, levels=20, method='cubic', cmap='viridis', log10_z=False):
        x = df.columns.to_numpy().astype('float')
        y = df.index.to_numpy().astype('float')
        z = df.values

        if log10_z:
            z = np.log10(z)

        # create grid points of x and y
        X, Y   = np.meshgrid(x, y)
        points = np.column_stack((X.ravel(), Y.ravel()))
        # values of the points
        values = z.ravel()
        # drop Nan
        nan_mask = ~np.isnan(values)
        points = points[nan_mask]
        values = values[nan_mask]

        # create mesh-grid of the contour
        x_linspace = np.linspace(min(x), max(x), number_of_x_meshgrid)
        y_linspace = np.linspace(min(y), max(y), number_of_y_meshgrid)
        grid_x, grid_y = np.meshgrid(x_linspace, y_linspace)
        # values of the mesh-grid
        grid_z = griddata(points, values, (grid_x, grid_y), method=method)

        contour = plt.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap)

        return contour

    def df_at_method_by_list(self, df, idx_list, col_list):
        l = []
        for idx, col in zip(idx_list, col_list):
            try:
                x = df.loc[idx, col]
            except:
                x = float('nan')
            l.append(x)

        return l


# OmaApp を実行
if __name__ == "__main__":
    root = tk.Tk()
    app = OmaApp(root)
    root.mainloop()