import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import os
import shlex
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

from pyoma2.algorithms import FDD, EFDD, FSDD, SSIdat, SSIcov
from pyoma2.setup import SingleSetup
from pyoma2.functions import gen

class OmaApp:
    def __init__(self, master):
        self.master = master
        master.title("OMA App")

        self.set_up_dict = {}
        self.var_param_set_to_run = {}
        self.chb_param_set_to_run = {}

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
        self.data_etr0_0_1.grid(row=0, column=1)
        self.data_etr0_1_1.grid(row=1, column=1)
        self.data_etr0_2_1.grid(row=2, column=1)
        self.data_etr0_3_1.grid(row=3, column=1)
        self.data_etr0_4_1.grid(row=4, column=1)
        self.data_btn0_0_2.grid(row=0, column=2)
        self.data_btn0_1_2.grid(row=1, column=2)
        self.data_btn0_4_2.grid(row=4, column=2)
        self.data_frm0.grid_columnconfigure(1, weight=1)
        self.data_frm0.pack()

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
        self.geo_etr0_0_1.grid(row=0, column=1)
        self.geo_btn0_0_2.grid(row=0, column=2)
        self.geo_frm0.grid_columnconfigure(1, weight=1)
        self.geo_frm0.pack()

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
        self.param_lab1_0_0_0 = tk.Label(self.param_lfm1_0, text='rtol')
        self.param_lab1_0_1_0 = tk.Label(self.param_lfm1_0, text='sel_freq')
        self.param_lab1_0_2_0 = tk.Label(self.param_lfm1_0, text='order_in')
        self.param_lab1_0_3_0 = tk.Label(self.param_lfm1_0, text='sel_freqlim')
        self.param_etr1_0_0_1 = tk.Entry(self.param_lfm1_0)
        self.param_etr1_0_1_1 = tk.Entry(self.param_lfm1_0)
        self.param_etr1_0_2_1 = tk.Entry(self.param_lfm1_0)
        self.param_etr1_0_3_1 = tk.Entry(self.param_lfm1_0)

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
        self.param_lab3_0_0_0 = tk.Label(self.param_lfm3_0, text='method')
        self.param_lab3_0_1_0 = tk.Label(self.param_lfm3_0, text='br')
        self.param_lab3_0_2_0 = tk.Label(self.param_lfm3_0, text='ord min')
        self.param_lab3_0_3_0 = tk.Label(self.param_lfm3_0, text='ord max')
        self.param_lab3_0_4_0 = tk.Label(self.param_lfm3_0, text='ord step')
        self.param_lab3_0_5_0 = tk.Label(self.param_lfm3_0, text='calc_unc')
        self.param_lab3_0_6_0 = tk.Label(self.param_lfm3_0, text='nb')
        self.param_lab3_0_7_0 = tk.Label(self.param_lfm3_0, text='hide poles')
        self.param_cmb3_0_0_1 = ttk.Combobox(self.param_lfm3_0, state='readonly', values=['SSIdat'])
        self.param_etr3_0_1_1 = tk.Entry(self.param_lfm3_0)
        self.param_etr3_0_2_1 = tk.Entry(self.param_lfm3_0)
        self.param_etr3_0_3_1 = tk.Entry(self.param_lfm3_0)
        self.param_etr3_0_4_1 = tk.Entry(self.param_lfm3_0)
        self.param_chb3_0_5_1 = tk.Checkbutton(self.param_lfm3_0, variable=self.var_to_calc_unc)
        self.param_etr3_0_6_1 = tk.Entry(self.param_lfm3_0)
        self.param_chb3_0_7_1 = tk.Checkbutton(self.param_lfm3_0, variable=self.var_to_hide_poles)
        self.param_lab3_1_0_0 = tk.Label(self.param_lfm3_1, text='percentile')
        self.param_lab3_1_1_0 = tk.Label(self.param_lfm3_1, text='min cluster elements')
        self.param_etr3_1_0_1 = tk.Entry(self.param_lfm3_1)
        self.param_etr3_1_1_1 = tk.Entry(self.param_lfm3_1)

        self.param_btn4 = tk.Button(self.tab_param, text='Save', command=self.save_parameters, font=('Helvetica', '35'))
        self.param_tbl5 = ttk.Treeview(self.tab_param, columns=('Name'))

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
        self.param_lab1_0_1_0.grid(row=1, column=0)
        self.param_lab1_0_2_0.grid(row=2, column=0)
        self.param_lab1_0_3_0.grid(row=3, column=0)
        self.param_etr1_0_0_1.grid(row=0, column=1)
        self.param_etr1_0_1_1.grid(row=1, column=1)
        self.param_etr1_0_2_1.grid(row=2, column=1)
        self.param_etr1_0_3_1.grid(row=3, column=1)
        self.param_lfm1_0.pack()
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
        self.param_lab3_0_6_0.grid(row=6, column=0)
        self.param_lab3_0_7_0.grid(row=7, column=0)
        self.param_cmb3_0_0_1.grid(row=0, column=1)
        self.param_etr3_0_1_1.grid(row=1, column=1)
        self.param_etr3_0_2_1.grid(row=2, column=1)
        self.param_etr3_0_3_1.grid(row=3, column=1)
        self.param_etr3_0_4_1.grid(row=4, column=1)
        self.param_chb3_0_5_1.grid(row=5, column=1)
        self.param_etr3_0_6_1.grid(row=6, column=1)
        self.param_chb3_0_7_1.grid(row=7, column=1)
        self.param_lab3_1_0_0.grid(row=0, column=0)
        self.param_lab3_1_1_0.grid(row=1, column=0)
        self.param_etr3_1_0_1.grid(row=0, column=1)
        self.param_etr3_1_1_1.grid(row=1, column=1)
        self.param_chb3_2.pack()
        self.param_lfm3_0.pack()
        self.param_lfm3_1.pack()
        self.param_lfm3.grid(row=0, column=2, rowspan=2)

        self.param_btn4.grid(row=2, column=0, columnspan=3)
        self.param_tbl5.grid(row=3, column=0, columnspan=3)

        self.initialize_widgets_of_params()

        #=========================================================
        # Run Frame
        #=========================================================
        # create widgets
        self.run_lfm0 = ttk.Labelframe(self.tab_run, text='select parameter_set to run')
        self.run_frm1 = tk.Frame(self.tab_run)
        self.run_btn = tk.Button(self.run_frm1, text='Run', command=self.run, font=('Helvetica', '35'))

        # layout widgets
        self.run_btn.pack(anchor='center')
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
        self.param_etr0_0_1_1.insert(0, '0, 15')
        self.param_etr0_1_0_1.insert(0, '0')
        self.param_etr0_1_1_1.insert(0, '0.1')
        self.param_etr0_1_2_1.insert(0, '0')
        self.param_cmb0_1_3_1.set("highpass")

        self.param_etr1_0_0_1.insert(0, '0.01')
        self.param_etr1_0_1_1.insert(0, '')
        self.param_etr1_0_2_1.insert(0, 'find_min')
        self.param_etr1_0_3_1.insert(0, '0, 10')

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
        self.param_etr2_0_9_1.insert(0, '0')
        self.param_etr2_0_10_1.insert(0, '0')

        self.var_to_run_ssi.set(True)
        self.param_cmb3_0_0_1.set('SSIdat')
        self.param_etr3_0_1_1.insert(0, '30')
        self.param_etr3_0_2_1.insert(0, '0')
        self.param_etr3_0_3_1.insert(0, '30')
        self.param_etr3_0_4_1.insert(0, '1')
        self.var_to_calc_unc.set(True)
        self.param_etr3_0_6_1.insert(0, '100')
        self.var_to_hide_poles.set(True)
        self.param_etr3_1_0_1.insert(0, '80')
        self.param_etr3_1_1_1.insert(0, '1')

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

        # get parameters
        param_dict = {
                    'data_folder'          : self.data_etr0_0_1.get(),
                    'data_files'           : shlex.split(self.data_etr0_1_1.get()),
                    'usecols'              : [int(x) for x in self.data_etr0_2_1.get().split(',')],
                    'start_time'           : start_time,
                    'save_dir'             : self.data_etr0_4_1.get(),
                    'geo_path'             : self.geo_etr0_0_1.get(),

                    'freqlim'              : freqlim,
                    'fs'                   : float(self.param_etr0_0_0_1.get()),
                    'q'                    : int(self.param_etr0_1_0_1.get()),
                    'Wn'                   : tuple([float(x) for x in self.param_etr0_1_1_1.get().split(',')]),
                    'order'                : int(self.param_etr0_1_2_1.get()),
                    'btype'                : self.param_cmb0_1_3_1.get(),

                    'rtol'                 : float(self.param_etr1_0_0_1.get()),
                    'sel_freq'             : sel_freq,
                    'order_in'             : order_in,
                    'sel_freqlim'          : sel_freqlim,

                    'run_fdd'              : self.var_to_run_fdd.get(),
                    'fdd_method'           : self.param_cmb2_0_0_1.get(),
                    'nxseg'                : float(self.param_etr2_0_1_1.get()),
                    'method_SD'            : self.param_cmb2_0_2_1.get(),
                    'pov'                  : float(self.param_etr2_0_3_1.get()),
                    'DF'                   : float(self.param_etr2_0_4_1.get()),
                    'DF1'                  : float(self.param_etr2_0_5_1.get()),
                    'DF2'                  : float(self.param_etr2_0_6_1.get()),
                    'cm'                   : int(self.param_etr2_0_7_1.get()),
                    'MAClim'               : float(self.param_etr2_0_8_1.get()),
                    'sppk'                 : int(self.param_etr2_0_9_1.get()),
                    'npmax'                : int(self.param_etr2_0_10_1.get()),

                    'run_ssi'              : self.var_to_run_ssi.get(),
                    'ssi_method'           : self.param_cmb3_0_0_1.get(),
                    'br'                   : int(self.param_etr3_0_1_1.get()),
                    'ord_min'              : int(self.param_etr3_0_2_1.get()),
                    'ord_max'              : int(self.param_etr3_0_3_1.get()),
                    'ord_step'             : int(self.param_etr3_0_4_1.get()),
                    'calc_unc'             : self.var_to_calc_unc.get(),
                    'nb'                   : int(self.param_etr3_0_6_1.get()),
                    'hide_poles'           : self.var_to_hide_poles.get(),
                    'percentile'           : float(self.param_etr3_1_0_1.get()),
                    'min_cluster_elements' : int(self.param_etr3_1_1_1.get())
                    }

        return param_dict

    def save_parameters(self):
        for i in range(1, len(self.set_up_dict) + 2):
            name = f'set_up_{i}'
            if name not in self.set_up_dict.keys():
                break
        self.set_up_dict[name] = {}
        self.set_up_dict[name]['params'] = self.get_parameters()
        self.param_tbl5.insert(parent='', index='end', iid=None, values=name)

        # construct a widget to run analysis by current parameter set
        self.set_up_dict[name]['var_to_run'] = tk.BooleanVar()
        self.set_up_dict[name]['chb_to_run'] = tk.Checkbutton(
            self.run_lfm0,
            text=name,
            variable=self.set_up_dict[name]['var_to_run']
            )
        self.set_up_dict[name]['chb_to_run'].pack()
        self.set_up_dict[name]['var_to_run'].set(True)

    def run(self):
        for key in self.set_up_dict.keys():
            if self.set_up_dict[key]['var_to_run'].get():
                parameters = self.set_up_dict[key]['params']
                print(parameters)
                ss, fdd, efdd, fsdd, ssidat, ssicov = self._run_fdd_ssi(parameters)

                self._run_other_analysis(parameters, ss, efdd, ssidat)

    def _run_fdd_ssi(self, params):
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
        ssidat = SSIdat(name="SSIdat", br=p['br'], ordmax=p['ord_max'], calc_unc=p['calc_unc'])
        ssicov = SSIcov(name="SSIcov", br=p['br'], ordmax=p['ord_max'], calc_unc=p['calc_unc'])

        # Add algorithms to the single setup class
        ss.add_algorithms(fdd, efdd, fsdd, ssidat, ssicov)

        # Run selected method
        if p['run_fdd']:
            ss.run_by_name(p['fdd_method'])
        if p['run_ssi']:
            ss.run_by_name(p['ssi_method'])

        return ss, fdd, efdd, fsdd, ssidat, ssicov

    def _run_other_analysis(self, params, ss, efdd, ssidat):
        p = params
        efdd_save_folder = p['save_dir'] + '/EFDD'
        ssi_save_folder  = p['save_dir'] + '/SSI (auto)'
        os.makedirs(efdd_save_folder, exist_ok=True)
        os.makedirs(ssi_save_folder , exist_ok=True)

        # output result (figs and parameters) ===================================================================
        fig, ax = ssidat.plot_stab(freqlim=p['freqlim'], hide_poles=p['hide_poles'])
        output_ssidat_fig = ssi_save_folder + "/" + "SSIdat_stab_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".png"
        plt.savefig(output_ssidat_fig, dpi=300, bbox_inches='tight')
        fig, ax = ssidat.plot_freqvsdamp(freqlim=p['freqlim'], hide_poles=p['hide_poles'])
        output_ssidat_fig = ssi_save_folder + "/" + "SSIdat_freq-damp_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".png"
        plt.savefig(output_ssidat_fig, dpi=300, bbox_inches='tight')

        #====================================================================================================================
        # output
        #====================================================================================================================
        # stabilization csv file ======================================================
        df_stab_poles = self.stab_poles_to_df(p['hide_poles'], ssidat.result, ssidat)
        outnm_stab=("SSIdat_stab_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".csv")
        output_path_stab = ssi_save_folder + "/" + outnm_stab
        df_stab_poles.to_csv(output_path_stab, index=False)

        # frequency-damping csv file  ===============================================================
        df_freq_damp_poles = self.freq_damp_poles_to_df(p['hide_poles'], ssidat.result)
        outnm_FnXi=("SSIdat_freq-damp_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".csv")
        output_path_FnXi = ssi_save_folder + "/" + outnm_FnXi
        df_freq_damp_poles.to_csv(output_path_FnXi, index=False)

        #====================================================================================================================
        # run Hierarchical clustering
        #====================================================================================================================
        d_percent, Z, clusters = self.ssidat_single_linkage_method(ssidat.result, p['percentile'])

        fig = self.plot_Hclus(Z, d_percent)
        output_Hclus_fig = ssi_save_folder + "/" + "Hclus_" + os.path.splitext(os.path.basename(p['data_folder']))[0] + ".png"
        fig.savefig(output_Hclus_fig, dpi=300, bbox_inches='tight')
        plt.close(fig)

        zoom_max = d_percent * 1.2
        fig_zoom = self.plot_Hclus(Z, d_percent, zoom_ylim=(0, zoom_max))
        output_zoom_fig = ssi_save_folder + "/" + "Hclus_zoom_" + os.path.splitext(os.path.basename(p['data_folder']))[0] + ".png"
        fig_zoom.savefig(output_zoom_fig, dpi=300, bbox_inches='tight')
        plt.close(fig_zoom)

        # output Hierarchical clustering csv file
        _, _, df_freq_and_cluster_info, df_Hclus = self.hcluster_to_df(ssidat.result, p['percentile'], p['min_cluster_elements'])
        outnm_Hclus = ("Hclus_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".csv")
        output_path_Hclus = ssi_save_folder + "/" + outnm_Hclus
        df_Hclus.to_csv(output_path_Hclus, index=False)

        # median frequency
        med_freq_list = df_freq_and_cluster_info.iloc[2, :].tolist()[1:]
        print(med_freq_list)

        n_list = []
        poles_list = poles_list = self.extract_valid_poles(ssidat.result)
        for f_med in med_freq_list:
            # freqに最も近いpoleを選ぶ
            closest_pole = min(poles_list, key=lambda x: abs(x[2] - f_med))
            n_list.append(closest_pole[1])

        print("Model orders:", n_list)

        #====================================================================================================================
        # Select modes to extract
        #====================================================================================================================
        sel_freq = med_freq_list
        ss.mpe("SSIdat", sel_freq=sel_freq, rtol=p['rtol'], order_in=n_list)
        mode_nr = len(sel_freq)
        plt.close()

        # output mode shape =========================================================
        self.save_mode_shape(ss, p['geo_path'], mode_nr, ssidat.result, ssi_save_folder)

        # create dataframe
        df_cluster_phi_FR = self.cluster_phi_and_FR_to_df(ssidat.result, p['percentile'], p['geo_path'], efdd.result, med_freq_list, p['min_cluster_elements'])
        # output dataframe
        outnm_cluster_phi_FR = ("SSIdat_cluster_shape_FR_"+os.path.splitext(os.path.basename(p['data_folder']))[0]+".csv")
        output_path_cluster_phi_FR = ssi_save_folder + "/" + outnm_cluster_phi_FR
        df_cluster_phi_FR.to_csv(output_path_cluster_phi_FR, index=False, header=False)

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

    def extract_valid_poles(self, ssidat_result):
        Fn = ssidat_result.Fn_poles
        Phi = ssidat_result.Phi_poles
        Lab = ssidat_result.Lab

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

    def compute_min_distances(self, poles_list):
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

    def compute_distance_matrix(self, poles_list):
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

    def plot_Hclus(self, Z, threshold, zoom_ylim=None):
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["lines.linewidth"] = 0.3

        fig = plt.figure(figsize=(12, 5))
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
                xerr = None

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

        return df_freq_damp_poles

    def ssidat_single_linkage_method(self, ssidat_result, percentile):

        poles_list = self.extract_valid_poles(ssidat_result)

        min_distances = self.compute_min_distances(poles_list)
        d_vec = np.array([entry['distance'] for entry in min_distances])
        d_percent = np.percentile(d_vec, q=percentile)

        distance_matrix = self.compute_distance_matrix(poles_list)
        condensed_dist = squareform(distance_matrix)
        Z = linkage(condensed_dist, method='single')
        clusters = fcluster(Z, t=d_percent, criterion='distance')

        return d_percent, Z, clusters

    def hcluster_to_df(self, ssidat_result, percentile, min_cluster_elements):
        _, _, clusters = self.ssidat_single_linkage_method(ssidat_result, percentile)
        poles_list = self.extract_valid_poles(ssidat_result)

        Hclus = []
        for cluster_id, pole_info in zip(clusters, poles_list):
            indx, n, freq, phi = pole_info
            Hclus.append([cluster_id, n, indx, freq])
        df_Hclus_info = pd.DataFrame(Hclus, columns=['ClusterID', 'ModelIndex', 'PoleIndex', 'Frequency'])

        def compute_CV(df):
            return df.std(ddof=0) / df.mean()
        freq_info = df_Hclus_info.groupby('ClusterID')['Frequency'].agg(CV=compute_CV, MedFrequency='median', AvgFrequency='mean', MinFrequency='min', MaxFrequency='max').reset_index()
        df_freq_info = freq_info.rename(columns={'ClusterID': 'ClusterNo.'})

        count_cluster_elements = df_Hclus_info['ClusterID'].value_counts().to_dict()
        df_Hclus_info['Elements'] = df_freq_info['ClusterNo.'].map(count_cluster_elements)
        df_Hclus_info['ClusterNo.'] = df_freq_info['ClusterNo.']
        df_Hclus_info[' '] = ''

        df_freq_info['Elements'] = df_Hclus_info['Elements']
        df_freq_info = df_freq_info.reindex(columns=['ClusterNo.', 'Elements', 'CV', 'MedFrequency', 'AvgFrequency', 'MinFrequency', 'MaxFrequency', ' '])
        df_freq_info = df_freq_info[df_freq_info['Elements'] > min_cluster_elements]
        df_freq_info = df_freq_info.sort_values('MedFrequency')
        df_freq_info_T = df_freq_info.T
        df_freq_info_T.reset_index(inplace=True)
        df_freq_info_T.columns = df_freq_info_T.iloc[0]
        df_freq_info_T = df_freq_info_T.drop(df_freq_info_T.index[0])
        df_freq_info_T.reset_index(drop=True, inplace=True)

        df_freqs_of_clusters = pd.DataFrame()
        for number in df_freq_info['ClusterNo.']:
            df_cluster = df_Hclus_info[df_Hclus_info['ClusterID'] == number]
            df_cluster.reset_index(drop=True, inplace=True)
            df_freqs_of_clusters = pd.concat([df_freqs_of_clusters, df_cluster['Frequency']], axis=1)
        df_freqs_of_clusters.columns = df_freq_info['ClusterNo.']

        df_freq_and_cluster_info = pd.concat([df_freq_info_T, df_freqs_of_clusters], axis=0)
        df_freq_and_cluster_info.reset_index(drop=True, inplace=True)
        df_Hclus = pd.concat([df_Hclus_info, df_freq_and_cluster_info], axis=1)

        return df_Hclus_info, df_freq_info, df_freq_and_cluster_info, df_Hclus

    def save_mode_shape(self, ss, geo, mode_nr, ssidat_result, save_folder):
        ss.def_geo2_by_file(geo)

        #if mpl == 'h':
        _, _ = ss.plot_geo2_mpl(scaleF=2)
        for i in range(1, mode_nr + 1):
            fig_shape, _ = ss.plot_mode_geo2_mpl(algo_res=ssidat_result, mode_nr=i, view="3D", scaleF=3)
            freq = ssidat_result.Fn[i - 1]
            Phi_filename = f"{freq:.2f}Hz.png"
            fig_shape.savefig(os.path.join(save_folder, Phi_filename), dpi=300)

        # if pyvista == 'h':
        #     _ = ss.plot_geo2(scaleF=2)
        #     for i in range(1, mode_nr + 1):
        #         _ = ss.plot_mode_geo2(algo_res=ssidat_result, mode_nr=i, scaleF=3)
        #         _ = ss.anim_mode_geo2(algo_res=ssidat_result, mode_nr=i, scaleF=3)

    def MPC_MPD_MCF(self, ssidat_result):
        Phi = ssidat_result.Phi
        mpc_list = [gen.MPC(Phi[:, i]) for i in range(Phi.shape[1])]
        mpd_list = [gen.MPD(Phi[:, i]) for i in range(Phi.shape[1])]
        one_minus_mpd_list = [1 - x for x in mpd_list]
        mcf_list = [gen.MCF(Phi[:, i])[0] for i in range(Phi.shape[1])]

        return mpc_list, mpd_list, one_minus_mpd_list, mcf_list

    def phi_to_df(self, ssidat_result, geo):
        Fn = ssidat_result.Fn
        Phi = ssidat_result.Phi

        mpc_list, mpd_list, one_minus_mpd_list, mcf_list = self.MPC_MPD_MCF(ssidat_result)

        df_geo = pd.read_excel(geo, sheet_name='sensors names', header=None)
        sensor_names = df_geo.iloc[1, 1:].dropna().tolist()

        header = [''] + [f"mode{i+1}" for i in range(len(Fn))]
        row_freq = ['freq(Hz)'] + list(np.round(Fn, 4))
        row_mpc = ['MPC'] + list(np.round(mpc_list, 6))
        row_mpd = ['MPD'] + list(np.round(mpd_list, 6))
        row_one_minus_mpd = ['1 - MPD'] + list(np.round(one_minus_mpd_list, 6))
        row_mcf = ['MCF'] + list(np.round(mcf_list, 6))

        data_phi = []
        for i, name in enumerate(sensor_names):
            row = [name] + list(np.round(Phi.real[i], 6))
            data_phi.append(row)

        df_phi = pd.DataFrame([header, row_freq, row_mpc, row_mpd, row_one_minus_mpd, row_mcf] + data_phi)

        return df_phi

    def cluster_and_phi_to_df(self, ssidat_result, percentile, geo, min_cluster_elements):
        _, df_freq_info, _, _ = self.hcluster_to_df(ssidat_result, percentile, min_cluster_elements)
        df_phi = self.phi_to_df(ssidat_result, geo)
        df_cluster_info = df_freq_info[['ClusterNo.', 'Elements', 'CV']]
        df_cluster_info = df_cluster_info.reset_index(drop=True)
        df_cluster_info = df_cluster_info.copy()
        df_cluster_info['1-CV'] = 1 - df_cluster_info['CV']
        df_cluster_info = df_cluster_info.T.reset_index()
        df_cluster_info = df_cluster_info.set_axis(df_phi.columns, axis='columns')
        idx1, idx2 = 2, 6; df_cluster_phi = pd.concat([df_phi[:idx1], df_cluster_info, df_phi[idx1:idx2], pd.DataFrame(index=[' ']), df_phi[idx2:]])

        return df_cluster_phi

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

        sel_freq_idx = self.idx_of_the_nearest(freq, selected_freq)

        sel_S_val = S_val[sel_freq_idx]
        max_S_val = S_val.max()

        FR = sel_S_val/max_S_val
        print(selected_freq, '\n', 'FR : ', FR)

        return FR

    def cluster_phi_and_FR_to_df(self, ssidat_result, percentile, geo, efdd_result, selected_freq, min_cluster_elements):
        FR = self.compute_FR(efdd_result, selected_freq)
        row_FR = ['FR'] + list(np.round(FR, 6))

        df_cluster_phi = self.cluster_and_phi_to_df(ssidat_result, percentile, geo, min_cluster_elements)
        df_FR = pd.DataFrame([row_FR])

        idx = 10
        df_cluster_phi_FR = pd.concat([df_cluster_phi[:idx], df_FR, df_cluster_phi[idx:]])

        return df_cluster_phi_FR

    def params_to_txt(self, output_path_param, fs, br, ordmax, ordmin, step, calc_unc, nb, hide_poles, q, Wn, order, btype, freqlim, percentile):
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

# OmaApp を実行
if __name__ == "__main__":
    root = tk.Tk()
    app = OmaApp(root)
    root.mainloop()