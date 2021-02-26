import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from tkinter import *


class NewWindow(Toplevel): 
      
    def __init__(self, master = None, data = None): 
          
        super().__init__(master = master) 

        self.geometry("800x700")

        self.plotter=None

        #Frame for performing EDA
        self.eda_frame = tk.LabelFrame(self, text="EDA")
        self.eda_frame.place(height=700, width=800, rely=0, relx=0)

        self.prices = data

        self.plot_eda()

    def plot_eda(self):

        if self.plotter:

            self.plotter.destroy()

        fig, ax = plt.subplots(ncols=1, nrows=4, sharex=True, figsize=(12,10))

        #Decompose data to spot trends
        data_decomp = seasonal_decompose(self.prices, model="additive", extrapolate_trend='freq', period=365)

        plt.rcParams.update({'figure.figsize': (15,15)})

        fig.suptitle("Seasonal Decomposition")
        data_decomp.observed.plot(ax=ax[0], legend=False)
        ax[0].set_ylabel('Observed')
        data_decomp.trend.plot(ax=ax[1], legend=False)
        ax[1].set_ylabel('Trend')
        data_decomp.seasonal.plot(ax=ax[2], legend=False)
        ax[2].set_ylabel('Seasonal')
        data_decomp.resid.plot(ax=ax[3], legend=False)
        ax[3].set_ylabel('Residual')

        canvas = FigureCanvasTkAgg(fig,master=self.eda_frame)
        self.plotter = canvas.get_tk_widget()
        self.plotter.pack(fill='both')
        canvas.draw()