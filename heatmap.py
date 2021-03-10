import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose

import seaborn as sns

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from tkinter import *


class Heatmap(Toplevel): 
      
    def __init__(self, master = None, data = None): 
          
        super().__init__(master = master) 

        self.geometry("800x700")

        self.plotter=None

        #Frame for heatmap
        self.heatmap_frame = tk.LabelFrame(self, text="Heatmap")
        self.heatmap_frame.place(height=700, width=800, rely=0, relx=0)

        self.df = data

        self.plot_heatmap()

    def plot_heatmap(self):

        if self.plotter:

            self.plotter.destroy()

        sns.set(font_scale = 0.5)

        fig = Figure(figsize=(20, 17))

        canvas = FigureCanvasTkAgg(fig,master=self.heatmap_frame)
        toolbar = NavigationToolbar2Tk(canvas, self.heatmap_frame)
        toolbar.update()
        self.plotter = canvas.get_tk_widget()
        self.plotter.pack(fill='both')

        ax = fig.subplots()
        ax.set_title("Input Varible Correlation Heatmap")

        correlation_data = self.df.corr()

        plot = sns.heatmap(correlation_data, ax=ax)

        plot.set_yticklabels(plot.get_yticklabels(), rotation = 0)

        plot.set_xticklabels(plot.get_xticklabels(), rotation = 90)

        plt.rcParams.update({'figure.figsize': (15,15)})