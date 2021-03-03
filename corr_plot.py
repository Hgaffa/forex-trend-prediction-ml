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


class CorrelationPlot(Toplevel): 
      
    def __init__(self, master = None, data = None): 
          
        super().__init__(master = master) 

        self.geometry("800x700")

        self.plotter=None

        #Frame for correlation plot
        self.corr_plot = tk.LabelFrame(self, text="Correlation with future returns")
        self.corr_plot.place(height=700, width=800, rely=0, relx=0)

        self.df = data

        self.plot_correlation()

    def plot_correlation(self):

        if self.plotter:

            self.plotter.destroy()

        fig = Figure(figsize=(20, 17))

        #canvas must be put before plotting to allow for interactiveness
        canvas = FigureCanvasTkAgg(fig,master=self.corr_plot)
        toolbar = NavigationToolbar2Tk(canvas, self.corr_plot)
        toolbar.update()
        self.plotter = canvas.get_tk_widget()
        self.plotter.pack(fill='both')

        ax = fig.subplots()

        corr_sent = self.df.copy().drop(columns=['Labels']).corrwith(self.df.Returns.shift(-1))

        corr_sent.sort_values(ascending=False).plot.barh(title='Feature Correlation with Next Day Returns', ax=ax)

        ax.tick_params(axis = 'both', which = 'major', labelsize = 5)
