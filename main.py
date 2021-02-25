import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from tkinter import *


class MainApplication(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, *kwargs)
        self.parent = parent

        self.initialise_tree()

        self.initialise_open_file()

        self.close_plot_frame = tk.LabelFrame(root, text="Close Price")
        self.close_plot_frame.place(height=350, width=500, rely=0.3, relx=0)

        self.plotter = None

    def initialise_open_file(self):
        # Frame for visualizing data
        self.file_frame = tk.LabelFrame(root, text="Open File")
        self.file_frame.place(height=100, width=400, rely=0.2, relx=0)

        # Buttons
        button1 = tk.Button(self.file_frame, text="Browse A File", command=lambda: self.File_dialog())
        button1.place(rely=0.65, relx=0.50)

        button2 = tk.Button(self.file_frame, text="Load File", command=lambda: self.Load_data())
        button2.place(rely=0.65, relx=0.30)

        # The file/file path text
        self.label_file = ttk.Label(self.file_frame, text="No File Selected")
        self.label_file.place(rely=0, relx=0)


    def initialise_tree(self):

         # Frame for TreeView
        self.frame1 = tk.LabelFrame(root, text="Excel Data")
        self.frame1.place(height=200, width=1000, rely=0, relx=0)

        ## Treeview Widget
        self.tv1 = ttk.Treeview(self.frame1)
        self.tv1.place(relheight=1, relwidth=1) # set the height and width of the widget to 100% of its container (frame1).

        treescrolly = tk.Scrollbar(self.frame1, orient="vertical", command=self.tv1.yview) # command means update the yaxis view of the widget
        treescrollx = tk.Scrollbar(self.frame1, orient="horizontal", command=self.tv1.xview) # command means update the xaxis view of the widget

        self.tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) # assign the scrollbars to the Treeview Widget

        treescrollx.pack(side="bottom", fill="x") # make the scrollbar fill the x axis of the Treeview widget
        treescrolly.pack(side="right", fill="y") # make the scrollbar fill the y axis of the Treeview widget



    def File_dialog(self):
        filename = filedialog.askopenfilename(initialdir="/E:/OneDrive/Documents/University/Year 3/Project/Notebooks/Data",
                                            title="Select A File",
                                            filetype=(("csv files", "*.csv"),("All Files", "*.*")))
        self.label_file["text"] = filename


    def Load_data(self):
        
        if self.plotter:

            self.plotter.destroy()

        file_path = self.label_file["text"]

        filename = r"{}".format(file_path)
        
        df = pd.read_csv(filename)

        self.clear_data()

        self.tv1["column"] = list(df.columns)
        self.tv1["show"] = "headings"
        for column in self.tv1["columns"]:
            self.tv1.heading(column, text=column) # let the column heading = column name

        df_rows = df.to_numpy().tolist() # turns the dataframe into a list of lists
        for row in df_rows:
            self.tv1.insert("", "end", values=row)

        df['Date'] = pd.to_datetime(df.Date, format='%d.%m.%Y %H:%M:%S.%f')
        df['Date'] = pd.DataFrame(df.Date).applymap(lambda x: x.date())
        df = df.set_index(df.Date).drop(columns=['Date'])

        self.plot_close_price(prices=df.Close)

    def plot_close_price(self, prices):
        #Plot close prices
        plt.rc('xtick',labelsize=6)
        plt.rc('ytick',labelsize=6)

        fig = Figure(figsize=(5,10))

        ax = fig.add_subplot(111)

        ax.plot(prices)

        ax.set_title("Close Price", fontsize=6)
        ax.set_ylabel("Price", fontsize=6)
        ax.set_xlabel("Date", fontsize=6)

        canvas = FigureCanvasTkAgg(fig,master=self.close_plot_frame)
        self.plotter = canvas.get_tk_widget()
        self.plotter.pack(fill='both')
        canvas.draw()

        ax.clear()

    def clear_data(self):
        self.tv1.delete(*self.tv1.get_children())

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Financial ML Project Demo")
    root.state("zoomed")
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.grid_propagate(False)
    root.mainloop()