from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QVBoxLayout, QToolBar, QFileDialog, QMessageBox
from PyQt5.QtCore import pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib import rcParams, cbook, ticker
import matplotlib.pyplot as plt
import numpy as np
import os, sys

class GpsoFigureWidget(QWidget):
    '''
    A widget that displays the current algorithm's progress in a matplotlib pyplot
    '''
    def __init__(self, parent=None, **kwargs):
        super(GpsoFigureWidget, self).__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure(figsize=(15, 6))
        self.ax = plt.gca()
        #plt.style.use(kwargs.get('style', 'dark_background'))

        self.ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        self.ax.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())
        self.ax.grid(b=True, which='major', color=(0.75, 0.5, 0.75, 0.75), linewidth=0.25)
        self.ax.grid(b=True, which='minor', color=(0.75, 0.5, 0.75, 0.75), linewidth=0.25)

        self.ax.set_xlabel('iterations', labelpad=7, fontsize=11)
        self.ax.set_ylabel('distance', labelpad=17, fontsize=11, rotation=270)

        for tick in self.ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(7) 
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(7)
        
        self.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

        self.ax.tick_params(
                            axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            #bottom=False,      # ticks along the bottom edge are off
                            #top=False,         # ticks along the top edge are off
                            direction='in',
                            #labelrotation=
                            #labelbottom=True,
                            )

        self.ax.tick_params(
                    axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    #bottom=False,      # ticks along the bottom edge are off
                    #top=False,         # ticks along the top edge are off
                    direction='in',
                    labelrotation=315,
                    #labelbottom=True,
                    )

        self.series_names = ['worst', 'std', 'avg', 'best']
        colors = ['#FF00EC', '#FF9B00', '#FF0000', '#FFFFFF']
        self.series_range = range(4)
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvasQTAgg(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.updatePlot)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)

        self.figure_data = {}

        for label, color in zip(self.series_names, colors):
            self.figure_data[label] = [self.ax.plot([], [], label=label, color=color)[0]]

        self.figure_data['std'].append(self.ax.plot([], [], label='-std', color=colors[1])[0])

        self.ax.legend()
        self.ax.legend(loc=2, prop={'size': 6})

    def updatePlot(self, id_, x, new_ys):
        for i, y in enumerate(new_ys):
            self.figure_data[id_][i].set_xdata(np.append(self.figure_data[id_][i].get_xdata(), x))
            self.figure_data[id_][i].set_ydata(np.append(self.figure_data[id_][i].get_ydata(), y))

    @pyqtSlot(str)
    def toggleVisible(self, id_):
        for plot in self.figure_data[id_]:
            plot.set_visible(not plot.__dict__['_visible'])

    @pyqtSlot(object)
    def updateFigure(self, new_data):
        ''' plot some random stuff '''
        # random data
        #print(new_data)
        for i in self.series_range:
            self.updatePlot(self.series_names[i], new_data[0], new_data[i + 1])

        self.ax.relim()
        self.ax.autoscale_view()
        plt.tight_layout()

        # refresh canvas
        self.canvas.draw()

class HillClimberFigureWidget(QWidget):
    '''
    A widget that displays the current algorithm's progress in a matplotlib pyplot
    '''
    def __init__(self, parent=None, **kwargs):
        super(HillClimberFigureWidget, self).__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure(figsize=(15, 6))
        self.ax = plt.gca()
        #plt.style.use(kwargs.get('style', 'dark_background'))

        self.ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        self.ax.get_yaxis().set_minor_locator(ticker.AutoMinorLocator())
        self.ax.grid(b=True, which='major', color=(0.75, 0.5, 0.75, 0.75), linewidth=0.25)
        self.ax.grid(b=True, which='minor', color=(0.75, 0.5, 0.75, 0.75), linewidth=0.25)

        self.ax.set_xlabel('iterations', labelpad=7, fontsize=11)
        self.ax.set_ylabel('distance', labelpad=17, fontsize=11, rotation=270)

        for tick in self.ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(7) 
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(7)
        
        self.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

        self.ax.tick_params(
                            axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            #bottom=False,      # ticks along the bottom edge are off
                            #top=False,         # ticks along the top edge are off
                            direction='in',
                            #labelrotation=
                            #labelbottom=True,
                            )

        self.ax.tick_params(
                            axis='y',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            #bottom=False,      # ticks along the bottom edge are off
                            #top=False,         # ticks along the top edge are off
                            direction='in',
                            labelrotation=315,
                            #labelbottom=True,
                            )

        self.series_names = ['dst']
        colors            = ['#FFFFFF']
        self.series_range = range(1)
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvasQTAgg(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.updatePlot)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)

        self.figure_data = {}
        
        for label, color in zip(self.series_names, colors):
            self.figure_data[label] = [self.ax.plot([], [], label=label, color=color)[0]]

        self.ax.legend()
        self.ax.legend(loc=2, prop={'size': 6})

    def updatePlot(self, id_, x, new_ys):
        for i, y in enumerate(new_ys):
            self.figure_data[id_][i].set_xdata(np.append(self.figure_data[id_][i].get_xdata(), x))
            self.figure_data[id_][i].set_ydata(np.append(self.figure_data[id_][i].get_ydata(), y))

    @pyqtSlot(str)
    def toggleVisible(self, id_):
        for plot in self.figure_data[id_]:
            plot.set_visible(not plot.__dict__['_visible'])

    @pyqtSlot(object)
    def updateFigure(self, new_data):
        ''' plot some random stuff '''
        # random data
        #print(new_data)
        for i in self.series_range:
            self.updatePlot(self.series_names[i], new_data[0], new_data[i + 1])

        self.ax.relim()
        self.ax.autoscale_view()
        plt.tight_layout()

        # refresh canvas
        self.canvas.draw()


class NavigationToolbar(NavigationToolbar2QT):
    '''
        This class is used to overwrite the 'save_figure' method of the NavigationToolbar2QT
        class and enable the non-native dialog options.
    '''
    def __init__(self, canvas, parent, coordinates=True):
        NavigationToolbar2QT.__init__(self, canvas, parent, coordinates=True)

    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        default_filetype = self.canvas.get_default_filetype()

        startpath = os.path.expanduser(
            rcParams['savefig.directory'])
        start = os.path.join(startpath, self.canvas.get_default_filename())
        filters = []
        selectedFilter = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filter_ = '%s (%s)' % (name, exts_list)
            if default_filetype in exts:
                selectedFilter = filter_
            filters.append(filter_)
        filters = ';;'.join(filters)

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        fname, filter_ = QFileDialog.getSaveFileName(self.canvas.parent(),
                                                    "Choose a filename to save to",
                                                     start, filters, selectedFilter, options=options)
        if fname:
            # Save dir for next time, unless empty str (i.e., use cwd).
            if startpath != "":
                rcParams['savefig.directory'] = (
                    os.path.dirname(fname))
            try:
                self.canvas.figure.savefig(fname)
            except Exception as e:
                QMessageBox.critical(
                    self, "Error saving file", str(e),
                    QMessageBox.Ok, QMessageBox.NoButton)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gpso = GpsoFigureWidget()
    hc = HillClimberFigureWidget()
    gpso.show()
    hc.show()
    sys.exit(app.exec_())