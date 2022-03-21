# from PyQt5 import QtWidgets, QtCore
# from pyqtgraph import PlotWidget, plot
# from pyqtgraph.Qt import QtGui, QtCore
# import pyqtgraph as pg
# import sys  # We need sys so that we can pass argv to QApplication
# import os
#
# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self, *args, **kwargs):
#         super(MainWindow, self).__init__(*args, **kwargs)
#
#         self.graphWidget = pg.PlotWidget()
#         self.setCentralWidget(self.graphWidget)
#
#         self.x = []
#         self.y = []
#         self.n_iter = 0
#
#         pen = pg.mkPen(color=(0, 0, 0))
#         self.data_line = self.graphWidget.plot(self.x, self.y, pen=pen)
#
#         #Add Background colour to white
#         self.graphWidget.setBackground('k')
#         # Add Title
#         self.graphWidget.setTitle("JWST Mirror Alignment", color="w", size="16pt")
#         # Add Axis Labels
#         styles = {"color": "w", "font-size": "10px"}
#         self.graphWidget.setLabel("left", "chi^2", **styles)
#         self.graphWidget.setLabel("bottom", "Iteration", **styles)
#         #Add legend
#         self.graphWidget.addLegend()
#         #Add grid
#         self.graphWidget.showGrid(x=True, y=True)
#         #Set Range
#         # self.graphWidget.setXRange(0, 10, padding=0)
#         self.graphWidget.setYRange(0., 2., padding=0)
#
#         self.timer = QtCore.QTimer()
#         self.timer.setInterval(50)
#         self.timer.timeout.connect(self.update_plot_data)
#         # self.timer.start()
#
#     def update_plot_data(self, x):
#         self.x.append(x)
#         self.y.append(self.n_iter)
#         self.data_line.setData(self.x, self.y)  # Update the data.
#
#         self.n_iter += 1
#
# def main():
#     app = QtWidgets.QApplication(sys.argv)
#     main = MainWindow()
#     main.show()
#
#     import time
#     import numpy as np
#     for i in np.linspace(0, 1.2, 20):
#         main.update_plot_data(i**2)
#         main.graphWidget.processEvents()
#         time.sleep(0.5)
#         print("hht")
#
#     sys.exit(app.exec_())
#
#
#
# if __name__ == '__main__':
#     main()

import numpy as np
import pyqtgraph as pg
import sys
from pyqtgraph.Qt import QtCore, QtGui
import threading
import time

class Plot2D(pg.GraphicsWindow):
    def __init__(self):
        pg.GraphicsWindow.__init__(self, title="MDS Argon")
        self.traces = dict()
        self.resize(1000, 600)
        pg.setConfigOptions(antialias=True)
        self.waveform1 = self.addPlot(title='Exponential', row=1, col=1, connect="finite")
        self.waveform2 = self.addPlot(title='WAVEFORM2', row=2, col=1)

    def set_plotdata(self, name, x, y):
        if name in self.traces:
            self.traces[name].setData(x, y)
        else:
            if name == "910D":
                self.traces[name] = self.waveform1.plot(x, y, pen='w', width=3)
            elif name == "MPU":
                self.traces[name] = self.waveform2.plot(x, y, pen='w', width=3)

    @QtCore.pyqtSlot(str, tuple)
    def updateData(self, name, ptm):
        x, y = ptm
        self.set_plotdata(name, x, y)

class Helper(QtCore.QObject):
    changedSignal = QtCore.pyqtSignal(str, tuple)

def create_data1(helper, name):
    t = np.arange(-2.0, 2.0, 0.1)
    i = 0.
    s = np.full_like(t, fill_value=np.nan)
    ind = 0
    while True:
        np.put(s, ind=ind, v=np.exp(i) / np.math.factorial(int(i)), mode="wrap")
        np.put(s, ind=range(ind+1, ind+5), v=np.inf, mode="wrap")
        time.sleep(.2)
        helper.changedSignal.emit(name, (t, s))
        i = i + 0.1
        ind += 1

def create_data2(helper, name):
    t = np.arange(-2.0, 2.0, 0.1)
    i = 0.0
    while True:
        s = np.cos(2 * 2 * 3.1416 * t) / (2 * 3.1416 * t - i)
        time.sleep(.2)
        helper.changedSignal.emit(name, (t, s))
        i = i + 0.1

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    helper = Helper()
    plot = Plot2D()
    helper.changedSignal.connect(plot.updateData, QtCore.Qt.QueuedConnection)
    threading.Thread(target=create_data1, args=(helper, "910D"), daemon=True).start()
    threading.Thread(target=create_data2, args=(helper, "MPU"), daemon=True).start()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()