from PyQt5 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, plot
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        self.x = []
        self.y = []
        self.n_iter = 0

        pen = pg.mkPen(color=(0, 0, 0))
        self.data_line = self.graphWidget.plot(self.x, self.y, pen=pen)

        #Add Background colour to white
        self.graphWidget.setBackground('k')
        # Add Title
        self.graphWidget.setTitle("JWST Mirror Alignment", color="w", size="16pt")
        # Add Axis Labels
        styles = {"color": "w", "font-size": "10px"}
        self.graphWidget.setLabel("left", "chi^2", **styles)
        self.graphWidget.setLabel("bottom", "Iteration", **styles)
        #Add legend
        self.graphWidget.addLegend()
        #Add grid
        self.graphWidget.showGrid(x=True, y=True)
        #Set Range
        # self.graphWidget.setXRange(0, 10, padding=0)
        self.graphWidget.setYRange(0., 2., padding=0)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_plot_data)
        # self.timer.start()

    def update_plot_data(self, x):
        self.x.append(x)
        self.y.append(self.n_iter)
        self.data_line.setData(self.x, self.y)  # Update the data.

        self.n_iter += 1

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()

    import time
    import numpy as np
    for i in np.linspace(0, 1.2, 20):
        main.update_plot_data(i**2)
        main.graphWidget.processEvents()
        time.sleep(0.5)
        print("hht")

    sys.exit(app.exec_())



if __name__ == '__main__':
    main()