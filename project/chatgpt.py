import psutil, GPUtil, time
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QTimer
import pyqtgraph as pg
import numpy as np

class MetricsChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget(background="#222")
        layout.addWidget(self.plot_widget)

        # --- Plot setup ---
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Usage", units="%")
        self.plot_widget.setYRange(0, 100)
        self.plot_widget.addLegend()

        # Lines for CPU, RAM, GPU
        self.cpu_line = self.plot_widget.plot(pen=pg.mkPen("y", width=2), name="CPU")
        self.ram_line = self.plot_widget.plot(pen=pg.mkPen("r", width=2), name="RAM")
        self.gpu_line = self.plot_widget.plot(pen=pg.mkPen("c", width=2), name="GPU")

        # Buffers (180 data points ≈ 3 min at 1 Hz)
        self.max_points = 180
        self.cpu_data, self.ram_data, self.gpu_data = [], [], []
        self.time_data = []

        # Prime psutil CPU measurement
        psutil.cpu_percent(interval=None)

        # Timer for updates
        self.start_time = time.time()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1000)  # every 1 s

    def update_data(self):
        now = time.time() - self.start_time
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent

        try:
            gpus = GPUtil.getGPUs()
            gpu = gpus[0].load * 100 if gpus else 0
        except Exception:
            gpu = 0

        # Append new samples
        self.time_data.append(now)
        self.cpu_data.append(cpu)
        self.ram_data.append(ram)
        self.gpu_data.append(gpu)

        # Keep only last 180 samples
        if len(self.time_data) > self.max_points:
            self.time_data = self.time_data[-self.max_points:]
            self.cpu_data = self.cpu_data[-self.max_points:]
            self.ram_data = self.ram_data[-self.max_points:]
            self.gpu_data = self.gpu_data[-self.max_points:]

        # Update plot lines
        self.cpu_line.setData(self.time_data, self.cpu_data)
        self.ram_line.setData(self.time_data, self.ram_data)
        self.gpu_line.setData(self.time_data, self.gpu_data)

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
import sys

class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.chart = ChartWidget()
        layout.addWidget(self.chart)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Dashboard()
    w.show()
    sys.exit(app.exec())
