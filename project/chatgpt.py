import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLabel, QWidget,
    QDockWidget, QVBoxLayout, QSizePolicy, QSplitter
)
from PySide6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dockable Dashboard (QSplitter for Cameras)")
        self.resize(1400, 900)

        # # === Cameras (2x2 grid using nested splitters inside one dock) ===
        # cam1 = self.make_camera_widget("Camera 1")
        # cam2 = self.make_camera_widget("Camera 2")
        # cam3 = self.make_camera_widget("Camera 3")
        # cam4 = self.make_camera_widget("Camera 4")

        # # top row (cam1 | cam2)
        # top_split = QSplitter(Qt.Horizontal)
        # top_split.addWidget(cam1)
        # top_split.addWidget(cam2)

        # # bottom row (cam3 | cam4)
        # bottom_split = QSplitter(Qt.Horizontal)
        # bottom_split.addWidget(cam3)
        # bottom_split.addWidget(cam4)

        # # vertical split (top row over bottom row)
        # cam_split = QSplitter(Qt.Vertical)
        # cam_split.addWidget(top_split)
        # cam_split.addWidget(bottom_split)

        # # put into dock
        # self.dock_cameras = self.add_dock("Cameras", cam_split, Qt.LeftDockWidgetArea)

        # === System Metrics (right dock) ===
        # metrics_panel = QWidget()
        # metrics_layout = QVBoxLayout()
        # for text in ["CPU: ...", "RAM: ...", "GPU: ...", "Uptime: ...", "Latency: ...", "Avg FPS: ..."]:
        #     lbl = QLabel(text)
        #     metrics_layout.addWidget(lbl)
        # metrics_panel.setLayout(metrics_layout)
        # metrics_panel.setFixedSize(200, 150)

        # put into dock
        # self.dock_metrics = self.add_dock("System Metrics", metrics_panel, Qt.RightDockWidgetArea)

        # === Logs (right dock, under metrics) ===
        # log_widget = QTextEdit()
        # log_widget.setReadOnly(True)
        # log_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # put into dock
        # self.dock_logs = self.add_dock("Logs", log_widget, Qt.RightDockWidgetArea)
        # self.splitDockWidget(self.dock_metrics, self.dock_logs, Qt.Vertical)

        # === Live Charts (bottom docks side by side) ===
        chart1 = self.make_chart_label("Live Chart 1")
        chart2 = self.make_chart_label("Live Chart 2")

        self.dock_chart1 = self.add_dock("Live Chart 1", chart1, Qt.BottomDockWidgetArea)
        self.dock_chart2 = self.add_dock("Live Chart 2", chart2, Qt.BottomDockWidgetArea)
        self.splitDockWidget(self.dock_chart1, self.dock_chart2, Qt.Horizontal)

        # === Initial proportions ===
        self.resizeDocks([self.dock_cameras, self.dock_chart1], [700, 200], Qt.Vertical)

    # def make_camera_widget(self, text):
    #     lbl = QLabel(text)
    #     lbl.setAlignment(Qt.AlignCenter)
    #     lbl.setStyleSheet("background: #4477aa; color: white; border: 1px solid black;")
    #     lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    #     return lbl

    # def make_chart_label(self, text):
    #     lbl = QLabel(text)
    #     lbl.setAlignment(Qt.AlignCenter)
    #     lbl.setStyleSheet("background: #55aa55; color: white; border: 1px solid black;")
    #     lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    #     lbl.setMinimumHeight(100)
    #     lbl.setMaximumHeight(300)
    #     return lbl

    # def add_dock(self, title, widget, area):
    #     dock = QDockWidget(title, self)
    #     dock.setWidget(widget)
    #     dock.setAllowedAreas(
    #         Qt.LeftDockWidgetArea |
    #         Qt.RightDockWidgetArea |
    #         Qt.TopDockWidgetArea |
    #         Qt.BottomDockWidgetArea
    #     )
    #     dock.setFeatures(
    #         QDockWidget.DockWidgetMovable |
    #         QDockWidget.DockWidgetFloatable |
    #         QDockWidget.DockWidgetClosable
    #     )
    #     self.addDockWidget(area, dock)
    #     return dock

    # def resizeEvent(self, event):
    #     """Force charts to stay smaller relative to cameras after resize/maximize."""
    #     super().resizeEvent(event)
    #     self.resizeDocks([self.dock_cameras, self.dock_chart1], [700, 200], Qt.Vertical)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
