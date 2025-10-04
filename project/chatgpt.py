from PySide6.QtWidgets import (
    QApplication, QMainWindow, QToolBar,
    QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFormLayout
)
from PySide6.QtGui import QIcon, QAction
import sys


class AddCameraDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Camera Source")
        self.setMinimumWidth(300)

        # Form layout
        layout = QFormLayout()

        self.name_edit = QLineEdit()
        self.url_edit = QLineEdit()

        layout.addRow("Camera Name:", self.name_edit)
        layout.addRow("RTSP URL:", self.url_edit)

        # Button
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.accept)  # closes with Accepted

        vbox = QVBoxLayout()
        vbox.addLayout(layout)
        vbox.addWidget(self.add_button)

        self.setLayout(vbox)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Toolbar Example")

        # Toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Add Camera Action
        add_camera_action = QAction(QIcon(), "Add Camera", self)
        add_camera_action.triggered.connect(self.show_add_camera_form)
        toolbar.addAction(add_camera_action)

    def show_add_camera_form(self):
        dialog = AddCameraDialog(self)
        if dialog.exec():  # shows dialog modally
            camera_name = dialog.name_edit.text()
            camera_url = dialog.url_edit.text()
            print("Added Camera:", camera_name, camera_url)
            # You can now save or process this data


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
