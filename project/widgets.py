
from PySide6.QtWidgets import (QApplication, QMainWindow, QToolBar, QMenuBar, QDialog, QLineEdit, QFormLayout, QVBoxLayout, QHBoxLayout, QPushButton,
                                QMessageBox, QStatusBar, QCheckBox, QGroupBox, QLabel, QFileDialog, QComboBox, QWidget, QSizePolicy, QSplitter,
                                QDockWidget, QTextEdit)
from PySide6.QtCore import Qt, QSize, QRegularExpression, QSettings, QByteArray
from PySide6.QtGui import QIcon, QAction, QIntValidator, QRegularExpressionValidator

import sys
import json
from pathlib import Path

from classes import Camera

asset_path = str(Path(__file__).parent / "assets")

class MainWindow(QMainWindow):
    def __init__(self, app: QApplication):
        super().__init__()
        self.setWindowTitle("TraffIQ-PH Control Panel")
        self.resize(1500, 900)
        self.app = app

        # Qsettings
        self.settings = QSettings("Lanaia Robotics", "TraffIQ-PH")

        # Hint docks and load layout
        self.load_layout()

        # Menu bar
        menu_bar = MenuBar(self)
        self.setMenuBar(menu_bar)

        # Tool Bar
        main_tool_bar = MainToolBar(self)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, main_tool_bar)
        
        # Status Bar
        status_bar = StatusBar(self)
        self.setStatusBar(status_bar)

        # Cameras
        self.cameras: list[Camera] = []

        # Load all settings and configurations
        self.display_settings = {}
        self.experimental_settings = {}
        self.load_all_settings()

        # TODO: Propagate changes in the settings to the current layout
        # TODO: Adjust save and load to include the geometries and states
    
    def load_layout(self):
        """ Load layout. """
        camera_widget = CameraWidget()
        metrics_widget = MetricsWidget()
        logs_widget = LogsWidget()
        chart_left_widget = ChartWidget("Live Chart A")
        chart_right_widget = ChartWidget("Live Chart B")

        # Dock the components
        self.dock_cameras = self._add_dock("Cameras", camera_widget, Qt.LeftDockWidgetArea)
        self.dock_metrics = self._add_dock("System Metrics", metrics_widget, Qt.RightDockWidgetArea)
        self.dock_logs = self._add_dock("Logs", logs_widget, Qt.RightDockWidgetArea)
        self.splitDockWidget(self.dock_metrics, self.dock_logs, Qt.Vertical)
        self.dock_chart_left = self._add_dock("Live Chart A", chart_left_widget, Qt.BottomDockWidgetArea)
        self.dock_chart_right = self._add_dock("Live Chart B", chart_right_widget, Qt.BottomDockWidgetArea)
        self.splitDockWidget(self.dock_chart_left, self.dock_chart_right, Qt.Horizontal)
        self.resizeDocks([self.dock_cameras, self.dock_chart_left], [700, 200], Qt.Vertical)
    
    def set_layout(self, reset=False):
        """Restore the window layout to its standard configuration."""

        try:
            if reset:
                # Load standard layout if reset
                standard_path = f"{asset_path}/standard_layout.json"
                with open(standard_path, "r") as f:
                    data = json.load(f)
            else:
                # Set the layout using current layout settings
                
                data = self.layout_settings
            
            if not data:
                print("No layout settings can be found.")
                return

            # Restore top-level window geometry and dock state
            self.restoreGeometry(QByteArray.fromHex(data["geometry"].encode()))
            self.restoreState(QByteArray.fromHex(data["state"].encode()))

            # Restore internal camera splitter layout
            if "camera" in data and hasattr(self, "dock_cameras"):
                self.dock_cameras.widget().restore_camera_layout(data["camera"])

            self.statusBar().showMessage("Layout reset to standard configuration.", 3000)

        except FileNotFoundError:
            self.statusBar().showMessage("Standard layout file not found.", 5000)

        except Exception as e:
            print(f"Error loading standard layout: {e}")
            self.statusBar().showMessage("Error resetting layout.", 5000)

    def load_all_settings(self):
        """ Load cameras and UI settings from system store. """

        # Load cameras
        raw_cams = self.settings.value("cameras")
        if raw_cams:
            try:
                data = json.loads(raw_cams)
                self.cameras.clear()
                for entry in data:
                    cam = Camera(**entry)
                    self.cameras.append(cam)
                print(f"Loaded {len(self.cameras)} cameras from the system.")
            except Exception as e:
                print("Error loading cameras:", e)
        
        # Load display settings
        raw_display = self.settings.value("display_settings")
        if raw_display:
            try:
                self.display_settings = json.loads(raw_display)
            except Exception:
                self.display_settings = {}
        else:
            # defaults
            self.display_settings = {
                "bounding_boxes": {
                    "obstructions": True,
                    "two_wheeled": True,
                    "light": True,
                    "heavy": True,
                },
                "osd": {
                    "date": True,
                    "name": True,
                    "location": False,
                    "traffic_phase": True,
                    "estimated_congestion": True,
                },
                "logs": {
                    "log_level": "Info",
                }
            }
        
        # ---- Experimental Settings (future-proof)
        raw_experimental = self.settings.value("experimental_settings")
        if raw_experimental:
            try:
                self.experimental_settings = json.loads(raw_experimental)
            except Exception:
                self.experimental_settings = {}
        else:
            self.experimental_settings = {}
        
        # Load layout settings
        raw_layout = self.settings.value("layout")
        if raw_layout:
            try:
                self.layout_settings = json.loads(raw_layout)
                self.set_layout()
                self.statusBar().showMessage("Successfully loaded settings from last session.")
            except Exception:
                self.statusBar().showMessage("Unsuccessfully loaded settings from last session.")
                self.layout_settings = {}
        else:
            self.layout_settings = {}

    def save_all_settings(self):
        """ Save all settings to system. """

        self._save_current_layout()
        
        # Save cameras
        data = [cam.__dict__.copy() for cam in self.cameras]
        for d in data:
            d.pop("full_link", None)
        self.settings.setValue("cameras", json.dumps(data))

        # Save display settings
        self.settings.setValue("display_settings", json.dumps(self.display_settings))
        
        # Save experimental settings
        self.settings.setValue("experimental_settings", json.dumps(self.experimental_settings))

        # Save geometry and state
        self.settings.setValue("layout", json.dumps(self.layout_settings))

    def _load_configs(self):
        """Import settings from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(self, "Load Configs", "", "JSON Files (*.json)")
        if not path:
            return
        
        with open(path, "r") as f:
            payload = json.load(f)

        # restore cameras
        self.cameras.clear()
        for entry in payload.get("cameras", []):
            self.cameras.append(Camera(**entry))

        # restore other settings
        self.display_settings = payload.get("display_settings", {})
        self.experimental_settings = payload.get("experimental_settings", {})
        
        # restore layout settings
        self.layout_settings = payload.get("layout", {})

        # set layout
        self.set_layout()

        # also update QSettings so it’s consistent
        self.save_all_settings()

        self.statusBar().showMessage(f"Successfully loaded settings from {path}.")

    def _save_as_configs(self):
        """Export settings to a JSON file."""
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Configs", "", "JSON Files (*.json)")

        self._save_current_layout()
        
        payload = {
            "cameras": [cam.__dict__.copy() for cam in self.cameras],
            "display_settings": self.display_settings,
            "experimental_settings": self.experimental_settings,
            "layout": self.layout_settings,
        }
        for d in payload["cameras"]:
            d.pop("full_link", None)

        with open(path, "w") as f:
            json.dump(payload, f, indent=4)
        
        self.statusBar().showMessage(f"Successfully saved settings to {path}.")

        # TODO: BUG: Saved settings.json doesn't restore the layout settings.

    def _save_current_layout(self):
        self._default_geometry = self.saveGeometry()
        self._default_state = self.saveState()
        self._default_camera_state = self.dock_cameras.widget().save_camera_layout()

        self.layout_settings = {
            "geometry": self._default_geometry.toHex().data().decode(), 
            "state": self._default_state.toHex().data().decode(),
            "camera": self._default_camera_state
        }

    def _quit(self):
        self.app.quit()

    def _add_dock(self, title: str, widget, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(title.replace(" ", "_").lower())
        dock.setWidget(widget)
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea |
            Qt.RightDockWidgetArea |
            Qt.TopDockWidgetArea |
            Qt.BottomDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        
        self.addDockWidget(area, dock)
        return dock
    # Overriden

    def resizeEvent(self, event):
        """Force charts to stay smaller relative to cameras after resize/maximize."""
        super().resizeEvent(event)
        self.resizeDocks([self.dock_cameras, self.dock_chart_left], [700, 200], Qt.Vertical)


class MenuBar(QMenuBar):
    def __init__(self, parent: MainWindow=None):
        super().__init__(parent)

        # Add menus
        self.addFileMenu()
        self.addViewMenu()
        self.addHelpMenu()

    def addFileMenu(self):
        self.file_menu = self.addMenu("File")
        load_action = self.file_menu.addAction("Load settings")
        save_action = self.file_menu.addAction("Save settings")
        save_as_action = self.file_menu.addAction("Save settings as")
        quit_action = self.file_menu.addAction("Quit")

        load_action.setToolTip("Load settings and configurations.")
        load_action.setStatusTip("Load settings and configurations.")
        save_action.setToolTip("Save settings and configurations.")
        save_action.setStatusTip("Save settings and configurations.")
        save_as_action.setToolTip("Save current settings to a file.")
        save_as_action.setStatusTip("Save current settings to a file.")
        quit_action.setToolTip("Quit program.")
        quit_action.setStatusTip("Quit program.")

        load_action.triggered.connect(self.parent()._load_configs)
        save_action.triggered.connect(self.parent().save_all_settings)
        save_as_action.triggered.connect(self.parent()._save_as_configs)
        quit_action.triggered.connect(self.parent()._quit)
    
    def addViewMenu(self):
        self.view_menu = self.addMenu("View")

        for dock in [self.parent().dock_cameras,
                     self.parent().dock_metrics,
                     self.parent().dock_logs,
                     self.parent().dock_chart_left,
                     self.parent().dock_chart_right]:
            action = dock.toggleViewAction()
            self.view_menu.addAction(action)
        
        self.view_menu.addSeparator()
        reset_action = self.view_menu.addAction("Reset layout")
        reset_action.triggered.connect(lambda: self.parent().set_layout(reset=True))

    def addHelpMenu(self):
        self.help_menu = self.addMenu("Help")
        
class MainToolBar(QToolBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.setOrientation(Qt.Orientation.Vertical)
        self.setIconSize(QSize(30, 30))
        self.setWindowTitle("Main Toolbar")
        self.setObjectName("main_toolbar")

        # Add actions
        self.actionAddSource()
        self.actionChangeDisplaySettings()
        self.actionEditRegion()
        self.actionChangeExperimentalSettings()

    def actionAddSource(self):
        """ Add RTSP source. """
        action = QAction(QIcon(f"{asset_path}/add.png"), "Add", self)
        action.setStatusTip("Add RTSP source.")
        action.setToolTip("Add RTSP source.")
        action.triggered.connect(self._add_source)

        self.addAction(action)
    
    def actionChangeDisplaySettings(self):
        """ Change OSD settings. """
        action = QAction(QIcon(f"{asset_path}/display_settings.png"), "Display Settings", self)
        action.setStatusTip("Change display settings.")
        action.setToolTip("Change display settings.")
        action.triggered.connect(self._change_display_settings)

        self.addAction(action)

    def actionEditRegion(self):
        """ Edit region of interest (ROI). """
        action = QAction(QIcon(f"{asset_path}/edit_region.png"), "Edit Region", self)
        action.setStatusTip("Edit region of interest.")
        action.setToolTip("Edit region of interest.")
        action.triggered.connect(self._edit_region)

        self.addAction(action)

    def actionChangeExperimentalSettings(self):
        """ Configure experimental and detection settings. """
        action = QAction(QIcon(f"{asset_path}/experimental.png"), "Experimental Settings", self)
        action.setStatusTip("Change experimental settings.")
        action.setToolTip("Change experimental settings.")
        action.triggered.connect(self._change_experimental_settings)

        self.addAction(action)
    
    # Helper functions

    def _add_source(self):
        dialog = AddSourceDialog(self)
        if dialog.exec():
            data = dialog.get_data()
            camera = Camera(**data)
            self.parent().cameras.append(camera) # add current camera to cameras
            self.parent().save_cameras() # save cameras persistently
            self.parent().statusBar().showMessage("Successfully added camera.", 3000)
            print(f"Camera added: {camera}")
    
    def _change_display_settings(self):
        dialog = ChangeDisplaySettingsDialog(self.window())
        dialog.set_settings(self.parent().display_settings) # load defaults

        if dialog.exec():
            settings = dialog.get_settings()
            self.parent().display_settings = settings
            self.parent().save_all_settings()
            self.parent().statusBar().showMessage("Updated display settings", 3000)
            print("User settings:", settings)
        
    def _edit_region(self):
        print("Edit region of interest")

    def _change_experimental_settings(self):
        """
        Features:
            1. Notify officials for possible obstructions.
            2. Checkbox of shown bounding boxes. 
            3. Use other models?
            4. Cap FPS

        
        """
        print("Change experimental settings!")

class StatusBar(QStatusBar):
    def __init__(self, parent):
        super().__init__(parent)

class AddSourceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Camera Source")
        self.setMinimumWidth(300)

        # Form layout
        layout = QFormLayout()

        self.name_edit = QLineEdit()
        self.address_edit = QLineEdit()
        self.username_edit = QLineEdit()
        self.port_edit = QLineEdit()
        self.channel_edit = QLineEdit()
        self.password_edit = QLineEdit()
        self.location_edit = QLineEdit()
        self.line_edit_fields: list[QLineEdit] = [self.name_edit, self.address_edit, self.username_edit, self.port_edit,
                                 self.channel_edit, self.password_edit, self.location_edit]

        self.port_edit.setValidator(QIntValidator(1, 65535, self))
        self.channel_edit.setValidator(QIntValidator(1, 512, self))
        self.password_edit.setEchoMode(QLineEdit.Password)
        ip_regex = QRegularExpression(
            r"^(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
            r"(\.(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3}$"
        )
        self.address_edit.setPlaceholderText("e.g. 192.168.0.64")
        self.address_edit.setValidator(QRegularExpressionValidator(ip_regex))
        self.channel_edit.setText("101")
        self.port_edit.setText("554")


        layout.addRow("Camera Name:", self.name_edit)
        layout.addRow("Location:", self.location_edit)
        layout.addRow("IP Address:", self.address_edit)
        layout.addRow("Username:", self.username_edit)
        layout.addRow("Password:", self.password_edit)
        layout.addRow("RTSP Port:", self.port_edit)
        layout.addRow("Channel:", self.channel_edit)
        

        # Button
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.validate_and_accept)

        vbox = QVBoxLayout()
        vbox.addLayout(layout)
        vbox.addWidget(self.add_button)

        self.setLayout(vbox)
    
    def validate_and_accept(self):
        """ Validates the form and checks if all fields are populated. """
        if not all([field.text().strip() for field in self.line_edit_fields]):
            QMessageBox.warning(self, "Missing Data", "Please fill all fields!")
            return

        self.accept()
    
    def get_data(self):
        """Package dialog data into a dict (or return Camera directly)."""
        return {
            "name": self.name_edit.text().strip(),
            "ip_address": self.address_edit.text().strip(),
            "username": self.username_edit.text().strip(),
            "password": self.password_edit.text().strip(),
            "port": int(self.port_edit.text().strip()),
            "channel": int(self.channel_edit.text().strip()),
            "location": self.location_edit.text().strip(),
        }

class ChangeDisplaySettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Display Settings")
        self.setMinimumWidth(400)

        main_layout = QVBoxLayout()

        # --- Group 1: Bounding Boxes ---
        bbox_group = QGroupBox("Bounding Box Display")
        bbox_layout = QVBoxLayout()
        bbox_layout.addWidget(QLabel("Choose which bounding boxes to show:"))

        self.cb_obstructions = QCheckBox("Obstructions")
        self.cb_two_wheeled = QCheckBox("Two-wheeled vehicles")
        self.cb_light = QCheckBox("Light vehicles")
        self.cb_heavy = QCheckBox("Heavy vehicles")

        bbox_layout.addWidget(self.cb_obstructions)
        bbox_layout.addWidget(self.cb_two_wheeled)
        bbox_layout.addWidget(self.cb_light)
        bbox_layout.addWidget(self.cb_heavy)

        bbox_group.setLayout(bbox_layout)
        main_layout.addWidget(bbox_group)

        # --- Group 2: OSD (On-Screen Display) ---
        osd_group = QGroupBox("On-Screen Display (OSD)")
        osd_layout = QVBoxLayout()
        osd_layout.addWidget(QLabel("Choose which information to overlay:"))

        self.cb_date = QCheckBox("Date")
        self.cb_name = QCheckBox("Name")
        self.cb_location = QCheckBox("Location")
        self.cb_phase = QCheckBox("Traffic Phase")
        self.cb_congestion = QCheckBox("Estimated Congestion")

        osd_layout.addWidget(self.cb_date)
        osd_layout.addWidget(self.cb_name)
        osd_layout.addWidget(self.cb_location)
        osd_layout.addWidget(self.cb_phase)
        osd_layout.addWidget(self.cb_congestion)

        osd_group.setLayout(osd_layout)
        main_layout.addWidget(osd_group)

        # --- Group 3: Logs
        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout()
        log_layout.addWidget(QLabel("Configure logs settings:"))
        self.cb_log_level = QComboBox()
        self.cb_log_level.addItems(["Debug", "Info" ,"Errors"])

        log_layout.addWidget(self.cb_log_level)

        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def get_settings(self):
        """Return the chosen settings as a dict"""
        return {
            "bounding_boxes": {
                "obstructions": self.cb_obstructions.isChecked(),
                "two_wheeled": self.cb_two_wheeled.isChecked(),
                "light": self.cb_light.isChecked(),
                "heavy": self.cb_heavy.isChecked(),
            },
            "osd": {
                "date": self.cb_date.isChecked(),
                "name": self.cb_name.isChecked(),
                "location": self.cb_location.isChecked(),
                "traffic_phase": self.cb_phase.isChecked(),
                "estimated_congestion": self.cb_congestion.isChecked(),
            },
            "logs": {
                "log_level": self.cb_log_level.currentText()
            }
        }
    
    def set_settings(self, settings: dict):
        bb = settings.get("bounding_boxes", {})
        osd = settings.get("osd", {})
        logs = settings.get("logs", {})

        self.cb_obstructions.setChecked(bb.get("obstructions", False))
        self.cb_two_wheeled.setChecked(bb.get("two_wheeled", False))
        self.cb_light.setChecked(bb.get("light", False))
        self.cb_heavy.setChecked(bb.get("heavy", False))

        self.cb_date.setChecked(osd.get("date", False))
        self.cb_name.setChecked(osd.get("name", False))
        self.cb_location.setChecked(osd.get("location", False))
        self.cb_phase.setChecked(osd.get("traffic_phase", False))
        self.cb_congestion.setChecked(osd.get("estimated_congestion", False))
        log_level = logs.get("log_level", "Info")   # default to "Info"
        idx = self.cb_log_level.findText(log_level)
        if idx >= 0:
            self.cb_log_level.setCurrentIndex(idx)

class EditRegionDialog(QDialog):
    pass

class ChangeExperimentalSettingsDialog(QDialog):
    pass

class CameraWidget(QSplitter):
    def __init__(self, parent=None):
        super().__init__(parent)

        cam1 = self.make_camera_widget("Camera 1")
        cam2 = self.make_camera_widget("Camera 2")
        cam3 = self.make_camera_widget("Camera 3")
        cam4 = self.make_camera_widget("Camera 4")

        # top row (cam1 | cam2)
        self.top_split = QSplitter(Qt.Horizontal)
        self.top_split.addWidget(cam1)
        self.top_split.addWidget(cam2)

        # bottom row (cam3 | cam4)
        self.bottom_split = QSplitter(Qt.Horizontal)
        self.bottom_split.addWidget(cam3)
        self.bottom_split.addWidget(cam4)

        # vertical split (top row over bottom row)
        self.setOrientation(Qt.Vertical)
        self.addWidget(self.top_split)
        self.addWidget(self.bottom_split)
    
    def make_camera_widget(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("background: #4477aa; color: white; border: 1px solid black;")
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return lbl

    def save_camera_layout(self):
        """Return all splitter states as dict."""
        return {
            "main": bytes(self.saveState()).hex(),
            "top": bytes(self.top_split.saveState()).hex(),
            "bottom": bytes(self.bottom_split.saveState()).hex(),
        }

    def restore_camera_layout(self, data: dict):
        """Restore all splitter geometries."""
        if not data:
            return
        if "main" in data:
            self.restoreState(bytes.fromhex(data["main"]))
        if "top" in data:
            self.top_split.restoreState(bytes.fromhex(data["top"]))
        if "bottom" in data:
            self.bottom_split.restoreState(bytes.fromhex(data["bottom"]))

class MetricsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        metrics_layout = QVBoxLayout()
        for text in ["CPU: ...", "RAM: ...", "GPU: ...", "Uptime: ...", "Latency: ...", "Avg FPS: ..."]:
            lbl = QLabel(text)
            metrics_layout.addWidget(lbl)
        self.setLayout(metrics_layout)
        self.setFixedSize(200, 150)

class LogsWidget(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

class ChartWidget(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(parent)

        self.make_chart_label(text)
    
    def make_chart_label(self, text):
        self.setText(text)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #55aa55; color: white; border: 1px solid black;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(100)
        self.setMaximumHeight(300)
