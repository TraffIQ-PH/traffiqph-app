
from PySide6.QtWidgets import (QApplication, QMainWindow, QToolBar, QMenuBar, QDialog, QLineEdit, QFormLayout, QVBoxLayout, QHBoxLayout, QPushButton,
                                QMessageBox, QStatusBar, QCheckBox, QGroupBox, QLabel, QFileDialog, QComboBox, QWidget, QSizePolicy, QSplitter,
                                QDockWidget, QTextEdit, QProgressBar, QListWidget, QListWidgetItem)
from PySide6.QtCore import Qt, QSize, QRegularExpression, QSettings, QByteArray, QTimer, QDateTime
from PySide6.QtGui import QIcon, QAction, QIntValidator, QRegularExpressionValidator
import pyqtgraph as pg
import json
from pathlib import Path
import psutil
import GPUtil
import time
from threading import Thread, Event

from inference import Inferencer
from classes import Camera, LatencyTracker

asset_path = str(Path(__file__).parent / "assets")

class MainWindow(QMainWindow):
    def __init__(self, app: QApplication):
        super().__init__()
        self.setWindowTitle("TraffIQ-PH Control Panel")
        self.resize(1500, 900)
        self.app = app

        # Qsettings
        self.settings = QSettings("Lanaia Robotics", "TraffIQ-PH")

        self.latency_tracker = LatencyTracker()

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
        self.logs = []
        self.load_all_settings()

        self.start_yolo()

    def start_yolo(self):
    # Stop existing YOLO thread
        if hasattr(self, "yolo_thread") and self.yolo_thread.is_alive():
            self.yolo_stop_flag.set()
            self.yolo_thread.join(timeout=2)
            print("[INFO] Previous YOLO thread stopped.")
        
        for lbl in self.cam_labels:
            lbl.clear()
            lbl.setText("Missing camera source")
            lbl.setAlignment(Qt.AlignCenter)

        # Create a new stop flag for the new thread
        self.yolo_stop_flag = Event()

        # Start YOLO again
        self.inferencer = Inferencer(self.cam_labels, self.cameras, self.yolo_stop_flag, self.latency_tracker)
        self.yolo_thread = Thread(
            target=self.inferencer.run,
            daemon=True
        )
        self.yolo_thread.start()

        self._log_message("YOLO restarted with updated camera list.", 3000)


    def load_layout(self):
        """ Load layout. """
        self.camera_widget = CameraWidget()
        self.metrics_widget = MetricsWidget()
        self.logs_widget = LogsWidget()
        self.chart_left_widget = HeatmapChartWidget("Heatmap Chart")
        self.chart_right_widget = MetricsChartWidget()

        self.cam_labels = [
            self.camera_widget.top_split.widget(0),
            self.camera_widget.top_split.widget(1),
            self.camera_widget.bottom_split.widget(0),
            self.camera_widget.bottom_split.widget(1),
        ]


        # Dock the components
        self.dock_cameras = self._add_dock("Cameras", self.camera_widget, Qt.LeftDockWidgetArea)
        self.dock_metrics = self._add_dock("System Metrics", self.metrics_widget, Qt.RightDockWidgetArea)
        self.dock_logs = self._add_dock("Logs", self.logs_widget, Qt.RightDockWidgetArea)
        self.splitDockWidget(self.dock_metrics, self.dock_logs, Qt.Vertical)
        self.dock_chart_left = self._add_dock("Live Chart A", self.chart_left_widget, Qt.BottomDockWidgetArea)
        self.dock_chart_right = self._add_dock("System Metrics Chart", self.chart_right_widget, Qt.BottomDockWidgetArea)
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

            self._log_message("Layout reset to standard configuration.", 3000)

        except FileNotFoundError:
            self._log_message("Standard layout file not found.", 5000)

        except Exception as e:
            print(f"Error loading standard layout: {e}")
            self._log_message("Error resetting layout.", 5000)

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
            self.experimental_settings = {
                "notify_officials": True,
                "use_videos": False,
            }
        
        # Load layout settings
        raw_layout = self.settings.value("layout")
        if raw_layout:
            try:
                self.layout_settings = json.loads(raw_layout)
                self.set_layout()
            except Exception:
                self.layout_settings = {}
        else:
            self.layout_settings = {}

    def save_all_settings(self):
        """ Save all settings to system. """

        self._save_current_layout()
        
        # Save cameras
        self.save_cameras()

        # Save display settings
        self.settings.setValue("display_settings", json.dumps(self.display_settings))
        
        # Save experimental settings
        self.settings.setValue("experimental_settings", json.dumps(self.experimental_settings))

        # Save geometry and state
        self.settings.setValue("layout", json.dumps(self.layout_settings))

        self._log_message("All settings have been saved in your computer.", 3000)

    def save_cameras(self):
        data = [cam.__dict__.copy() for cam in self.cameras]
        for d in data:
            d.pop("full_link", None)
        self.settings.setValue("cameras", json.dumps(data))

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
        self.start_yolo()

        # restore other settings
        self.display_settings = payload.get("display_settings", {})
        self.experimental_settings = payload.get("experimental_settings", {})
        
        # restore layout settings
        self.layout_settings = payload.get("layout", {})

        # set layout
        self.set_layout()

        # also update QSettings so it’s consistent
        self.save_all_settings()

        self._log_message(f"Successfully loaded settings from {path}.", 2000)

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
        
        self._log_message(f"Successfully saved settings to {path}.")


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
    
    def _log_message(self, message: str, duration: int):
        self.statusBar().showMessage(message, duration)
        self.logs = [message] + self.logs
        self.dock_logs.widget().setText("\n\n".join(self.logs))
    
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
        self.actionRemoveSource()
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
    
    def actionRemoveSource(self):
        """ Remove RTSP source. """
        action = QAction(QIcon(f"{asset_path}/remove.png"), "Remove", self)
        action.setStatusTip("Remove RTSP source.")
        action.setToolTip("Remove RTSP source.")
        action.triggered.connect(self._remove_source)

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
        use_video = self.parent().experimental_settings.get("use_videos", False)
        if use_video:
            dialog = AddVideoSourceDialog(self)
            dialog.exec()
        else:
            dialog = AddSourceDialog(self)
            if dialog.exec():
                data = dialog.get_data()
                camera = Camera(**data)
                self.parent().cameras.append(camera) # add current camera to cameras
                self.parent().save_cameras() # save cameras persistently
                self.parent().start_yolo()
                self.parent()._log_message("Successfully added camera.", 3000)
                print(f"Camera added: {camera}")
    
    def _remove_source(self):
        dialog = RemoveSourceDialog(self)
        if dialog.exec():
            indexes = dialog.get_selected_indexes()
            if not indexes:
                QMessageBox.information(self, "No Selection", "No cameras were selected for removal.")
                return

            # remove from end to start (so indices remain valid)
            for idx in sorted(indexes, reverse=True):
                removed_cam = self.parent().cameras.pop(idx)
                print(f"[INFO] Removed camera: {removed_cam.name} ({removed_cam.ip_address})")

            # save updated list
            self.parent().save_cameras()

            # restart YOLO backend
            self.parent().start_yolo()

            self.parent()._log_message("Camera(s) removed and YOLO restarted.", 3000)


    def _change_display_settings(self):
        dialog = ChangeDisplaySettingsDialog(self.window())
        dialog.set_settings(self.parent().display_settings) # load defaults

        if dialog.exec():
            settings = dialog.get_settings()
            self.parent().display_settings = settings
            self.parent().save_all_settings()
            self.parent()._log_message("Updated display settings", 3000)
            print("User settings:", settings)
        
    def _edit_region(self):
        print("Edit region of interest")

    def _change_experimental_settings(self):
        """
        Features:
            1. Notify officials for possible obstructions.
            2. Checkbox of shown bounding boxes. 
            3. Use other models?
            5. Use videos for demonstration.
        
        """
        dialog = ChangeExperimentalSettingsDialog(self.window())
        dialog.set_settings(self.parent().experimental_settings) # load defaults

        if dialog.exec():
            settings = dialog.get_settings()
            self.parent().experimental_settings = settings
            self.parent().save_all_settings()
            self.parent()._log_message("Updated experimental settings", 3000)
            print("User settings:", settings)

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
            "rtsp_port": int(self.port_edit.text().strip()),
            "channel": int(self.channel_edit.text().strip()),
            "location": self.location_edit.text().strip(),
        }
    
class RemoveSourceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Remove Camera Source(s)")
        self.setMinimumWidth(400)
        self.parent_window = parent.parent()  # reference to MainWindow

        layout = QVBoxLayout(self)

        # --- List of cameras ---
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.list_widget.setAlternatingRowColors(True)

        # populate from parent's cameras
        for i, cam in enumerate(self.parent_window.cameras):
            item_text = f"{i+1}. {cam.name} — {cam.ip_address} ({cam.location})"
            item = QListWidgetItem(item_text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        self.remove_btn = QPushButton("Remove Selected")
        self.cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

        # --- Connections ---
        self.remove_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def get_selected_indexes(self):
        """Return indexes of selected (checked) cameras."""
        indexes = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                indexes.append(i)
        return indexes
    
class AddVideoSourceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Video Source")
        self.setMinimumWidth(300)

        layout = QVBoxLayout()
        layout.setSpacing(20)
        self.label = QLabel()
        self.label.setText("The use videos settings is set to True, please select the video."
                           "\nOtherwise, change the settings to False to link RTSP streams.")
        self.btn = QPushButton("Select Video")
        self.btn.clicked.connect(self.load_video)
        
        layout.addWidget(self.label)
        layout.addWidget(self.btn)
        

        self.setLayout(layout)

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")

        if file_path:
            print("File selected:", file_path)

        # TODO: continue loading video path

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

        self.cb_name = QCheckBox("Name")
        self.cb_location = QCheckBox("Location")
        self.cb_phase = QCheckBox("Traffic Phase")
        self.cb_congestion = QCheckBox("Estimated Congestion")

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
        self.ok_button = QPushButton("Save")
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
    """
        1. Notify officials through email.
        2. Use videos for demonstration.

    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Experimental Settings")
        self.setMinimumWidth(400)

        main_layout = QVBoxLayout()

        # Group
        settings_group = QGroupBox("Experimental Settings")
        settings_layout = QVBoxLayout()
        settings_layout.addWidget(QLabel("Configure experimental settings:"))

        self.cb_notify = QCheckBox("Notify officials for possible obstructions.")
        self.cb_use_videos = QCheckBox("Use videos instead for demonstration purposes.")

        settings_layout.addWidget(self.cb_notify)
        settings_layout.addWidget(self.cb_use_videos)

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Save")
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
            "notify_officials": self.cb_notify.isChecked(),
            "use_videos": self.cb_use_videos.isChecked()
        }
    
    def set_settings(self, settings: dict):
        # back here
        self.cb_notify.setChecked(settings.get("notify_officials", False))
        self.cb_use_videos.setChecked(settings.get("use_videos", False))

class CameraWidget(QSplitter):
    def __init__(self, parent=None):
        super().__init__(parent)

        cam1 = self.make_camera_widget("Missing camera source")
        cam2 = self.make_camera_widget("Missing camera source")
        cam3 = self.make_camera_widget("Missing camera source")
        cam4 = self.make_camera_widget("Missing camera source")

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
        lbl.setStyleSheet("background: #772953; color: white; border: 1px solid black;")
        lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        lbl.setScaledContents(False)
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("System Metrics")
        self.setFixedSize(250, 250)

        layout = QFormLayout()
        layout.setVerticalSpacing(4)
        layout.setHorizontalSpacing(10)

        self.date_label = QLabel()
        self.date_label.setAlignment(Qt.AlignCenter)
        self.date_label.setStyleSheet("font-weight: bold; font-size: 13px;")

        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        

        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        self.cpu_bar.setFormat("%p%")          # show percentage
        self.cpu_bar.setAlignment(Qt.AlignCenter)
        self.cpu_bar.setTextVisible(True)

        self.ram_bar = QProgressBar()
        self.ram_bar.setRange(0, 100)
        self.ram_bar.setFormat("%p%")          # show percentage
        self.ram_bar.setAlignment(Qt.AlignCenter)
        self.ram_bar.setTextVisible(True)

        self.gpu_bar = QProgressBar()
        self.gpu_bar.setRange(0, 100)
        self.gpu_bar.setFormat("%p%")          # show percentage
        self.gpu_bar.setAlignment(Qt.AlignCenter)
        self.gpu_bar.setTextVisible(True)

        self.uptime_label = QLabel("-- s")
        self.fps_label = QLabel("--")
        self.latency_total = QLabel("-- ms")
        self.latency_camera = QLabel("-- ms")
        self.latency_inference = QLabel("-- ms")
        self.latency_display = QLabel("-- ms")

        layout.addRow(self.date_label)
        layout.addRow(self.time_label)
        layout.addRow(QLabel("CPU:"), self.cpu_bar)
        layout.addRow(QLabel("RAM:"), self.ram_bar)
        layout.addRow(QLabel("GPU:"), self.gpu_bar)
        layout.addRow(QLabel("Uptime:"), self.uptime_label)
        layout.addRow(QLabel("FPS:"), self.fps_label)
        layout.addRow(QLabel("Total Delay:"), self.latency_total)
        layout.addRow(QLabel("Cam Delay:"), self.latency_camera)
        layout.addRow(QLabel("Inf Delay:"), self.latency_inference)
        layout.addRow(QLabel("Disp Delay:"), self.latency_display)

        self.setLayout(layout)

        psutil.cpu_percent(interval=None)

        self.start_time = time.time()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)  # update every second

    def update_metrics(self):
        cpu = psutil.cpu_percent(interval=None)
        self.cpu_bar.setValue(int(cpu))

        ram = psutil.virtual_memory().percent
        self.ram_bar.setValue(int(ram))

        try:
            gpus = GPUtil.getGPUs()
            gpu_usage = gpus[0].load * 100 if gpus else 0
        except Exception:
            gpu_usage = 0
        self.gpu_bar.setValue(int(gpu_usage))

        uptime = int(time.time() - self.start_time)
        self.uptime_label.setText(f"{uptime} s")

        current = QDateTime.currentDateTime()
        self.date_label.setText(current.toString("dddd, MMMM dd yyyy"))
        self.time_label.setText(current.toString("hh:mm:ss ap"))

        summary = self.parent().parent().latency_tracker.summary()

        self.fps_label.setText(f"{len(self.parent().parent().cameras) / summary['total']:.1f}")
        self.latency_total.setText(f"{summary['total']*1000:.1f} ms")
        self.latency_camera.setText(f"{summary['capture']*1000:.1f} ms")
        self.latency_inference.setText(f"{summary['inference']*1000:.1f} ms")
        self.latency_display.setText(f"{summary['display']*1000:.1f} ms")


class LogsWidget(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

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

class HeatmapChartWidget(QLabel):
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
