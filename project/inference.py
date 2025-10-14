# file: multi_rtsp_yolo_pure_gst.py
import site; site.addsitedir('/usr/lib/python3/dist-packages')
import os, time, threading, queue
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

# --- GStreamer (PyGObject) ---
import gi # type: ignore
gi.require_version('Gst', '1.0')
from gi.repository import Gst # type: ignore
Gst.init(None)

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Slot
import torch
from ultralytics import YOLO
from classes import Camera, ProxyCamera


# --- CONFIG ---
MODEL_PATH = str(Path(__file__).parent.parent / "yolo" / "yolo11x.pt")  # adjust to your model
WINDOW = "YOLO 2x2 (low-latency)"
TILE_W, TILE_H = 640, 360
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Draw settings
DRAW_THICKNESS = 2
DRAW_FONT_SCALE = 0.6

class GstCam:
    """Minimal wrapper around a GStreamer pipeline with an appsink."""
    def __init__(self, pipeline_str: str):
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("appsink")
        if self.appsink is None:
            raise RuntimeError("appsink not found in pipeline")
        self.appsink.set_property("emit-signals", False)
        self.appsink.set_property("sync", False)
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("drop", True)
        self.pipeline.set_state(Gst.State.PLAYING)

    def read(self):
        # Non-blocking: returns (ok, frame) or (False, None) if no fresh frame
        sample = self.appsink.emit("try-pull-sample", 0)
        if sample is None:
            return False, None
        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        w = s.get_value('width'); h = s.get_value('height')
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            # IMPORTANT: make a WRITEABLE copy so OpenCV can draw on it
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3)).copy()
        finally:
            buf.unmap(mapinfo)
        return True, frame

    def release(self):
        self.pipeline.set_state(Gst.State.NULL)

@dataclass
class Cam:
    url: str
    cap: GstCam
    q: "queue.Queue[np.ndarray]"
    last: Optional[np.ndarray] = None
    alive: bool = True

class Inferencer:
    def __init__(self, label_widgets, target_sources, stop_flag, latency_tracker, display_settings):
        self.label_widgets = label_widgets
        self.target_sources = target_sources
        self.stop_flag = stop_flag
        self.latency_tracker = latency_tracker
        self.display_settings = display_settings

    def build_pipeline(self, rtsp_url: str, w: int | None = None, h: int | None = None) -> str:
        """
        Builds the GStreamer pipeline.
        Low-latency pipeline:
        - small RTSP jitter buffer (latency=50)
        - decode → convert to BGR
        - appsink drops old frames and never blocks
        """
        size_caps = ""
        if w is not None and h is not None:
            size_caps = f" ! videoscale ! video/x-raw,width={w},height={h},format=BGR"
        return (
            f'rtspsrc location="{rtsp_url}" latency=50 protocols=tcp ! '
            f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert{size_caps} ! '
            f'appsink name=appsink caps=video/x-raw,format=BGR drop=true max-buffers=1 sync=false'
        )


    def reader_thread(self,cam: Cam):
        while cam.alive:
            frame_start = time.perf_counter()
            ok, frame = cam.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            if not cam.q.empty():
                try:
                    _ = cam.q.get_nowait()
                except queue.Empty:
                    pass
            capture_latency = time.perf_counter() - frame_start
            self.latency_tracker.record_capture(capture_latency)
            cam.q.put(frame)


    def make_grid(self, frames, tile_w, tile_h):
        tiles = []
        for f in frames:
            if f is None:
                tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
            else:
                tiles.append(cv2.resize(f, (tile_w, tile_h)))
        while len(tiles) < 4:
            tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
        top = np.hstack((tiles[0], tiles[1]))
        bot = np.hstack((tiles[2], tiles[3]))
        return np.vstack((top, bot))


    def draw_boxes(self, frame, boxes, class_names):
        """
        boxes: (N,6) array [x1,y1,x2,y2,conf,cls]
        Draws ALL classes so you can see boxes.
        """
        if boxes is None or len(boxes) == 0:
            return frame
        for x1, y1, x2, y2, conf, cls in boxes:
            name = class_names[int(cls)]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            label = f"{name} {float(conf):.2f}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), DRAW_THICKNESS)
            cv2.putText(frame, label, (x1 + 3, max(0, y1 - 4)), FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

    def _init_yolo(self):
        """Initialize YOLO model and send to device."""
        device = 0 if torch.cuda.is_available() else "cpu"
        model = YOLO(MODEL_PATH)
        model.to(device=device)

        if device != "cpu":
            model.fuse()
            try:
                model.model.half()
            except Exception:
                pass

        return model, device

    def _batch_infer(self, model, batch_imgs, device):
        """Run YOLO prediction and record latency."""
        inf_start = time.perf_counter()
        preds = model.predict(
            batch_imgs,
            imgsz=max(TILE_W, TILE_H),
            device=device,
            half=(device != "cpu"),
            conf=0.25,
            iou=0.45,
            verbose=False
        )
        self.latency_tracker.record_inference(time.perf_counter() - inf_start)
        return preds


    def _display_frames(self, frames, results, idx_map):
        """Draw boxes and display frames on their respective QLabel widgets."""
        display_start = time.perf_counter()
        if results:
            class_names = results[0].names
            for r, i in zip(results, idx_map):
                num_objs = 0
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    conf = r.boxes.conf.unsqueeze(1).cpu().numpy()
                    cls  = r.boxes.cls.unsqueeze(1).cpu().numpy()
                    boxes = np.concatenate([xyxy, conf, cls], axis=1)
                    frames[i] = self.draw_boxes(frames[i], boxes, class_names)

                    num_objs = len(r.boxes.cls)

                cam = self.target_sources[i]  # the corresponding camera or proxy
                frames[i] = self.draw_osd(frames[i], cam, num_objs)

        # Display all frames
        for i, frame in enumerate(frames):
            if frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.label_widgets[i].setPixmap(pix.scaled(
                self.label_widgets[i].size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

        self.latency_tracker.record_display(time.perf_counter() - display_start)

    def draw_osd(self, frame, cam, num_objs):
        """
        Draw On-Screen Display (OSD) info such as camera name, location,
        and congestion. Background size auto-adjusts to text content.
        """
        if not hasattr(self, "display_settings"):
            return frame

        display_settings = getattr(self, "display_settings", {})
        osd = display_settings.get("osd", {})

        # --- Adjustable visual parameters ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        line_h = int(25 * scale)
        padding = 10
        margin_x = 10
        margin_y = 10
        color = (255, 255, 255)
        bg_color = (30, 30, 30)
        alpha = 0.3

        # --- Generate all text lines to draw ---
        lines = []
        if osd.get("name", False):
            lines.append(f"Cam: {getattr(cam, 'name', 'Unknown')}")
        if osd.get("location", False):
            lines.append(f"Loc: {getattr(cam, 'location', 'Unknown')}")
        if osd.get("estimated_congestion", False):
            try:
                max_cap = int(getattr(cam, 'max_cap', 100))
            except (ValueError, TypeError):
                max_cap = 100
            congestion_ratio = num_objs / max_cap if max_cap > 0 else 0
            lines.append(f"Congestion: {congestion_ratio:.2f}")

        if not lines:
            return frame

        # --- Compute dynamic size using cv2.getTextSize ---
        text_widths = []
        for text in lines:
            (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
            text_widths.append(tw)

        rect_width = padding * 2 + max(text_widths)
        rect_height = padding * 2 + line_h * len(lines)

        # --- Draw translucent background ---
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (margin_x, margin_y),
            (margin_x + rect_width, margin_y + rect_height),
            bg_color,
            -1
        )
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # --- Draw text lines ---
        y = margin_y + padding + int(line_h * 0.8)
        for text in lines:
            cv2.putText(frame, text, (margin_x + padding, y),
                        font, scale, color, thickness, cv2.LINE_AA)
            y += line_h

        return frame


    @Slot(dict)
    def on_display_settings_changed(self, display_dict):
        self.display_settings = display_dict.copy()
        print(f"[INFO] OSD settings updated: {self.display_settings}")

    def run(self):
        if not self.target_sources:
            return
        if isinstance(self.target_sources[0], Camera):
            self.run_rtsp_mode()
        if isinstance(self.target_sources[0], ProxyCamera):
            self.run_video_mode()

    def run_video_mode(self):
        model, device = self._init_yolo()
        caps = []

        # OpenCV-based sources (ProxyCamera.file_path)
        for proxy in self.target_sources:
            cap = cv2.VideoCapture(proxy.file_path)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open video: {proxy.file_path}")
                continue
            caps.append(cap)

        try:
            while not self.stop_flag.is_set():
                capture_start = time.perf_counter()
                frames = []
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    frames.append(frame)
                self.latency_tracker.record_capture(time.perf_counter() - capture_start)

                batch_imgs, idx_map = [], []
                for i, f in enumerate(frames):
                    if f is not None:
                        batch_imgs.append(f)
                        idx_map.append(i)

                results = []
                if batch_imgs:
                    results = self._batch_infer(model, batch_imgs, device)

                self._display_frames(frames, results, idx_map)

        finally:
            for cap in caps:
                cap.release()

    def run_rtsp_mode(self):
        """ Runs the inferencer in RTSP mode. """
        model, device = self._init_yolo()
        cams = []

        for cam in self.target_sources:
            cap = GstCam(self.build_pipeline(cam.full_link, TILE_W, TILE_H))
            cam = Cam(url=cam.full_link, cap=cap, q=queue.Queue(maxsize=1))
            t = threading.Thread(target=self.reader_thread, args=(cam,), daemon=True)
            t.start()
            cams.append(cam)

        try:
            while not self.stop_flag.is_set():
                capture_start = time.perf_counter()
                frames = []
                for cam in cams:
                    try:
                        cam.last = cam.q.get(timeout=0.02)
                    except queue.Empty:
                        pass
                    frames.append(cam.last)
                self.latency_tracker.record_capture(time.perf_counter() - capture_start)

                batch_imgs, idx_map = [], []
                for i, f in enumerate(frames):
                    if f is not None:
                        batch_imgs.append(f)
                        idx_map.append(i)

                results = []

                # Batch infer
                if batch_imgs:
                    results = self._batch_infer(model, batch_imgs, device)
                
                self._display_frames(frames, results, idx_map)

        finally:
            for cam in cams:
                cam.alive = False
                try:
                    cam.cap.release()
                except Exception:
                    pass