# file: multi_rtsp_yolo_pure_gst.py
import site; site.addsitedir('/usr/lib/python3/dist-packages')
import os, time, threading, queue
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

# --- GStreamer (PyGObject) ---
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import torch
from ultralytics import YOLO


# --- CONFIG ---
MODEL_PATH = str(Path(__file__).parent.parent / "yolo" / "yolo11x.pt")  # adjust to your model
WINDOW = "YOLO 2x2 (low-latency)"
TILE_W, TILE_H = 640, 360
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Draw settings
DRAW_THICKNESS = 2
DRAW_FONT_SCALE = 0.6


def build_pipeline(rtsp_url: str, w: int | None = None, h: int | None = None) -> str:
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


def reader_thread(cam: Cam):
    while cam.alive:
        ok, frame = cam.cap.read()
        if not ok or frame is None:
            time.sleep(0.005)
            continue
        if not cam.q.empty():
            try:
                _ = cam.q.get_nowait()
            except queue.Empty:
                pass
        cam.q.put(frame)


def make_grid(frames, tile_w, tile_h):
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


def draw_boxes(frame, boxes, class_names):
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

def run_in_qt_mode(label_widgets, cameras, stop_flag):
    """
    Same as main(), but instead of cv2.imshow(),
    updates QLabel widgets in a PyQt application.
    """
    if not cameras:
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)
    model.to(device=device)
    if device != "cpu":
        model.fuse()
        try:
            model.model.half()
        except Exception:
            pass

    cams = []
    for cam in cameras:
        cap = GstCam(build_pipeline(cam.full_link, TILE_W, TILE_H))
        cam = Cam(url=cam.full_link, cap=cap, q=queue.Queue(maxsize=1))
        t = threading.Thread(target=reader_thread, args=(cam,), daemon=True)
        t.start()
        cams.append(cam)

    smoothed_fps = 0.0
    last_fps_print = time.perf_counter()

    try:
        while not stop_flag.is_set():
            frames = []
            for cam in cams:
                try:
                    cam.last = cam.q.get(timeout=0.02)
                except queue.Empty:
                    pass
                frames.append(cam.last)

            batch_imgs, idx_map = [], []
            for i, f in enumerate(frames):
                if f is not None:
                    batch_imgs.append(f)
                    idx_map.append(i)

            results = []
            if batch_imgs:
                preds = model.predict(
                    batch_imgs,
                    imgsz=max(TILE_W, TILE_H),
                    device=device,
                    half=(device != "cpu"),
                    conf=0.25,
                    iou=0.45,
                    verbose=False
                )
                results = preds

            if results:
                class_names = results[0].names
                for r, i in zip(results, idx_map):
                    if r.boxes is not None and len(r.boxes) > 0:
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        conf = r.boxes.conf.unsqueeze(1).cpu().numpy()
                        cls  = r.boxes.cls.unsqueeze(1).cpu().numpy()
                        boxes = np.concatenate([xyxy, conf, cls], axis=1)
                        frames[i] = draw_boxes(frames[i], boxes, class_names)

            # update each QLabel
            for i, frame in enumerate(frames):
                if frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
                label_widgets[i].setPixmap(pix.scaled(
                    label_widgets[i].size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))

    finally:
        for cam in cams:
            cam.alive = False
            try:
                cam.cap.release()
            except Exception:
                pass