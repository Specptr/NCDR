# ui.py
"""
MNIST UI with Real-Time Prediction
"""
import sys
from typing import Callable
import numpy as np
from PyQt5.QtCore import Qt, QSize, QPoint, QRect
from PyQt5.QtGui import QPainter, QPixmap, QColor, QImage, QPen
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QLCDNumber, QGraphicsDropShadowEffect,
    QRadioButton, QButtonGroup
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from config import STYLESHEET

class CardFrame(QFrame):
    def __init__(self, parent=None, padding: int = 10):
        super().__init__(parent)
        self.setObjectName("card")
        self.setProperty("class", "card")
        self.setStyleSheet("")
        self.setContentsMargins(padding, padding, padding, padding)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 160))
        self.setGraphicsEffect(shadow)

class PixelCanvas(QWidget):
    """
    Free drawing canvas implemented with a QPixmap backing store.
    Visual size = canvas_pixels * cell_size (keeps original signature compatible).
    Provides:
      - undo stack
      - clear
      - get_qimage(resize=(w,h)) -> returns grayscale QImage (useful for model)
    """
    def __init__(self, canvas_pixels: int = 28, cell_size: int = 18, parent=None, on_draw: Callable = None):
        super().__init__(parent)
        # keep compatibility: compute width/height same as your old visual size
        self.visual_w = canvas_pixels * cell_size
        self.visual_h = canvas_pixels * cell_size
        self.setFixedSize(QSize(self.visual_w, self.visual_h))

        # drawing state
        self.brush_size = 3  # default visually comfortable
        self.brush_color = QColor(255, 255, 255)
        self._drawing = False
        self._last_pos = QPoint()

        # pixmap backing store
        self.pixmap = QPixmap(self.visual_w, self.visual_h)
        self.pixmap.fill(Qt.black)

        # undo stack (store QPixmap copies). Push on mouse press for performance.
        self.undo_stack = []
        self.undo_limit = 30  # keep limited history

        # callback used by MainWindow to perform prediction
        self.on_draw = on_draw

    # ---- drawing control ----
    def set_brush_size(self, s: int):
        # map simple integers to pen widths (1,2,3) -> 2,6,12 maybe; keep small scale
        if s <= 0:
            s = 1
        # choose mapping: 1->2, 2->6, 3->12 for distinct feel
        map_tbl = {1: 12, 2: 24, 3: 36}
        self.brush_size = map_tbl.get(s, max(1, s))

    def undo(self):
        if self.undo_stack:
            pix = self.undo_stack.pop()
            # restore pixmap
            self.pixmap = pix
            self.update()

    def clear(self):
        # save current to undo stack
        self._push_undo()
        self.pixmap.fill(Qt.black)
        self.update()
        # trigger prediction (empty)
        if self.on_draw:
            qimg = self.get_qimage(resize=(28, 28))
            self.on_draw(qimg)

    # ---- internal helpers ----
    def _push_undo(self):
        # push a copy of current pixmap
        if len(self.undo_stack) >= self.undo_limit:
            # drop oldest
            self.undo_stack.pop(0)
        self.undo_stack.append(self.pixmap.copy())

    # ---- mouse events ----
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # push undo snapshot once at stroke start (fast)
            self._push_undo()
            self._drawing = True
            self._last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self._drawing:
            painter = QPainter(self.pixmap)
            pen = QPen(self.brush_color)
            pen.setWidth(self.brush_size)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self._last_pos, event.pos())
            painter.end()
            self._last_pos = event.pos()
            # update widget to redraw
            self.update()
            # realtime prediction: send resized image (28x28) for model input
            if self.on_draw:
                try:
                    qimg = self.get_qimage(resize=(28, 28))
                    self.on_draw(qimg)
                except Exception:
                    pass

    def mouseReleaseEvent(self, event):
        self._drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        # draw the backing pixmap
        painter.drawPixmap(0, 0, self.pixmap)

    # ---- expose QImage for model ----
    def get_qimage(self, resize: tuple = None) -> QImage:
        """
        Return grayscale QImage of current canvas.
        If resize is provided (w,h), scale to that size (useful to produce 28x28).
        """
        # obtain QImage from pixmap (RGBA/ARGB) then convert to grayscale
        qimg_rgba = self.pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
        if resize is not None:
            w, h = resize
            qimg_scaled = qimg_rgba.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # ensure final size exactly matches (pad if necessary)
            if qimg_scaled.width() != w or qimg_scaled.height() != h:
                # create target and paint centered
                target = QImage(w, h, QImage.Format_Grayscale8)
                target.fill(Qt.black)
                p = QPainter(target)
                x = (w - qimg_scaled.width()) // 2
                y = (h - qimg_scaled.height()) // 2
                p.drawImage(x, y, qimg_scaled)
                p.end()
                return target
            return qimg_scaled
        return qimg_rgba

    # allow external set (keeps compatibility)
    def set_pixmap_from_numpy(self, arr: np.ndarray):
        """
        Accept numpy array grayscale [H,W] 0..255 and set canvas.
        """
        h, w = arr.shape
        img = QImage(arr.data.tobytes(), w, h, w, QImage.Format_Grayscale8)
        # scale to canvas size
        qimg = img.scaled(self.visual_w, self.visual_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.pixmap = QPixmap.fromImage(qimg)
        self.update()
        if self.on_draw:
            self.on_draw(self.get_qimage(resize=(28, 28)))

class ProbBarPanel(QWidget):
    def __init__(self, parent=None, figsize=(3.0, 3.0)):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, dpi=100, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.update_probs(np.zeros(10))

    def update_probs(self, probs: np.ndarray):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#000000")
        self.figure.patch.set_facecolor("#000000")
        x = np.arange(10)
        bars = self.ax.bar(x, probs, color="#3D3D3D", edgecolor="white", width=0.55)
        self.ax.set_yticks([])
        self.ax.set_ylabel("")
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.set_ylim(0, 1.05)
        self.ax.grid(False)
        for spine in self.ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(1.2)
        self.ax.set_xticks(x)
        self.ax.set_xticklabels([str(i) for i in x], color="white", fontsize=11, fontweight="bold")
        for rect, p in zip(bars, probs):
            self.ax.text(rect.get_x() + rect.get_width() / 2,
                         rect.get_height() + 0.02,
                         f"{p*100:.1f}%", color="#BEBEBE", fontsize=9,
                         ha='center', va='bottom', fontweight="bold")
        self.canvas.draw()

class PredictionPanel(CardFrame):
    def __init__(self, parent=None):
        super().__init__(parent, padding=12)
        layout = QVBoxLayout()
        layout.setSpacing(8)
        title = QLabel("≡ Model Prediction ≡")
        title.setObjectName("sectionTitle")
        title.setProperty("class", "sectionTitle")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight:700; color:#bebebe;")
        layout.addWidget(title)
        self.lcd = QLCDNumber(self)
        self.lcd.setDigitCount(1)
        self.lcd.setSegmentStyle(QLCDNumber.Flat)
        self.lcd.setFixedSize(180, 120)

        self.lcd.setStyleSheet("""

        """)
        layout.addWidget(self.lcd, alignment=Qt.AlignCenter)
        self.prob_panel = ProbBarPanel(figsize=(3.0, 3.0))
        layout.addWidget(self.prob_panel)
        self.setLayout(layout)

    def update(self, probs: np.ndarray, pred: int):
        self.prob_panel.update_probs(probs)
        try:
            self.lcd.display(int(pred))
        except Exception:
            self.lcd.display(0)

class ControlsPanel(CardFrame):
    def __init__(self, parent=None):
        super().__init__(parent, padding=8)
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)

        # top buttons
        button_row = QHBoxLayout()
        button_row.setSpacing(12)

        self.undo_btn = QPushButton("Undo [Ctrl+Z]")
        self.clear_btn = QPushButton("Clear [ESC]")

        h = 42
        self.clear_btn.setMinimumHeight(h)
        self.undo_btn.setMinimumHeight(h)

        button_row.addWidget(self.undo_btn)
        button_row.addWidget(self.clear_btn)

        # brush selector row
        brush_row = QHBoxLayout()
        brush_row.setSpacing(8)

        brush_title = QLabel("Brush sizes:")
        brush_title.setStyleSheet("font-weight:600; color:#C0C0C0;")

        self.radio1 = QRadioButton("Thin")
        self.radio2 = QRadioButton("Normal")
        self.radio3 = QRadioButton("Thick")
        self.radio2.setChecked(True)

        self.brush_group = QButtonGroup(self)
        self.brush_group.addButton(self.radio1, 1)
        self.brush_group.addButton(self.radio2, 2)
        self.brush_group.addButton(self.radio3, 3)


        brush_row.addWidget(brush_title)
        brush_row.addWidget(self.radio1)
        brush_row.addWidget(self.radio2)
        brush_row.addWidget(self.radio3)
        brush_row.addStretch()

        main_layout.addLayout(button_row)
        main_layout.addLayout(brush_row)
        self.setLayout(main_layout)

class MainWindow(QMainWindow):
    def __init__(self, on_predict: Callable[[QImage], tuple], canvas_pixels: int = 28, cell_size: int = 18):
        """
        on_predict: callable that accepts a QImage (28x28) and returns (probs: np.ndarray length 10, pred: int)
        canvas_pixels & cell_size kept for visual size backward compatibility:
          visual_width = canvas_pixels * cell_size
        """
        super().__init__()
        self.on_predict = on_predict
        self.setWindowTitle("NCDR by EnoLaice")
        self.setStyleSheet(STYLESHEET)
        self.setMinimumSize(1200, 540)

        # central layout
        central = QWidget()
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(4)

        ncdrLabel = QLabel(" ▁▂▃ 𝙉 𝘾 𝘿 𝙍 ▃▂▁ \n◇▪ 𝙉𝙚𝙪𝙧𝙖𝙡 𝘾𝙤𝙜𝙣𝙞𝙩𝙞𝙫𝙚 𝘿𝙞𝙜𝙞𝙩𝙨 𝙍𝙚𝙘𝙤𝙜𝙣𝙞𝙩𝙞𝙤𝙣 ▪◇")
        ncdrLabel.setAlignment(Qt.AlignCenter)
        ncdrLabel.setStyleSheet("color:#3c3c3c; font-size:12pt; padding-top:4px;")
        root_layout.addWidget(ncdrLabel, alignment=Qt.AlignCenter)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        root_layout.addLayout(main_layout, stretch=10)

        # Left canvas card
        canvas_card = CardFrame()
        canvas_layout = QVBoxLayout()
        canvas_layout.setSpacing(8)
        title = QLabel("■ Canvas Panel ■")
        title.setStyleSheet("font-weight:700; color:#bebebe;")
        canvas_layout.addWidget(title, alignment=Qt.AlignCenter)

        self.canvas = PixelCanvas(canvas_pixels, cell_size, on_draw=self._on_draw)
        # override visual size to match original calculation (PixelCanvas already does this)
        canvas_layout.addWidget(self.canvas, alignment=Qt.AlignCenter)

        hint = QLabel("Canvas will be resized to 28×28 for prediction")
        hint.setStyleSheet("color: #9AA7B2; font-size: 9pt;")
        canvas_layout.addWidget(hint, alignment=Qt.AlignCenter)

        canvas_card.setLayout(canvas_layout)
        main_layout.addWidget(canvas_card, stretch=3)

        # Right: Prediction + controls
        right_col = QVBoxLayout()
        right_col.setSpacing(12)
        self.pred_panel = PredictionPanel()
        right_col.addWidget(self.pred_panel, stretch=5)
        self.controls = ControlsPanel()
        right_col.addWidget(self.controls, stretch=1)
        main_layout.addLayout(right_col, stretch=4)

        watermark = QLabel("◇▪  By EnoLaice  ▪◇")
        watermark.setAlignment(Qt.AlignCenter)
        watermark.setStyleSheet("color:#888; font-size:9pt; padding-top:4px;")
        root_layout.addWidget(watermark, alignment=Qt.AlignCenter)

        central.setLayout(root_layout)
        self.setCentralWidget(central)

        # Connect buttons & shortcuts
        self.controls.clear_btn.clicked.connect(self._on_clear)
        self.controls.undo_btn.clicked.connect(self.canvas.undo)
        self.controls.undo_btn.setShortcut("Ctrl+Z")
        self.controls.clear_btn.setShortcut("Escape")

        # brush size binding (1/2/3)
        self.controls.brush_group.buttonClicked[int].connect(self._on_brush_changed)
        # set initial brush from default radio
        checked = self.controls.brush_group.checkedId()
        if checked <= 0:
            checked = 2
        self.canvas.set_brush_size(checked)

        # initialize empty prediction
        self.pred_panel.update(np.zeros(10), 0)

    def _on_brush_changed(self, size: int):
        # size will be 1/2/3 as assigned
        try:
            self.canvas.set_brush_size(size)
        except Exception:
            pass

    def _on_draw(self, qimg: QImage):
        """Real-time callback on every brush stroke (qimg is 28x28)"""
        try:
            # call user-provided predictor
            probs, pred = self.on_predict(qimg)
            probs = np.asarray(probs, dtype=float).flatten()
            self.pred_panel.update(probs, int(pred))
        except Exception:
            # swallow exceptions to avoid UI freeze on prediction errors
            pass

    def _on_clear(self):
        self.canvas.clear()
        self.pred_panel.update(np.zeros(10), 0)
