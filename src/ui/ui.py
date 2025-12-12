# ui.py 25 12 12
"""
Industrial-grade MNIST UI
"""

import sys
from typing import Callable
import numpy as np
from PyQt5.QtCore import Qt, QSize, QPoint, QTimer
from PyQt5.QtGui import QPainter, QPixmap, QPen, QColor, QImage, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QSizePolicy, QLCDNumber, QFileDialog, QGraphicsDropShadowEffect
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------------------
# Theme / Stylesheet
# ---------------------------
STYLESHEET = """
QWidget {
    background-color: #0D0D0D;
    color: #EAEAEA;
    font-family: "Segoe UI";
    font-size: 10pt;
}

#titleLabel {
    font-size: 14pt;
    color: #BFD7FF;
    font-weight: 600;
    letter-spacing: 1px;
}

QFrame.card {
    background-color: #111111;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
}

QLabel.sectionTitle {
    color: #CFCFCF;
    font-weight: 600;
    font-size: 10pt;
}

QPushButton {
    background-color: #151515;
    color: #EAEAEA;
    border: 2px solid #222222;
    padding: 8px 14px;
    border-radius: 8px;
    font-weight: 700;
}

QPushButton:hover {
    background-color: #1F5BFF;
    color: white;
    border: 2px solid #3FA0FF;
}

#watermark {
    color: rgba(255,255,255,0.14);
    font-size: 9pt;
}
"""

# ---------------------------
# Utility: Card container
# ---------------------------
class CardFrame(QFrame):
    def __init__(self, parent=None, padding: int = 10):
        super().__init__(parent)
        self.setObjectName("card")
        self.setProperty("class", "card")
        self.setStyleSheet("")
        self.setContentsMargins(padding, padding, padding, padding)
        # subtle shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 160))
        self.setGraphicsEffect(shadow)
# ---------------------------
# Pixel Canvas with Brush Size
# ---------------------------
class PixelCanvas(QWidget):
    """
    Pixel canvas: logical size = 28x28. Visual size = 28 * cell_size.
    Supports brush size for smoother drawing.
    """

    def __init__(self, canvas_pixels: int = 28, cell_size: int = 18, parent=None):
        super().__init__(parent)
        self.pixels = canvas_pixels
        self.cell_size = cell_size
        self.setFixedSize(QSize(self.pixels * self.cell_size, self.pixels * self.cell_size))

        self.matrix = np.zeros((self.pixels, self.pixels), dtype=np.uint8)
        self.pixmap = QPixmap(self.size())
        self.pixmap.fill(Qt.black)

        self.pen_value = 255
        self.brush_size = 1
        self.last_cell = None

        self._redraw_pixmap()

    # ---------------------------
    # Utility: circle paint brush
    # ---------------------------
    def _paint_circle(self, cx, cy):
        r = self.brush_size
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:  # 圆形笔刷
                    x = cx + dx
                    y = cy + dy
                    if 0 <= x < self.pixels and 0 <= y < self.pixels:
                        self.matrix[y, x] = self.pen_value

    # ---------------------------
    # Internal redraw
    # ---------------------------
    def _redraw_pixmap(self):
        self.pixmap.fill(Qt.black)
        painter = QPainter(self.pixmap)

        # subtle grid
        grid_pen = QPen(QColor(30, 30, 30, 60))
        grid_pen.setWidth(2)
        painter.setPen(grid_pen)

        for y in range(self.pixels):
            for x in range(self.pixels):
                if self.matrix[y, x]:
                    painter.fillRect(
                        x * self.cell_size, y * self.cell_size,
                        self.cell_size, self.cell_size,
                        QColor(255, 255, 255)
                    )
                painter.drawRect(
                    x * self.cell_size, y * self.cell_size,
                    self.cell_size, self.cell_size
                )

        painter.end()
        self.update()

    def paintEvent(self, event):
        QPainter(self).drawPixmap(0, 0, self.pixmap)

    # ---------------------------
    # Coordinate conversion
    # ---------------------------
    def _pos_to_cell(self, pos: QPoint):
        x = pos.x() // self.cell_size
        y = pos.y() // self.cell_size
        if 0 <= x < self.pixels and 0 <= y < self.pixels:
            return x, y
        return None

    # ---------------------------
    # Mouse Events (with brush)
    # ---------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            cell = self._pos_to_cell(event.pos())
            if cell:
                self.last_cell = cell
                self._paint_circle(*cell)
                self._redraw_pixmap()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            cell = self._pos_to_cell(event.pos())
            if cell and cell != self.last_cell:
                self.last_cell = cell
                self._paint_circle(*cell)
                self._redraw_pixmap()

    def mouseReleaseEvent(self, event):
        self.last_cell = None

    # ---------------------------
    # API
    # ---------------------------
    def clear(self):
        self.matrix.fill(0)
        self._redraw_pixmap()

    def get_qimage(self) -> QImage:
        img = QImage(self.pixels, self.pixels, QImage.Format_Grayscale8)
        for y in range(self.pixels):
            for x in range(self.pixels):
                v = int(self.matrix[y, x])
                img.setPixel(x, y, QColor(v, v, v).rgb())
        return img

    def set_matrix(self, mat: np.ndarray):
        assert mat.shape == (self.pixels, self.pixels)
        self.matrix = (mat.astype(np.uint8) > 0).astype(np.uint8) * 255
        self._redraw_pixmap()

# ---------------------------
# Probability Bar Panel (refined)
# ---------------------------
class ProbBarPanel(QWidget):
    """
    Vertical bar chart for 10 probabilities.
    """
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
        # Fully clear figure to avoid leftover texts
        self.figure.clear()

        # Recreate axis
        self.ax = self.figure.add_subplot(111)

        # Black background
        self.ax.set_facecolor("#000000")
        self.figure.patch.set_facecolor("#000000")

        x = np.arange(10)
        bars = self.ax.bar(
            x, probs,
            color="#3D3D3D",
            edgecolor="white",
            width=0.55
        )

        # Remove y axis entirely
        self.ax.set_yticks([])
        self.ax.set_ylabel("")
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)

        self.ax.set_ylim(0, 1.05)

        # Clean grid / borders
        self.ax.grid(False)
        for spine in self.ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(1.2)

        # X-axis ticks
        self.ax.set_xticks(x)
        self.ax.set_xticklabels([str(i) for i in x], color="white", fontsize=11, fontweight="bold")

        # Add probability text on top of each bar
        for rect, p in zip(bars, probs):
            self.ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.02,
                f"{p*100:.1f}%",
                color="#BEBEBE",
                fontsize=9,
                ha='center',
                va='bottom',
                fontweight="bold"
            )

        self.canvas.draw()

# ---------------------------
# Prediction Panel (LCD + title)
# ---------------------------
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
            QLCDNumber {
                background-color: #0B0B0B;
                color: #ffffff;
                border: 2px solid #222;
                border-radius: 8px;
            }
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

# ---------------------------
# Controls Panel
# ---------------------------
class ControlsPanel(CardFrame):
    def __init__(self, parent=None):
        super().__init__(parent, padding=8)
        layout = QHBoxLayout()
        layout.setSpacing(12)

        self.predict_btn = QPushButton("Predict [ENTER]")
        self.clear_btn = QPushButton("Clear [ESC]")

        h = 42
        self.clear_btn.setMinimumHeight(h)
        self.predict_btn.setMinimumHeight(h)

        layout.addWidget(self.predict_btn)
        layout.addWidget(self.clear_btn)

        self.setLayout(layout)

# ---------------------------
# Main Window
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self, on_predict: Callable[[QImage], tuple], canvas_pixels: int = 28, cell_size: int = 18):
        """
        on_predict: callable that accepts a QImage (28x28) and returns (probs: np.ndarray length 10, pred: int)
        """
        super().__init__()
        self.on_predict = on_predict
        self.setWindowTitle("NCDR by EnoLaice")
        self.setStyleSheet(STYLESHEET)
        self.setMinimumSize(1200, 540)

        # central layout: left = canvas card, right = prediction + controls
        central = QWidget()

        # OUTER ROOT LAYOUT (VERTICAL)
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(4)

        ncdrLabel = QLabel(" ▁▂▃ 𝙉 𝘾 𝘿 𝙍 ▃▂▁ \n◇▪ 𝙉𝙚𝙪𝙧𝙖𝙡 𝘾𝙤𝙜𝙣𝙞𝙩𝙞𝙫𝙚 𝘿𝙞𝙜𝙞𝙩𝙨 𝙍𝙚𝙘𝙤𝙜𝙣𝙞𝙩𝙞𝙤𝙣 ▪◇")
        ncdrLabel.setObjectName("ncdrLabel")
        ncdrLabel.setAlignment(Qt.AlignCenter)
        ncdrLabel.setStyleSheet("color:#3c3c3c; font-size:12pt; padding-top:4px;")
        root_layout.addWidget(ncdrLabel, alignment=Qt.AlignCenter)

        # main horizontal layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        root_layout.addLayout(main_layout, stretch=10)

        # Left: Canvas card
        canvas_card = CardFrame()
        canvas_layout = QVBoxLayout()
        canvas_layout.setSpacing(8)
        # Title
        title = QLabel("■ Canvas Panel")
        title.setStyleSheet("font-weight:700; color:#bebebe;")
        canvas_layout.addWidget(title, alignment=Qt.AlignCenter)

        # Canvas widget
        self.canvas = PixelCanvas(canvas_pixels, cell_size)
        canvas_layout.addWidget(self.canvas, alignment=Qt.AlignCenter)

        hint = QLabel("Draw — 28×28 pixels")
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
        watermark.setObjectName("watermark")
        watermark.setAlignment(Qt.AlignCenter)
        watermark.setStyleSheet("color:#888; font-size:9pt; padding-top:4px;")
        root_layout.addWidget(watermark, alignment=Qt.AlignCenter)

        central.setLayout(root_layout)
        self.setCentralWidget(central)

        # Connect buttons
        self.controls.clear_btn.clicked.connect(self._on_clear)
        self.controls.predict_btn.clicked.connect(self._on_predict)

        # Shortcuts
        self.controls.predict_btn.setShortcut("Return")
        self.controls.clear_btn.setShortcut("Escape")

    # -----------------------
    # Button handlers (expected to be connected to user functions)
    # -----------------------
    def _on_clear(self):
        self.canvas.clear()
        self.pred_panel.update(np.zeros(10), 0)

    def _on_predict(self):
        qimg = self.canvas.get_qimage()
        try:
            probs, pred = self.on_predict(qimg)
            probs = np.asarray(probs, dtype=float).flatten()
            self.pred_panel.update(probs, int(pred))
        except Exception as e:
            # show error on title

            self.pred_panel.findChild(QLabel).setText(f"Error: {e}")
