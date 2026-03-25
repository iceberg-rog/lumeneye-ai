import queue
import traceback
from pathlib import Path

import cv2
from PySide6.QtCore import QSignalBlocker, QTimer, Qt, QRect, Signal, QSize
from PySide6.QtGui import QCloseEvent, QIcon, QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QAbstractItemView,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.config import PROFILES_DIR
from vision.local_object import build_detect_prompt, looks_meaningful_text


APP_QT = None
UI_ERROR_LOG = Path(__file__).resolve().parent.parent / 'data' / 'events' / 'ui_errors.log'


def log_ui_exception(context, exc):
    UI_ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(UI_ERROR_LOG, 'a', encoding='utf-8') as handle:
        handle.write(f'[{context}] {exc}\n')
        handle.write(traceback.format_exc())
        handle.write('\n')


class PreviewLabel(QLabel):
    selectionChanged = Signal(object)
    selectionStarted = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(920, 520)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet('background:#0b1625; border:1px solid #223c5c; border-radius:16px;')
        self.frame = None
        self.image_box = None
        self.drag_origin = None
        self.selection = None
        self._rendering = False

    def set_frame(self, frame):
        self.frame = frame
        self._render_frame()

    def _render_frame(self):
        if self._rendering:
            return
        frame = self.frame
        if frame is None:
            self.setText('Waiting for live video feed...')
            self.setPixmap(QPixmap())
            return
        self._rendering = True
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            scaled = image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            left = (self.width() - scaled.width()) // 2
            top = (self.height() - scaled.height()) // 2
            self.image_box = (left, top, scaled.width(), scaled.height())
            pixmap = QPixmap.fromImage(scaled)
            self.setPixmap(pixmap)
        finally:
            self._rendering = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.frame is not None:
            QTimer.singleShot(0, self._render_frame)

    def mousePressEvent(self, event: QMouseEvent):
        if self.frame is None or event.button() != Qt.LeftButton:
            return
        self.selectionStarted.emit()
        self.drag_origin = event.position().toPoint()
        self.selection = QRect(self.drag_origin, self.drag_origin)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drag_origin is None:
            return
        current = event.position().toPoint()
        self.selection = QRect(self.drag_origin, current).normalized()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.drag_origin is None:
            return
        current = event.position().toPoint()
        self.selection = QRect(self.drag_origin, current).normalized()
        self.drag_origin = None
        self.update()
        self.selectionChanged.emit(self.selected_roi())

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selection and not self.selection.isNull():
            from PySide6.QtGui import QPainter, QPen
            painter = QPainter(self)
            pen = QPen(Qt.GlobalColor.green, 2)
            painter.setPen(pen)
            painter.drawRect(self.selection)
            painter.end()

    def selected_roi(self):
        if self.frame is None or not self.selection or self.image_box is None:
            return None
        left, top, disp_w, disp_h = self.image_box
        rect = self.selection.normalized()
        x1 = max(left, min(rect.left(), left + disp_w))
        y1 = max(top, min(rect.top(), top + disp_h))
        x2 = max(left, min(rect.right(), left + disp_w))
        y2 = max(top, min(rect.bottom(), top + disp_h))
        if abs(x2 - x1) < 6 or abs(y2 - y1) < 6:
            return None
        src_h, src_w = self.frame.shape[:2]
        scale_x = src_w / float(disp_w)
        scale_y = src_h / float(disp_h)
        rx = int((min(x1, x2) - left) * scale_x)
        ry = int((min(y1, y2) - top) * scale_y)
        rw = int(abs(x2 - x1) * scale_x)
        rh = int(abs(y2 - y1) * scale_y)
        return (max(0, rx), max(0, ry), max(2, rw), max(2, rh))


class StaticImageLabel(QLabel):
    def __init__(self, minimum_width=320, minimum_height=220):
        super().__init__()
        self.setMinimumSize(minimum_width, minimum_height)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet('background:#0b1625; border:1px solid #223c5c; border-radius:16px;')
        self._frame = None
        self._rendering = False
        self.setText('No saved sample yet.')

    def set_frame(self, frame):
        self._frame = frame
        self._render_frame()

    def _render_frame(self):
        if self._rendering:
            return
        frame = self._frame
        if frame is None:
            self.setText('No saved sample yet.')
            self.setPixmap(QPixmap())
            return
        self._rendering = True
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            scaled = image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(QPixmap.fromImage(scaled))
        finally:
            self._rendering = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._frame is not None:
            QTimer.singleShot(0, self._render_frame)


class GuidedCaptureDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Guided Capture Wizard')
        self.setModal(True)
        self.resize(460, 260)
        layout = QVBoxLayout(self)
        intro = QLabel(
            'This wizard will auto-capture multiple samples of the current model from the live feed.\n\n'
            'Keep the object centered, then slightly rotate or move it between captures so the model learns multiple views.'
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.sample_count = QSpinBox()
        self.sample_count.setRange(2, 8)
        self.sample_count.setValue(4)
        form.addRow('Number of samples', self.sample_count)

        self.delay_ms = QSpinBox()
        self.delay_ms.setRange(700, 3000)
        self.delay_ms.setSingleStep(100)
        self.delay_ms.setValue(1300)
        form.addRow('Delay between samples (ms)', self.delay_ms)
        layout.addLayout(form)

        tip = QLabel('Tip: use English category and detect prompts, for example: blue pen, bic pen, ballpoint pen.')
        tip.setWordWrap(True)
        layout.addWidget(tip)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class DesktopDashboard(QMainWindow):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setWindowTitle('LumenEye AI')
        self.resize(1540, 980)
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#08111d; color:#eef5ff; font-family:'Segoe UI'; }
            QFrame#panel { background:#102035; border:1px solid #223c5c; border-radius:18px; }
            QLabel#title { font-size:30px; font-weight:700; }
            QLabel#muted { color:#a8bcda; font-size:13px; }
            QLabel#sectionTitle { color:#f3f8ff; font-size:20px; font-weight:700; }
            QLabel#fieldLabel { color:#d6e4fb; font-size:13px; font-weight:600; padding-top:4px; }
            QLabel#cardTitle { color:#f3f8ff; font-size:18px; font-weight:700; }
            QLabel#cardValue { color:#ffffff; font-size:30px; font-weight:700; }
            QPushButton { background:#1a3554; border:none; border-radius:12px; padding:11px 14px; color:#eef5ff; font-weight:600; min-height:42px; }
            QPushButton#primary { background:#1f8f73; }
            QPushButton:hover { background:#214264; }
            QPushButton#primary:hover { background:#247e67; }
            QLineEdit, QTextEdit, QComboBox { background:#0d1a2a; border:1px solid #315072; border-radius:10px; padding:10px; color:#eef5ff; selection-background-color:#2d5b86; min-height:22px; }
            QTabWidget::pane { border:0; }
            QTabBar::tab { background:#132740; color:#dce8ff; padding:12px 18px; border-radius:14px; margin-right:8px; min-width:120px; font-size:13px; font-weight:600; }
            QTabBar::tab:selected { background:#1e3b5f; color:#ffffff; }
            QTreeWidget, QListWidget, QTableWidget { background:#0d1a2a; border:1px solid #27415f; border-radius:12px; color:#eef5ff; }
            QHeaderView::section { background:#17314d; color:#b7cae8; padding:8px; border:none; }
        """)
        self.current_frame = None
        self.frozen_preview_frame = None
        self.preview_is_frozen = False
        self.selected_profile_name = ''
        self.pause_refresh_until = 0.0
        self.last_profiles_signature = ()
        self.last_events_signature = ()
        self.last_preview_frame_index = -1
        self.last_saved_signature = ('', '')
        self.last_gallery_signature = ('', 0)
        self.wizard_timer = QTimer(self)
        self.wizard_timer.timeout.connect(self._wizard_step)
        self.wizard_total = 0
        self.wizard_remaining = 0
        self.wizard_interval_ms = 1300
        self.wizard_metadata = None
        self._build_ui()
        self.clear_form()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(180)

    def _panel(self):
        panel = QFrame()
        panel.setObjectName('panel')
        return panel

    def _metric_card(self, title):
        card = self._panel()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        small = QLabel(title.upper())
        small.setObjectName('fieldLabel')
        big = QLabel('0')
        big.setObjectName('cardValue')
        layout.addWidget(small)
        layout.addWidget(big)
        return card, big

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(16)

        hero = QHBoxLayout()
        root.addLayout(hero)

        left = self._panel()
        left_l = QVBoxLayout(left)
        left_l.setSpacing(14)
        title = QLabel('LumenEye AI')
        title.setObjectName('title')
        subtitle = QLabel('Local desktop control center for model definition, live verification, and open-vocabulary detection.')
        subtitle.setObjectName('muted')
        subtitle.setWordWrap(True)
        left_l.addWidget(title)
        left_l.addWidget(subtitle)

        metrics_row = QHBoxLayout()
        self.metric_widgets = {}
        for key, title_text in [('fps', 'FPS'), ('frames', 'Frames'), ('models', 'Models'), ('events', 'Events')]:
            card, big = self._metric_card(title_text)
            metrics_row.addWidget(card)
            self.metric_widgets[key] = big
        left_l.addLayout(metrics_row)
        hero.addWidget(left, 3)

        right = self._panel()
        right_l = QVBoxLayout(right)
        right_l.setSpacing(12)
        self.detector_title = QLabel('Detector')
        self.detector_title.setObjectName('cardTitle')
        self.detector_status = QLabel('Standby')
        self.detector_status.setObjectName('muted')
        self.self_status = QLabel('Identity pending')
        self.self_status.setObjectName('cardTitle')
        self.self_detail = QLabel('-')
        self.self_detail.setObjectName('muted')
        self.prompt_label = QLabel('Prompts: -')
        self.prompt_label.setObjectName('muted')
        for widget in [self.detector_title, self.detector_status, self.self_status, self.self_detail, self.prompt_label]:
            widget.setWordWrap(True)
            right_l.addWidget(widget)
        right_l.addStretch(1)
        hero.addWidget(right, 1)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        self.overview_tab = QWidget()
        self.studio_tab = QWidget()
        self.events_tab = QWidget()
        self.registry_tab = QWidget()
        self.tabs.addTab(self.overview_tab, 'Overview')
        self.tabs.addTab(self.studio_tab, 'Model Studio')
        self.tabs.addTab(self.events_tab, 'Events')
        self.tabs.addTab(self.registry_tab, 'Registry')

        self._build_overview()
        self._build_studio()
        self._build_events()
        self._build_registry()

    def _build_overview(self):
        layout = QVBoxLayout(self.overview_tab)
        top = QHBoxLayout()
        layout.addLayout(top)
        self.ov_cards = []
        for title in ['Detector Readiness', 'Live Matches', 'Prompt Coverage']:
            card = self._panel()
            l = QVBoxLayout(card)
            head = QLabel(title)
            head.setObjectName('cardTitle')
            big = QLabel('-')
            big.setObjectName('cardValue')
            desc = QLabel('-')
            desc.setObjectName('muted')
            desc.setWordWrap(True)
            l.addWidget(head)
            l.addWidget(big)
            l.addWidget(desc)
            top.addWidget(card)
            self.ov_cards.append((big, desc))

        bottom = QHBoxLayout()
        layout.addLayout(bottom, 1)

        preview_panel = self._panel()
        preview_layout = QVBoxLayout(preview_panel)
        preview_title = QLabel('Scene Monitor')
        preview_title.setObjectName('sectionTitle')
        preview_layout.addWidget(preview_title)
        preview_note = QLabel('Live scene with current overlays for verified objects.')
        preview_note.setObjectName('muted')
        preview_note.setWordWrap(True)
        preview_layout.addWidget(preview_note)
        self.overview_preview = PreviewLabel()
        self.overview_preview.setMinimumSize(760, 420)
        preview_layout.addWidget(self.overview_preview, 1)
        bottom.addWidget(preview_panel, 3)

        insight_panel = self._panel()
        insight_layout = QVBoxLayout(insight_panel)
        insight_title = QLabel('Detected Models')
        insight_title.setObjectName('sectionTitle')
        insight_layout.addWidget(insight_title)
        insight_note = QLabel('Models that are currently visible with their confidence and prompt trace.')
        insight_note.setObjectName('muted')
        insight_note.setWordWrap(True)
        insight_layout.addWidget(insight_note)
        self.overview_matches = QListWidget()
        insight_layout.addWidget(self.overview_matches, 1)
        bottom.addWidget(insight_panel, 2)

    def _build_studio(self):
        layout = QHBoxLayout(self.studio_tab)
        left = self._panel()
        left_l = QVBoxLayout(left)
        head = QLabel('Live Preview')
        head.setObjectName('sectionTitle')
        left_l.addWidget(head)
        sub = QLabel('Draw a box for a precise manual sample, or use Auto Capture after filling in the model metadata.')
        sub.setObjectName('muted')
        sub.setWordWrap(True)
        left_l.addWidget(sub)
        preview_actions = QHBoxLayout()
        self.refresh_live_btn = QPushButton('Refresh Live')
        self.refresh_live_btn.clicked.connect(self.resume_live_preview)
        preview_actions.addWidget(self.refresh_live_btn)
        preview_actions.addStretch(1)
        left_l.addLayout(preview_actions)
        self.preview = PreviewLabel()
        self.preview.selectionStarted.connect(self.freeze_preview)
        self.preview.selectionChanged.connect(self._on_selection)
        left_l.addWidget(self.preview, 1)
        self.selection_label = QLabel('Selection: none')
        self.selection_label.setObjectName('muted')
        left_l.addWidget(self.selection_label)
        layout.addWidget(left, 3)

        right = self._panel()
        right_wrap = QVBoxLayout(right)
        title = QLabel('Model Definition')
        title.setObjectName('sectionTitle')
        right_wrap.addWidget(title)
        note = QLabel('For now we keep model setup minimal: name, label, category, expected color, then save a clean sample and confirm it in the preview.')
        note.setObjectName('muted')
        note.setWordWrap(True)
        right_wrap.addWidget(note)
        self.wizard_status = QLabel('Guided capture is idle.')
        self.wizard_status.setObjectName('muted')
        self.wizard_status.setWordWrap(True)
        right_wrap.addWidget(self.wizard_status)
        self.selected_model_status = QLabel('Selected model: none')
        self.selected_model_status.setObjectName('muted')
        self.selected_model_status.setWordWrap(True)
        right_wrap.addWidget(self.selected_model_status)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet('background:transparent;')
        right_wrap.addWidget(scroll, 1)
        body = QWidget()
        scroll.setWidget(body)
        form = QVBoxLayout(body)
        form.setSpacing(10)
        self.fields = {}
        visible_fields = [('name', 'Model Name'), ('label', 'Label'), ('category', 'Category'), ('expected_color', 'Expected Color')]
        hidden_fields = ['brand', 'detect_as', 'aliases']
        for key, title in visible_fields:
            lbl = QLabel(title)
            lbl.setObjectName('fieldLabel')
            form.addWidget(lbl)
            entry = QLineEdit()
            form.addWidget(entry)
            self.fields[key] = entry
        for key in hidden_fields:
            self.fields[key] = QLineEdit()
            self.fields[key].hide()
        self.notes = QTextEdit()
        self.notes.hide()
        self.kind = QComboBox()
        self.kind.addItems(['object', 'phone'])
        self.kind.hide()
        primary_row = QHBoxLayout()
        self.new_btn = QPushButton('New Model')
        self.new_btn.clicked.connect(self.clear_form)
        primary_row.addWidget(self.new_btn)
        self.save_btn = QPushButton('Add Selection Sample')
        self.save_btn.setObjectName('primary')
        self.save_btn.clicked.connect(self.save_model_sample)
        primary_row.addWidget(self.save_btn)
        form.addLayout(primary_row)
        secondary_row = QHBoxLayout()
        self.auto_btn = QPushButton('Add Auto Sample')
        self.auto_btn.clicked.connect(self.auto_capture_sample)
        secondary_row.addWidget(self.auto_btn)
        self.wizard_btn = QPushButton('Guided Capture x4')
        self.wizard_btn.clicked.connect(self.start_guided_capture)
        secondary_row.addWidget(self.wizard_btn)
        self.delete_btn = QPushButton('Delete Selected')
        self.delete_btn.clicked.connect(self.delete_selected_model)
        secondary_row.addWidget(self.delete_btn)
        form.addLayout(secondary_row)
        confirm_title = QLabel('Saved Sample Confirmation')
        confirm_title.setObjectName('fieldLabel')
        form.addWidget(confirm_title)
        self.last_saved_preview = StaticImageLabel(320, 220)
        form.addWidget(self.last_saved_preview)
        self.last_saved_label = QLabel('No model sample has been saved yet.')
        self.last_saved_label.setObjectName('muted')
        self.last_saved_label.setWordWrap(True)
        form.addWidget(self.last_saved_label)
        gallery_title = QLabel('All Samples For Selected Model')
        gallery_title.setObjectName('fieldLabel')
        form.addWidget(gallery_title)
        self.sample_gallery = QListWidget()
        self.sample_gallery.setViewMode(QListWidget.IconMode)
        self.sample_gallery.setFlow(QListWidget.LeftToRight)
        self.sample_gallery.setWrapping(True)
        self.sample_gallery.setResizeMode(QListWidget.Adjust)
        self.sample_gallery.setMovement(QListWidget.Static)
        self.sample_gallery.setIconSize(QSize(100, 100))
        self.sample_gallery.setGridSize(QSize(118, 132))
        self.sample_gallery.setFixedHeight(180)
        form.addWidget(self.sample_gallery)
        lbl = QLabel('Saved Models')
        lbl.setObjectName('fieldLabel')
        form.addWidget(lbl)
        self.model_tree = QTreeWidget()
        self.model_tree.setHeaderLabels(['Label', 'State'])
        self.model_tree.itemSelectionChanged.connect(self.load_selected_model)
        form.addWidget(self.model_tree, 1)
        layout.addWidget(right, 2)
        for entry in self.fields.values():
            entry.textChanged.connect(self.update_action_states)

    def _build_events(self):
        layout = QVBoxLayout(self.events_tab)
        self.events_list = QListWidget()
        layout.addWidget(self.events_list)

    def _build_registry(self):
        layout = QVBoxLayout(self.registry_tab)
        title = QLabel('Model Registry')
        title.setObjectName('sectionTitle')
        layout.addWidget(title)
        note = QLabel('Click any row to load that model back into Model Studio.')
        note.setObjectName('muted')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.registry = QTableWidget(0, 8)
        self.registry.setHorizontalHeaderLabels(['Label', 'Category', 'Expected', 'Seen', 'Match', 'Detect', 'Samples', 'State'])
        self.registry.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.registry.setSelectionMode(QAbstractItemView.SingleSelection)
        self.registry.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.registry.setAlternatingRowColors(True)
        self.registry.verticalHeader().setVisible(False)
        self.registry.horizontalHeader().setStretchLastSection(True)
        self.registry.itemSelectionChanged.connect(self.load_selected_registry_model)
        layout.addWidget(self.registry)

    def _on_selection(self, roi):
        if roi:
            self.selection_label.setText(f'Selection: {roi} | preview frozen')
        else:
            self.selection_label.setText('Selection: none')
        self.update_action_states()

    def _queue(self, payload):
        self.state['commands'].put(payload)

    def _pause_refresh(self, seconds=0.6):
        import time
        self.pause_refresh_until = max(self.pause_refresh_until, time.time() + float(seconds))

    def freeze_preview(self):
        if self.preview_is_frozen or self.current_frame is None:
            return
        self.frozen_preview_frame = self.current_frame.copy()
        self.preview_is_frozen = True
        self.preview.set_frame(self.frozen_preview_frame)
        self.update_action_states()

    def resume_live_preview(self):
        self.preview_is_frozen = False
        self.frozen_preview_frame = None
        self.preview.selection = None
        self.preview.update()
        self.selection_label.setText('Selection: none')
        if self.current_frame is not None:
            self.preview.set_frame(self.current_frame)
        self.update_action_states()

    def _refresh_sample_gallery(self, profile_name, sample_count):
        signature = (profile_name or '', int(sample_count or 0))
        if signature == self.last_gallery_signature:
            return
        self.sample_gallery.clear()
        if not profile_name:
            self.sample_gallery.addItem(QListWidgetItem('No model selected'))
            self.last_gallery_signature = signature
            return
        folder = PROFILES_DIR / profile_name
        sample_paths = sorted(folder.glob('sample_*.png'))
        if not sample_paths:
            self.sample_gallery.addItem(QListWidgetItem('No saved samples yet'))
            self.last_gallery_signature = signature
            return
        for index, sample_path in enumerate(sample_paths, start=1):
            pixmap = QPixmap(str(sample_path))
            item = QListWidgetItem(f'Sample {index}')
            item.setIcon(QIcon(pixmap))
            item.setToolTip(str(sample_path.name))
            self.sample_gallery.addItem(item)
        self.last_gallery_signature = signature

    def _metadata(self):
        meta = {
            'name': self.fields['name'].text().strip(),
            'label': self.fields['label'].text().strip(),
            'kind': self.kind.currentText(),
            'category': self.fields['category'].text().strip() or 'object',
            'brand': self.fields['brand'].text().strip(),
            'expected_color': self.fields['expected_color'].text().strip().lower(),
            'detect_as': self.fields['detect_as'].text().strip(),
            'aliases': self.fields['aliases'].text().strip(),
            'notes': self.notes.toPlainText().strip(),
        }
        if not meta['detect_as']:
            meta['detect_as'] = build_detect_prompt(meta)
        return meta

    def validate_metadata(self, meta):
        if not looks_meaningful_text(meta['name']):
            return 'Model name must be meaningful, for example: purple_tube'
        if not looks_meaningful_text(meta['label']):
            return 'Label must be meaningful.'
        if not looks_meaningful_text(meta['category']):
            return 'Category must be meaningful, for example: tube, bottle, pen'
        if not looks_meaningful_text(meta['detect_as']):
            return 'At least one meaningful detect prompt is required.'
        return ''

    def clear_form(self):
        for field in self.fields.values():
            field.clear()
        self.notes.clear()
        self.kind.setCurrentText('object')
        self.resume_live_preview()
        self.model_tree.clearSelection()
        self.registry.clearSelection()
        self.selected_profile_name = ''
        self.selected_model_status.setText('Selected model: none')
        self._refresh_sample_gallery('', 0)
        self.stop_guided_capture(reset_label=True)
        self.update_action_states()

    def _has_meaningful_metadata(self):
        meta = self._metadata()
        return self.validate_metadata(meta) == ''

    def _selected_existing_model_name(self):
        if self.selected_profile_name:
            return self.selected_profile_name
        items = self.model_tree.selectedItems()
        if items:
            return items[0].data(0, Qt.UserRole)
        row = self.registry.currentRow()
        if row >= 0:
            cell = self.registry.item(row, 0)
            if cell is not None:
                return cell.data(Qt.UserRole)
        name = self.fields['name'].text().strip()
        profiles = self.state.get('profiles', {})
        return name if name in profiles else ''

    def update_action_states(self):
        has_frame = self.current_frame is not None
        has_roi = self.preview.selected_roi() is not None if self.preview.frame is not None else False
        has_metadata = self._has_meaningful_metadata()
        has_existing_model = bool(self._selected_existing_model_name())
        wizard_running = self.wizard_timer.isActive()

        self.new_btn.setEnabled(not wizard_running)
        self.save_btn.setEnabled((has_existing_model or has_metadata) and has_frame and has_roi and not wizard_running)
        self.auto_btn.setEnabled((has_existing_model or has_metadata) and has_frame and not wizard_running)
        self.wizard_btn.setEnabled((has_existing_model or has_metadata) and has_frame and not wizard_running)
        self.delete_btn.setEnabled(has_existing_model and not wizard_running)
        self.refresh_live_btn.setEnabled(self.preview_is_frozen)

    def save_model_sample(self):
        meta = self._metadata()
        roi = self.preview.selected_roi()
        error = self.validate_metadata(meta)
        if error:
            QMessageBox.warning(self, 'Model Definition', error)
            return
        if roi is None or self.current_frame is None:
            QMessageBox.warning(self, 'Selection Required', 'Draw a selection on the live preview before saving.')
            return
        source_frame = self.preview.frame if self.preview.frame is not None else self.current_frame
        self._pause_refresh(0.8)
        self._queue({'type': 'save_model_sample', 'frame': source_frame.copy(), 'rect': roi, 'metadata': meta})
        self.resume_live_preview()
        self.update_action_states()

    def auto_capture_sample(self):
        meta = self._metadata()
        error = self.validate_metadata(meta)
        if error:
            QMessageBox.warning(self, 'Model Definition', error)
            return
        if self.current_frame is None:
            QMessageBox.warning(self, 'Live Feed', 'Wait for the live camera preview before using Auto Capture.')
            return
        self._pause_refresh(0.8)
        self._queue({'type': 'auto_capture_sample', 'frame': self.current_frame.copy(), 'metadata': meta})
        self.update_action_states()

    def start_guided_capture(self):
        meta = self._metadata()
        error = self.validate_metadata(meta)
        if error:
            QMessageBox.warning(self, 'Model Definition', error)
            return
        if self.current_frame is None:
            QMessageBox.warning(self, 'Live Feed', 'Wait for the live camera preview before starting the guided capture wizard.')
            return
        dialog = GuidedCaptureDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        self.wizard_metadata = meta
        self.wizard_total = dialog.sample_count.value()
        self.wizard_remaining = self.wizard_total
        self.wizard_interval_ms = dialog.delay_ms.value()
        self.wizard_status.setText(
            f"Guided capture running for '{meta['label'] or meta['name']}'. Keep the object visible and slightly rotate it between shots."
        )
        self.update_action_states()
        self._wizard_step()
        if self.wizard_remaining > 0:
            self.wizard_timer.start(self.wizard_interval_ms)

    def stop_guided_capture(self, reset_label=False):
        self.wizard_timer.stop()
        self.wizard_total = 0
        self.wizard_remaining = 0
        self.wizard_metadata = None
        if reset_label:
            self.wizard_status.setText('Guided capture is idle.')
        self.update_action_states()

    def _wizard_step(self):
        if self.wizard_remaining <= 0 or self.wizard_metadata is None:
            self.stop_guided_capture(reset_label=True)
            QMessageBox.information(self, 'Guided Capture Wizard', 'Guided capture completed.')
            return
        if self.current_frame is None:
            self.wizard_status.setText('Guided capture paused: waiting for live camera preview.')
            return
        completed = self.wizard_total - self.wizard_remaining + 1
        self.wizard_status.setText(
            f"Capturing sample {completed}/{self.wizard_total} for '{self.wizard_metadata['label'] or self.wizard_metadata['name']}'. Slightly rotate or move the object now."
        )
        self._pause_refresh(0.6)
        self._queue({'type': 'auto_capture_sample', 'frame': self.current_frame.copy(), 'metadata': dict(self.wizard_metadata)})
        self.wizard_remaining -= 1
        if self.wizard_remaining <= 0:
            self.stop_guided_capture(reset_label=True)
            QMessageBox.information(self, 'Guided Capture Wizard', 'Guided capture completed.')

    def update_metadata_only(self):
        meta = self._metadata()
        error = self.validate_metadata(meta)
        if error:
            QMessageBox.warning(self, 'Model Definition', error)
            return
        self._queue({'type': 'update_model_metadata', 'metadata': meta})

    def _load_profile_into_form(self, item):
        if not item:
            return
        self.selected_profile_name = item.get('name', '')
        self.fields['name'].setText(item.get('name', ''))
        self.fields['label'].setText(item.get('label', ''))
        self.fields['category'].setText(item.get('category', ''))
        self.fields['brand'].setText(item.get('brand', ''))
        self.fields['expected_color'].setText(item.get('expected_color', ''))
        self.fields['detect_as'].setText(item.get('detect_as', ''))
        self.fields['aliases'].setText(item.get('aliases', ''))
        self.notes.setPlainText(item.get('notes', ''))
        self.kind.setCurrentText(item.get('kind', 'object'))
        self.selected_model_status.setText(
            f"Selected model: {item.get('label', item.get('name', ''))} | samples: {item.get('samples', 0)} | "
            f"To add more photos, keep this model selected and use 'Add Selection Sample', 'Add Auto Sample', or 'Guided Capture x4'."
        )

    def load_selected_model(self):
        try:
            items = self.model_tree.selectedItems()
            if not items:
                return
            name = items[0].data(0, Qt.UserRole)
            item = self.state.get('profiles', {}).get(name)
            self._load_profile_into_form(item)
            self.update_action_states()
        except Exception as exc:
            log_ui_exception('load_selected_model', exc)
            QMessageBox.critical(self, 'UI Error', 'A model selection error occurred. Details were written to data/events/ui_errors.log.')

    def load_selected_registry_model(self):
        try:
            row = self.registry.currentRow()
            if row < 0:
                return
            item = self.registry.item(row, 0)
            if item is None:
                return
            name = item.data(Qt.UserRole)
            profile = self.state.get('profiles', {}).get(name)
            self._load_profile_into_form(profile)
            for index in range(self.model_tree.topLevelItemCount()):
                tree_item = self.model_tree.topLevelItem(index)
                if tree_item.data(0, Qt.UserRole) == name:
                    self.model_tree.setCurrentItem(tree_item)
                    break
            self.update_action_states()
        except Exception as exc:
            log_ui_exception('load_selected_registry_model', exc)
            QMessageBox.critical(self, 'UI Error', 'A registry selection error occurred. Details were written to data/events/ui_errors.log.')

    def delete_selected_model(self):
        items = self.model_tree.selectedItems()
        name = self.fields['name'].text().strip()
        if items:
            name = items[0].data(0, Qt.UserRole)
        if not name:
            QMessageBox.warning(self, 'Select Model', 'Choose a model to delete.')
            return
        if QMessageBox.question(self, 'Delete Model', f"Delete model '{name}'?") == QMessageBox.Yes:
            self.selected_profile_name = ''
            self._queue({'type': 'delete_model', 'name': name})

    def closeEvent(self, event: QCloseEvent):
        self._queue({'type': 'shutdown'})
        super().closeEvent(event)

    def refresh(self):
        try:
            import time
            if time.time() < self.pause_refresh_until:
                return
            lock = self.state.get('_lock')
            if lock is None:
                snapshot = dict(self.state)
            else:
                with lock:
                    snapshot = {
                        'profiles': dict(self.state.get('profiles', {})),
                        'fps': self.state.get('fps', 0.0),
                        'frame_index': self.state.get('frame_index', 0),
                        'events': list(self.state.get('events', [])),
                        'detector_ready': self.state.get('detector_ready', False),
                        'detector_model': self.state.get('detector_model', 'Detector'),
                        'detector_error': self.state.get('detector_error', ''),
                        'self_registered': self.state.get('self_registered', False),
                        'self_present': self.state.get('self_present', False),
                        'self_score': self.state.get('self_score', 0.0),
                        'open_vocab_prompts': list(self.state.get('open_vocab_prompts', [])),
                        'preview_frame': self.state.get('preview_frame').copy() if self.state.get('preview_frame') is not None else None,
                        'last_model_capture': self.state.get('last_model_capture').copy() if self.state.get('last_model_capture') is not None else None,
                        'last_model_label': self.state.get('last_model_label', ''),
                        'last_model_source': self.state.get('last_model_source', ''),
                    }

            profiles = list(snapshot.get('profiles', {}).values())
            self.metric_widgets['fps'].setText(f"{snapshot.get('fps', 0.0):.1f}")
            self.metric_widgets['frames'].setText(str(snapshot.get('frame_index', 0)))
            self.metric_widgets['models'].setText(str(len(profiles)))
            self.metric_widgets['events'].setText(str(len(snapshot.get('events', []))))
            detector_ready = snapshot.get('detector_ready', False)
            self.detector_title.setText(snapshot.get('detector_model', 'Detector'))
            self.detector_status.setText(snapshot.get('detector_error') or ('Detector online and processing live prompts.' if detector_ready else 'Detector is initializing.'))
            self.self_status.setText('Identity ready' if snapshot.get('self_registered') else 'Identity pending')
            self.self_detail.setText(f"Presence: {'Yes' if snapshot.get('self_present') else 'No'} | Score: {snapshot.get('self_score', 0.0):.2f}")
            prompts = snapshot.get('open_vocab_prompts', [])
            self.prompt_label.setText('Prompts: ' + (', '.join(prompts[:4]) + (' ...' if len(prompts) > 4 else '') if prompts else '-'))

            live_matches = sum(1 for item in profiles if item.get('visible'))
            self.ov_cards[0][0].setText('Online' if detector_ready else 'Standby')
            self.ov_cards[0][1].setText(snapshot.get('detector_error') or ('The detector is actively evaluating open-vocabulary prompts.' if detector_ready else 'The detector is still preparing its runtime.'))
            self.ov_cards[1][0].setText(str(live_matches))
            self.ov_cards[1][1].setText('Verified objects currently visible in the live scene.')
            self.ov_cards[2][0].setText(str(len(prompts)))
            self.ov_cards[2][1].setText('Dynamic prompts generated from model metadata and aliases.')

            preview_frame_index = snapshot.get('frame_index', 0)
            self.current_frame = snapshot.get('preview_frame')
            if preview_frame_index != self.last_preview_frame_index:
                if not self.preview_is_frozen:
                    self.preview.set_frame(self.current_frame)
                self.overview_preview.set_frame(self.current_frame)
                self.last_preview_frame_index = preview_frame_index
            saved_signature = (snapshot.get('last_model_label', ''), snapshot.get('last_model_source', ''))
            if saved_signature != self.last_saved_signature:
                self.last_saved_preview.set_frame(snapshot.get('last_model_capture'))
                self.last_saved_signature = saved_signature
            last_label = snapshot.get('last_model_label', '')
            last_source = snapshot.get('last_model_source', '')
            if last_label:
                self.last_saved_label.setText(f"Latest saved sample: {last_label} via {last_source}. Confirm the crop looks correct before continuing.")
            else:
                self.last_saved_label.setText('No model sample has been saved yet.')

            self.overview_matches.clear()
            visible_profiles = [item for item in profiles if item.get('visible')]
            if visible_profiles:
                for item in visible_profiles:
                    msg = (
                        f"{item.get('label', item['name'])} | "
                        f"match {item.get('confidence', 0.0):.2f} | "
                        f"detect {item.get('detector_confidence', 0.0):.2f} | "
                        f"prompt {item.get('prompt_used') or '-'}"
                    )
                    self.overview_matches.addItem(QListWidgetItem(msg))
            else:
                self.overview_matches.addItem(QListWidgetItem('No registered model is currently verified in the scene.'))

            selected_name = self.selected_profile_name or self.fields['name'].text().strip()
            profiles_signature = tuple(
                (item.get('name', ''), item.get('label', ''), item.get('state', ''), item.get('samples', 0))
                for item in profiles
            )
            if profiles_signature != self.last_profiles_signature:
                with QSignalBlocker(self.model_tree):
                    self.model_tree.clear()
                    for item in profiles:
                        node = QTreeWidgetItem([item.get('label', item['name']), item.get('state', '')])
                        node.setData(0, Qt.UserRole, item['name'])
                        self.model_tree.addTopLevelItem(node)
                        if item['name'] == selected_name:
                            self.model_tree.setCurrentItem(node)
                with QSignalBlocker(self.registry):
                    self.registry.setRowCount(len(profiles))
                    selected_row = -1
                    for row, item in enumerate(profiles):
                        values = [
                            item.get('label', item['name']),
                            item.get('category', ''),
                            item.get('expected_color', ''),
                            item.get('color', ''),
                            str(item.get('confidence', '')),
                            str(item.get('detector_confidence', '')),
                            str(item.get('samples', '')),
                            item.get('state', ''),
                        ]
                        for col, value in enumerate(values):
                            table_item = QTableWidgetItem(value)
                            if col == 0:
                                table_item.setData(Qt.UserRole, item['name'])
                            self.registry.setItem(row, col, table_item)
                        if item['name'] == selected_name:
                            selected_row = row
                    if selected_row >= 0:
                        self.registry.selectRow(selected_row)
                self.last_profiles_signature = profiles_signature

            current_item = snapshot.get('profiles', {}).get(selected_name) if selected_name else None
            if current_item:
                self.selected_model_status.setText(
                    f"Selected model: {current_item.get('label', selected_name)} | samples: {current_item.get('samples', 0)} | "
                    f"To add more photos, keep this model selected and use 'Add Selection Sample', 'Add Auto Sample', or 'Guided Capture x4'."
                )
                self._refresh_sample_gallery(selected_name, current_item.get('samples', 0))
            else:
                self._refresh_sample_gallery('', 0)

            filtered_events = [it for it in snapshot.get('events', []) if it.get('type') != 'BLINK']
            events_signature = tuple((it.get('ts'), it.get('type'), it.get('message')) for it in filtered_events[-120:])
            if events_signature != self.last_events_signature:
                self.events_list.clear()
                for it in reversed(filtered_events[-120:]):
                    self.events_list.addItem(QListWidgetItem(f"[{time_fmt(it['ts'])}] {it['type']}: {it['message']}"))
                self.last_events_signature = events_signature

            self.update_action_states()
        except Exception as exc:
            log_ui_exception('refresh', exc)


def time_fmt(ts):
    import time
    return time.strftime('%H:%M:%S', time.localtime(ts))


def run_dashboard(state):
    global APP_QT
    APP_QT = QApplication.instance() or QApplication([])
    window = DesktopDashboard(state)
    window.show()
    APP_QT.exec()
