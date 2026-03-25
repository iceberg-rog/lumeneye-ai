import queue
import sys
import threading
import time
import traceback
import faulthandler
from pathlib import Path

import cv2

from core.config import Config
from core.events import EventBus
from core.rules import RuleEngine
from desktop.dashboard import run_dashboard
from vision.detector import OpenVocabularyDetector
from vision.face import FaceModule
from vision.local_object import LocalTeachEngine, build_detect_prompt, build_metadata_prompts

STATE = {
    'fps': 0.0,
    'frame_index': 0,
    'start_time': time.time(),
    'active_profile': None,
    'profiles': {},
    'events': [],
    'blink_total': 0,
    'self_registered': False,
    'self_present': False,
    'self_score': 0.0,
    'detector_ready': False,
    'detector_error': '',
    'detector_model': Config.DETECTOR_MODEL,
    'open_vocab_prompts': [],
    'missing_event_repeat_sec': Config.MISSING_EVENT_REPEAT_SEC,
    'preview_frame': None,
    'last_model_capture': None,
    'last_model_label': '',
    'last_model_source': '',
    'commands': queue.Queue(),
    'running': True,
}

BUS = EventBus()
STATE_LOCK = threading.Lock()
STATE['_lock'] = STATE_LOCK
CRASH_LOG = Path(__file__).resolve().parent / 'data' / 'events' / 'runtime_crash.log'


def _append_runtime_log(title, details):
    CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CRASH_LOG, 'a', encoding='utf-8') as handle:
        handle.write(f'[{title}] {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        handle.write(details)
        handle.write('\n\n')


def _install_runtime_logging():
    CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
    crash_handle = open(CRASH_LOG, 'a', encoding='utf-8')
    faulthandler.enable(crash_handle)

    def _log_excepthook(exc_type, exc_value, exc_tb):
        _append_runtime_log('sys.excepthook', ''.join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    def _log_thread_exception(args):
        _append_runtime_log(
            f'threading.excepthook:{getattr(args.thread, "name", "thread")}',
            ''.join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
        )

    sys.excepthook = _log_excepthook
    threading.excepthook = _log_thread_exception


def snapshot_profiles(profiles):
    snapshot = {}
    for name, p in profiles.items():
        center = p.center if p.center is not None else (0, 0)
        snapshot[name] = {
            'name': name,
            'label': p.metadata.get('label', name),
            'kind': p.kind,
            'category': p.metadata.get('category', ''),
            'brand': p.metadata.get('brand', ''),
            'expected_color': p.metadata.get('expected_color', ''),
            'detect_as': p.metadata.get('detect_as', ''),
            'aliases': p.metadata.get('aliases', ''),
            'notes': p.metadata.get('notes', ''),
            'visible': p.visible,
            'center': f'{int(center[0])},{int(center[1])}',
            'color': p.dominant_color_name,
            'motion': round(float(p.motion_px), 1),
            'confidence': round(float(p.match_score), 2),
            'detector_confidence': round(float(p.detector_score), 2),
            'prompt_used': p.prompt_used,
            'samples': len(p.samples),
            'state': p.phone_state if p.kind == 'phone' else ('matched' if p.visible else 'unknown'),
        }
    return snapshot


def _detection_area(rect):
    return max(0, int(rect[2])) * max(0, int(rect[3]))


def process_commands(face, teach, detector, live_frame):
    while True:
        try:
            cmd = STATE['commands'].get_nowait()
        except queue.Empty:
            break
        kind = cmd.get('type')
        if kind == 'shutdown':
            with STATE_LOCK:
                STATE['running'] = False
        elif kind == 'register_self':
            if face.register_self():
                with STATE_LOCK:
                    STATE['self_registered'] = True
                BUS.emit('SELF', 'Self face registered', beep=True, cooldown_key='self-registered')
        elif kind == 'save_model_sample':
            meta = cmd['metadata']
            if not meta.get('detect_as'):
                meta['detect_as'] = build_detect_prompt(meta)
            if teach.get_profile(meta['name']) is not None:
                teach.update_profile_metadata(meta['name'], meta['kind'], meta)
            ok, created = teach.add_profile(meta['name'], meta['kind'], cmd['frame'], cmd['rect'], meta)
            if ok:
                teach.active_name = meta['name']
                with STATE_LOCK:
                    STATE['active_profile'] = teach.active_name
                    STATE['last_model_capture'] = teach.get_profile(meta['name']).samples[-1].image.copy()
                    STATE['last_model_label'] = meta['label'] or meta['name']
                    STATE['last_model_source'] = 'manual selection'
                action = 'Created' if created else 'Updated'
                sample_count = len(teach.get_profile(meta['name']).samples)
                BUS.emit('PROFILE', f"{action} model '{meta['label'] or meta['name']}' sample #{sample_count}", beep=True, cooldown_key=f'profile-add:{meta["name"]}')
            else:
                BUS.emit('PROFILE', f"Could not save sample for '{meta['label'] or meta['name']}'", beep=True, cooldown_key=f'profile-add-failed:{meta["name"]}')
        elif kind == 'auto_capture_sample':
            meta = dict(cmd['metadata'])
            if not meta.get('detect_as'):
                meta['detect_as'] = build_detect_prompt(meta)
            capture_frame = cmd.get('frame')
            if capture_frame is None:
                capture_frame = live_frame
            if capture_frame is None:
                BUS.emit('PROFILE', 'Auto capture failed: no live frame available yet.', beep=True, cooldown_key='auto-capture:no-frame')
                continue
            prompts = build_metadata_prompts(meta)
            if not prompts:
                BUS.emit('PROFILE', f"Auto capture failed for '{meta['label'] or meta['name']}': add a meaningful category or prompt.", beep=True, cooldown_key=f'auto-capture:no-prompt:{meta["name"]}')
                continue
            detections = detector.detect(capture_frame, prompts)
            if not detections:
                BUS.emit('PROFILE', f"Auto capture could not find '{meta['label'] or meta['name']}' in the current frame.", beep=True, cooldown_key=f'auto-capture:not-found:{meta["name"]}')
                continue
            best_detection = max(detections, key=lambda item: (float(item.get('confidence', 0.0)), _detection_area(item.get('rect', (0, 0, 0, 0)))))
            rect = best_detection['rect']
            if teach.get_profile(meta['name']) is not None:
                teach.update_profile_metadata(meta['name'], meta['kind'], meta)
            ok, created = teach.add_profile(meta['name'], meta['kind'], capture_frame, rect, meta)
            if ok:
                teach.active_name = meta['name']
                with STATE_LOCK:
                    STATE['active_profile'] = teach.active_name
                    STATE['last_model_capture'] = teach.get_profile(meta['name']).samples[-1].image.copy()
                    STATE['last_model_label'] = meta['label'] or meta['name']
                    STATE['last_model_source'] = 'auto capture'
                action = 'Created' if created else 'Updated'
                sample_count = len(teach.get_profile(meta['name']).samples)
                BUS.emit(
                    'PROFILE',
                    f"{action} model '{meta['label'] or meta['name']}' via auto capture sample #{sample_count} ({best_detection['prompt']} {best_detection['confidence']:.2f})",
                    beep=True,
                    cooldown_key=f'auto-capture:ok:{meta["name"]}',
                )
            else:
                BUS.emit('PROFILE', f"Auto capture crop failed for '{meta['label'] or meta['name']}'", beep=True, cooldown_key=f'auto-capture:crop:{meta["name"]}')
        elif kind == 'update_model_metadata':
            meta = cmd['metadata']
            if not meta.get('detect_as'):
                meta['detect_as'] = build_detect_prompt(meta)
            if teach.update_profile_metadata(meta['name'], meta['kind'], meta):
                teach.active_name = meta['name']
                with STATE_LOCK:
                    STATE['active_profile'] = teach.active_name
                BUS.emit('PROFILE', f"Updated model metadata '{meta['label'] or meta['name']}'", cooldown_key=f'profile-update:{meta["name"]}')
        elif kind == 'delete_model':
            name = cmd['name']
            if teach.delete_profile(name):
                BUS.emit('PROFILE', f"Removed model '{name}'", beep=True, cooldown_key=f'remove:{name}')


def vision_loop():
    try:
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        if not cap.isOpened():
            print('camera open failed')
            with STATE_LOCK:
                STATE['running'] = False
            return

        face = FaceModule()
        detector = OpenVocabularyDetector()
        teach = LocalTeachEngine()
        rules = RuleEngine()
        prev_t = time.time()
        last_detections = []
        last_prompt_to_profiles = {}
        last_prompt_signature = ()

        try:
            while True:
                with STATE_LOCK:
                    if not STATE['running']:
                        break
                ok, frame = cap.read()
                if not ok:
                    break
                with STATE_LOCK:
                    STATE['frame_index'] += 1
                    frame_index = STATE['frame_index']
                process_commands(face, teach, detector, frame)

                faces = face.process(frame)
                if face.blink_total != STATE['blink_total']:
                    with STATE_LOCK:
                        STATE['blink_total'] = face.blink_total
                        self_registered = STATE.get('self_registered')
                    if self_registered:
                        BUS.emit('BLINK', f"Blink {STATE['blink_total']}", cooldown_key='blink')
                self_present = any(f['is_self'] for f in faces)
                self_score = max([f['score'] for f in faces], default=0.0)
                with STATE_LOCK:
                    STATE['self_present'] = self_present
                    STATE['self_score'] = self_score

                prompts, prompt_to_profiles = teach.build_detection_prompts()
                prompt_signature = tuple(prompts)
                should_detect = bool(prompts) and (
                    prompt_signature != last_prompt_signature
                    or (frame_index % max(1, Config.DETECTOR_FRAME_STRIDE) == 0)
                )
                if should_detect:
                    last_detections = detector.detect(frame, prompts)
                    last_prompt_signature = prompt_signature
                    last_prompt_to_profiles = prompt_to_profiles
                elif not prompts:
                    last_detections = []
                    last_prompt_to_profiles = {}
                    last_prompt_signature = ()

                teach.update(frame, last_detections, last_prompt_to_profiles, BUS)
                profile_snapshot = snapshot_profiles(teach.profiles)
                with STATE_LOCK:
                    STATE['open_vocab_prompts'] = list(prompts)
                    STATE['detector_ready'] = detector.ready
                    STATE['detector_error'] = detector.error
                    STATE['profiles'] = profile_snapshot
                    STATE['active_profile'] = teach.active_name

                rules.process(STATE, teach.profiles, BUS)
                events_snapshot = BUS.list()

                display = frame.copy()
                with STATE_LOCK:
                    show_self = STATE.get('self_registered')
                if show_self:
                    for f in faces:
                        if not f['is_self']:
                            continue
                        x, y, w, h = f['rect']
                        color = (0, 255, 255)
                        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                        label = f"SELF {f['score']:.2f}"
                        cv2.putText(display, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                for name, p in teach.profiles.items():
                    if not p.visible:
                        continue
                    x, y, w, h = map(int, p.rect)
                    color = (0, 200, 255) if name == teach.active_name else (0, 255, 120)
                    cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(display, p.metadata.get('label', name), (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)

                now = time.time()
                fps = 1.0 / max(1e-6, now - prev_t)
                prev_t = now
                with STATE_LOCK:
                    STATE['events'] = events_snapshot
                    STATE['preview_frame'] = display
                    STATE['fps'] = fps
        finally:
            cap.release()
    except Exception:
        _append_runtime_log('vision_loop', traceback.format_exc())
        with STATE_LOCK:
            STATE['running'] = False
        raise


def main():
    _install_runtime_logging()
    worker = threading.Thread(target=vision_loop, daemon=True)
    worker.start()
    run_dashboard(STATE)
    with STATE_LOCK:
        STATE['running'] = False
    worker.join(timeout=2.0)


if __name__ == '__main__':
    main()
