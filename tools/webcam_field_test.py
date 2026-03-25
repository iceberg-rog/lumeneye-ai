import argparse
import time
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Config
from vision.detector import OpenVocabularyDetector
from vision.local_object import LocalTeachEngine


class SilentBus:
    def emit(self, *args, **kwargs):
        return None


def parse_args():
    parser = argparse.ArgumentParser(description='Run a live webcam field test for a registered model.')
    parser.add_argument('--model', required=True, help='Registered model name to evaluate, for example blue_bic_pen')
    parser.add_argument('--seconds', type=int, default=20, help='Duration of the field test window')
    parser.add_argument('--camera', type=int, default=Config.CAMERA_INDEX, help='Camera index')
    parser.add_argument('--show-window', action='store_true', help='Show annotated live preview during the test')
    return parser.parse_args()


def draw_overlay(frame, profile, elapsed, remaining, matched_frames, total_frames):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (780, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.48, frame, 0.52, 0, frame)
    lines = [
        f'Field Test: {profile.metadata.get("label", profile.name)}',
        f'Elapsed: {elapsed:.1f}s  Remaining: {max(0.0, remaining):.1f}s',
        f'Frames: {total_frames}  Matched: {matched_frames}',
        f'Match rate: {(matched_frames / max(1, total_frames)) * 100:.1f}%',
        f'Visible: {profile.visible}  Match score: {profile.match_score:.2f}  Detector: {profile.detector_score:.2f}',
        f'Prompt: {profile.prompt_used or "-"}  Seen color: {profile.dominant_color_name}',
    ]
    for idx, text in enumerate(lines):
        cv2.putText(frame, text, (22, 36 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (235, 242, 255), 2, cv2.LINE_AA)
    if profile.visible:
        x, y, w, h = map(int, profile.rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (77, 208, 168), 3)


def main():
    args = parse_args()
    engine = LocalTeachEngine()
    profile = engine.get_profile(args.model)
    if profile is None:
        print(f"[FIELD TEST] model '{args.model}' was not found.")
        print('[FIELD TEST] register it first with app.py and key T.')
        return 1

    detector = OpenVocabularyDetector()
    prompts, prompt_to_profiles = engine.build_detection_prompts()
    if not prompts:
        print('[FIELD TEST] no prompts available for this model set.')
        return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'[FIELD TEST] camera {args.camera} could not be opened.')
        return 1

    window_name = f'FIELD TEST :: {args.model}'
    if args.show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    bus = SilentBus()
    start = time.time()
    total_frames = 0
    matched_frames = 0
    max_match = 0.0
    max_detector = 0.0
    seen_prompts = set()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total_frames += 1
            detections = detector.detect(frame, prompts)
            engine.update(frame, detections, prompt_to_profiles, bus)
            profile = engine.get_profile(args.model)

            if profile.visible:
                matched_frames += 1
                max_match = max(max_match, profile.match_score)
                max_detector = max(max_detector, profile.detector_score)
                if profile.prompt_used:
                    seen_prompts.add(profile.prompt_used)

            elapsed = time.time() - start
            remaining = args.seconds - elapsed
            if args.show_window:
                display = frame.copy()
                draw_overlay(display, profile, elapsed, remaining, matched_frames, total_frames)
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
            if elapsed >= args.seconds:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    match_rate = (matched_frames / max(1, total_frames)) * 100.0
    print(f"[FIELD TEST] model={args.model}")
    print(f"[FIELD TEST] frames={total_frames}")
    print(f"[FIELD TEST] matched_frames={matched_frames}")
    print(f"[FIELD TEST] match_rate={match_rate:.1f}%")
    print(f"[FIELD TEST] max_match_score={max_match:.2f}")
    print(f"[FIELD TEST] max_detector_score={max_detector:.2f}")
    print(f"[FIELD TEST] prompts_seen={', '.join(sorted(seen_prompts)) or '-'}")
    print(f"[FIELD TEST] detector_ready={detector.ready}")
    print(f"[FIELD TEST] detector_error={detector.error or '-'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
