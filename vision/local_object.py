import json
import cv2
import numpy as np

from core.config import Config, PROFILES_DIR
from core.utils import crop_rect, center_of, mean_color_bgr, color_name, mse_change, distance, clip_rect, object_embedding, cosine_similarity


def normalize_color_tag(value):
    if not value:
        return ''
    value = value.strip().lower()
    aliases = {
        'dark': 'black/dark',
        'black': 'black/dark',
        'light': 'white/light',
        'white': 'white/light',
        'purple': 'purple',
        'violet': 'purple',
    }
    return aliases.get(value, value)


def split_csv_text(value):
    if not value:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = str(value).split(',')
    return [item.strip() for item in items if item and item.strip()]


def looks_meaningful_text(value):
    value = (value or '').strip()
    return len(value) >= 2 and any(ch.isalpha() for ch in value)


def build_detect_prompt(metadata):
    detect_as = (metadata.get('detect_as') or '').strip()
    if detect_as:
        return detect_as
    prompts = []
    color = (metadata.get('expected_color') or '').strip()
    category = (metadata.get('category') or '').strip()
    brand = (metadata.get('brand') or '').strip()
    label = (metadata.get('label') or '').strip()
    if color and category:
        prompts.append(f'{color} {category}')
    if brand and category:
        prompts.append(f'{brand} {category}')
    if category:
        prompts.append(category)
    if label:
        prompts.append(label)
    prompts = [p for p in prompts if looks_meaningful_text(p)]
    return ', '.join(dict.fromkeys(prompts))


def build_metadata_prompts(metadata):
    prompts = []
    prompts.extend(split_csv_text(metadata.get('detect_as', '')))
    prompts.extend(split_csv_text(metadata.get('aliases', '')))
    category = (metadata.get('category') or '').strip()
    expected_color = (metadata.get('expected_color') or '').strip()
    label = (metadata.get('label') or '').strip()
    brand = (metadata.get('brand') or '').strip()
    if expected_color and category:
        prompts.append(f'{expected_color} {category}')
    if brand and category:
        prompts.append(f'{brand} {category}')
    if category:
        prompts.append(category)
    if label:
        prompts.append(label)
    return list(dict.fromkeys([p for p in prompts if looks_meaningful_text(p)]))


class ModelSample:
    def __init__(self, image, orb):
        self.image = image.copy()
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.mean_color = mean_color_bgr(self.image)
        self.dominant_color_name = color_name(self.image)
        self.identity_embedding = object_embedding(self.image)
        self.kp, self.des = orb.detectAndCompute(self.gray, None)


class LocalProfile:
    def __init__(self, name, kind, metadata):
        self.name = name
        self.kind = kind
        self.metadata = metadata
        self.samples = []
        self.reset_runtime()

    def reset_runtime(self):
        self.rect = (0, 0, 0, 0)
        self.visible = False
        self.ever_visible = False
        self.center = (0.0, 0.0)
        self.prev_center = self.center
        self.motion_px = 0.0
        self.frame_count = 0
        self.lost_frames = 0
        self.last_crop = None
        self.prev_crop = None
        self.visual_delta = 0.0
        self.changed = False
        self.mean_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.color_delta = 0.0
        self.dominant_color_name = 'unknown'
        self.phone_state = 'unknown'
        self.phone_on_since = None
        self.phone_last_screen = None
        self.match_score = 0.0
        self.detector_score = 0.0
        self.orb_score = 0.0
        self.prompt_used = ''

    @property
    def expected_color(self):
        return normalize_color_tag(self.metadata.get('expected_color', ''))

    @property
    def label(self):
        return self.metadata.get('label', self.name)

    def detection_prompts(self):
        return build_metadata_prompts(self.metadata)


class LocalTeachEngine:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=Config.ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.profiles = {}
        self.active_name = None
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        self._load_profiles()

    def get_profile(self, name):
        return self.profiles.get(name)

    def update_profile_metadata(self, name, kind, metadata):
        profile = self.profiles.get(name)
        if profile is None:
            return False
        profile.kind = kind
        profile.metadata.update(metadata)
        self._save_profile(profile)
        self.active_name = name
        return True

    def add_profile(self, name, kind, frame, rect, metadata):
        crop = crop_rect(frame, rect)
        if crop is None or crop.size == 0:
            return False, False
        profile = self.profiles.get(name)
        created = False
        if profile is None:
            profile = LocalProfile(name, kind, metadata)
            self.profiles[name] = profile
            created = True
        else:
            profile.kind = kind
            profile.metadata.update(metadata)
        profile.samples.append(ModelSample(crop, self.orb))
        if len(profile.samples) > Config.MAX_SAMPLES_PER_PROFILE:
            profile.samples = profile.samples[-Config.MAX_SAMPLES_PER_PROFILE:]
        self.active_name = name
        self._save_profile(profile)
        return True, created

    def cycle_active(self):
        if not self.profiles:
            self.active_name = None
            return
        names = list(self.profiles.keys())
        if self.active_name not in names:
            self.active_name = names[0]
            return
        idx = (names.index(self.active_name) + 1) % len(names)
        self.active_name = names[idx]

    def delete_profile(self, name):
        if name not in self.profiles:
            return False
        del self.profiles[name]
        folder = PROFILES_DIR / name
        if folder.exists():
            for item in folder.iterdir():
                item.unlink()
            folder.rmdir()
        self.active_name = next(iter(self.profiles), None)
        return True

    def delete_active(self):
        if self.active_name and self.active_name in self.profiles:
            self.delete_profile(self.active_name)

    def build_detection_prompts(self):
        prompt_to_profiles = {}
        for name, profile in self.profiles.items():
            for prompt in profile.detection_prompts():
                prompt_to_profiles.setdefault(prompt, set()).add(name)
        prompts = list(prompt_to_profiles.keys())
        return prompts, prompt_to_profiles

    def _profile_metadata_path(self, name):
        return PROFILES_DIR / name / 'profile.json'

    def _save_profile(self, profile):
        folder = PROFILES_DIR / profile.name
        folder.mkdir(parents=True, exist_ok=True)
        for item in folder.glob('sample_*.png'):
            item.unlink()
        for idx, sample in enumerate(profile.samples, start=1):
            cv2.imwrite(str(folder / f'sample_{idx:02d}.png'), sample.image)
        payload = {
            'name': profile.name,
            'kind': profile.kind,
            'metadata': profile.metadata,
            'samples': len(profile.samples),
        }
        with open(self._profile_metadata_path(profile.name), 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load_profiles(self):
        for folder in PROFILES_DIR.iterdir():
            if not folder.is_dir():
                continue
            meta_path = folder / 'profile.json'
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            profile = LocalProfile(payload.get('name', folder.name), payload.get('kind', 'object'), payload.get('metadata', {}))
            for sample_path in sorted(folder.glob('sample_*.png')):
                image = cv2.imread(str(sample_path))
                if image is None or image.size == 0:
                    continue
                profile.samples.append(ModelSample(image, self.orb))
            if profile.samples:
                self.profiles[profile.name] = profile
        self.active_name = next(iter(self.profiles), None)

    def _score_orb(self, sample, candidate_gray):
        kp, des = self.orb.detectAndCompute(candidate_gray, None)
        if sample.des is None or des is None or len(sample.kp) < 4 or len(kp) < 4:
            return 0.0
        matches = self.matcher.match(sample.des, des)
        if not matches:
            return 0.0
        good = [m for m in matches if m.distance <= 48]
        denom = max(8, min(len(sample.kp), len(kp)))
        return min(1.0, len(good) / float(denom))

    def _prepare_candidate_crop(self, crop):
        if crop is None or crop.size == 0:
            return crop
        h, w = crop.shape[:2]
        short_side = min(h, w)
        if short_side >= 160:
            return crop
        scale = min(2.0, 160.0 / max(1.0, short_side))
        resized = cv2.resize(crop, (max(2, int(w * scale)), max(2, int(h * scale))), interpolation=cv2.INTER_CUBIC)
        return resized

    def _score_crop(self, profile, crop, detector_score):
        prepared_crop = self._prepare_candidate_crop(crop)
        candidate_gray = cv2.cvtColor(prepared_crop, cv2.COLOR_BGR2GRAY)
        candidate_embedding = object_embedding(prepared_crop)
        detected_color = normalize_color_tag(color_name(prepared_crop))
        sample_count = max(1, len(profile.samples))
        sample_metrics = []
        for sample in profile.samples:
            orb_score = self._score_orb(sample, candidate_gray)
            appearance_score = 1.0 - min(1.0, mse_change(sample.image, prepared_crop) / Config.APPEARANCE_MSE_LIMIT)
            identity_score = max(0.0, cosine_similarity(sample.identity_embedding, candidate_embedding))
            color_distance = float(np.linalg.norm(mean_color_bgr(prepared_crop) - sample.mean_color))
            color_score = 1.0 - min(1.0, color_distance / Config.COLOR_DISTANCE_LIMIT)
            if profile.expected_color and detected_color != profile.expected_color:
                color_score *= 0.1
            detector_weight = 0.34
            orb_weight = 0.16 if sample_count >= 6 else 0.22
            identity_weight = 0.28
            appearance_weight = 0.14
            color_weight = 1.0 - detector_weight - orb_weight - identity_weight - appearance_weight
            score = (
                (detector_weight * detector_score)
                + (orb_weight * orb_score)
                + (identity_weight * identity_score)
                + (appearance_weight * appearance_score)
                + (color_weight * color_score)
            )
            metrics = {
                'score': score,
                'orb_score': orb_score,
                'identity_score': identity_score,
                'appearance_score': appearance_score,
                'color_score': color_score,
                'detected_color': detected_color or 'unknown',
            }
            sample_metrics.append(metrics)

        if not sample_metrics:
            return None

        top_metrics = sorted(sample_metrics, key=lambda item: item['score'], reverse=True)[:min(3, len(sample_metrics))]
        support_count = sum(1 for item in sample_metrics if item['identity_score'] >= Config.ADAPTIVE_MULTI_SAMPLE_IDENTITY_SCORE)
        combined = {
            'score': float(np.mean([item['score'] for item in top_metrics])),
            'orb_score': float(np.mean([item['orb_score'] for item in top_metrics])),
            'identity_score': float(np.mean([item['identity_score'] for item in top_metrics])),
            'appearance_score': float(np.mean([item['appearance_score'] for item in top_metrics])),
            'color_score': float(np.mean([item['color_score'] for item in top_metrics])),
            'detected_color': top_metrics[0]['detected_color'],
            'support_count': support_count,
        }
        return combined

    def _candidate_profiles(self, detection, prompt_to_profiles):
        names = set(prompt_to_profiles.get(detection['prompt'], set()))
        if names:
            return [self.profiles[name] for name in names if name in self.profiles]
        return list(self.profiles.values())

    def _apply_detection(self, profile, detection, metrics, frame, bus):
        rect = clip_rect(detection['rect'], frame.shape[1], frame.shape[0])
        profile.rect = rect
        profile.visible = True
        profile.ever_visible = True
        profile.lost_frames = 0
        profile.prev_center = profile.center
        profile.center = center_of(rect)
        profile.motion_px = distance(profile.center, profile.prev_center)
        profile.prev_crop = profile.last_crop.copy() if profile.last_crop is not None else None
        profile.last_crop = crop_rect(frame, rect)
        profile.visual_delta = mse_change(profile.prev_crop, profile.last_crop)
        profile.changed = profile.visual_delta >= Config.VISUAL_CHANGE_THRESHOLD
        prev_color = profile.mean_color.copy()
        profile.mean_color = mean_color_bgr(profile.last_crop)
        profile.color_delta = float(np.linalg.norm(profile.mean_color - prev_color))
        profile.dominant_color_name = metrics['detected_color']
        profile.match_score = metrics['score']
        profile.detector_score = detection['confidence']
        profile.orb_score = metrics['orb_score']
        profile.prompt_used = detection['prompt']
        if profile.motion_px >= Config.POSITION_CHANGE_PX:
            bus.emit('MOVE', f'{profile.name} moved ({profile.motion_px:.1f}px)', cooldown_key=f'move:{profile.name}')
        if profile.changed:
            bus.emit('CHANGE', f'{profile.name} visual change ({profile.visual_delta:.1f})', cooldown_key=f'change:{profile.name}')
        if profile.color_delta >= Config.COLOR_CHANGE_THRESHOLD:
            bus.emit('COLOR', f'{profile.name} color changed -> {profile.dominant_color_name}', cooldown_key=f'color:{profile.name}')
        if profile.kind == 'phone':
            self._analyze_phone(profile, bus)

    def _analyze_phone(self, profile, bus):
        crop = profile.last_crop
        if crop is None or crop.size == 0:
            return
        h, w = crop.shape[:2]
        sx1, sy1, sx2, sy2 = int(w * 0.18), int(h * 0.12), int(w * 0.82), int(h * 0.88)
        screen = crop[sy1:sy2, sx1:sx2]
        if screen.size == 0:
            return
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        prev = profile.phone_state
        import time
        if brightness >= Config.PHONE_ON_BRIGHTNESS:
            profile.phone_state = 'screen_on'
            if profile.phone_on_since is None:
                profile.phone_on_since = time.time()
            if profile.phone_last_screen is None:
                profile.phone_last_screen = gray.copy()
            else:
                g1 = cv2.resize(profile.phone_last_screen, (100, 160)).astype(np.float32)
                g2 = cv2.resize(gray, (100, 160)).astype(np.float32)
                delta = float(np.mean(np.abs(g1 - g2)))
                if delta >= Config.PHONE_CHANGE_THRESHOLD:
                    bus.emit('PHONE_CHANGE', f'{profile.name} screen changed ({delta:.1f})', beep=True, cooldown_key=f'phone-change:{profile.name}')
                    profile.phone_last_screen = gray.copy()
        else:
            profile.phone_state = 'screen_off'
            if profile.phone_on_since is not None:
                dur = time.time() - profile.phone_on_since
                bus.emit('PHONE_OFF', f'{profile.name} screen off after {dur:.1f}s', beep=True, cooldown_key=f'phone-off:{profile.name}')
            profile.phone_on_since = None
            profile.phone_last_screen = None
        if prev != profile.phone_state:
            bus.emit('PHONE_STATE', f'{profile.name} -> {profile.phone_state}', beep=True, cooldown_key=f'phone-state:{profile.name}')

    def update(self, frame, detections, prompt_to_profiles, bus):
        for profile in self.profiles.values():
            profile.frame_count += 1
            profile.visible = False
            profile.match_score = 0.0
            profile.detector_score = 0.0
            profile.orb_score = 0.0
            profile.prompt_used = ''

        candidates = []
        for det_idx, detection in enumerate(detections):
            rect = clip_rect(detection['rect'], frame.shape[1], frame.shape[0])
            crop = crop_rect(frame, rect)
            if crop is None or crop.size == 0:
                continue
            rect_area = max(1, rect[2] * rect[3])
            frame_area = max(1, frame.shape[0] * frame.shape[1])
            small_object = (rect_area / float(frame_area)) < 0.035
            for profile in self._candidate_profiles(detection, prompt_to_profiles):
                metrics = self._score_crop(profile, crop, detection['confidence'])
                if metrics is None:
                    continue
                if profile.expected_color and metrics['detected_color'] != profile.expected_color:
                    continue
                sample_count = max(1, len(profile.samples))
                threshold_bonus = min(0.18, 0.01 * max(0, sample_count - 1))
                orb_threshold = max(Config.ADAPTIVE_MIN_ORB_SCORE, Config.ORB_MATCH_THRESHOLD - threshold_bonus)
                score_threshold = max(Config.ADAPTIVE_MIN_MATCH_SCORE, Config.MATCH_SCORE_THRESHOLD - threshold_bonus)
                identity_threshold = max(Config.ADAPTIVE_MIN_IDENTITY_SCORE, 0.66 - threshold_bonus)
                if small_object:
                    orb_threshold = max(Config.ADAPTIVE_MIN_ORB_SCORE, orb_threshold - 0.02)
                    score_threshold = max(Config.ADAPTIVE_MIN_MATCH_SCORE, score_threshold - 0.03)
                    identity_threshold = max(Config.ADAPTIVE_MIN_IDENTITY_SCORE, identity_threshold - 0.03)
                if metrics['orb_score'] < orb_threshold:
                    continue
                if metrics['identity_score'] < identity_threshold:
                    continue
                if sample_count >= 6 and metrics.get('support_count', 0) < Config.ADAPTIVE_MULTI_SAMPLE_SUPPORT:
                    continue
                if metrics['score'] < score_threshold:
                    continue
                candidates.append((metrics['score'], profile.name, det_idx, metrics))

        used_profiles = set()
        used_detections = set()
        for _, profile_name, det_idx, metrics in sorted(candidates, key=lambda item: item[0], reverse=True):
            if profile_name in used_profiles or det_idx in used_detections:
                continue
            profile = self.profiles[profile_name]
            self._apply_detection(profile, detections[det_idx], metrics, frame, bus)
            used_profiles.add(profile_name)
            used_detections.add(det_idx)

        for name, profile in self.profiles.items():
            if name not in used_profiles:
                profile.lost_frames += 1
                if profile.lost_frames <= Config.ADAPTIVE_GRACE_FRAMES and len(profile.samples) >= 6 and profile.rect[2] > 0 and profile.rect[3] > 0:
                    profile.visible = True
