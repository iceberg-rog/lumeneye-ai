import mediapipe as mp
from core.config import Config
from core.utils import clip_rect, crop_rect, face_embedding, cosine_similarity

class FaceModule:
    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.self_embedding = None
        self.last_faces = []
        self.blink_counter = 0
        self.blink_total = 0
        self.left_eye = [33, 160, 158, 133, 153, 144]
        self.right_eye = [362, 385, 387, 263, 373, 380]

    def _eye_ratio(self, pts, idxs):
        import numpy as np
        p = [pts[i] for i in idxs]
        A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
        B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
        C = np.linalg.norm(np.array(p[0]) - np.array(p[3])) + 1e-8
        return (A + B) / (2.0 * C)

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = frame[:, :, ::-1]
        faces = []
        det = self.detector.process(rgb)
        if det.detections:
            for d in det.detections:
                box = d.location_data.relative_bounding_box
                x = max(0, int(box.xmin * w))
                y = max(0, int(box.ymin * h))
                bw = int(box.width * w)
                bh = int(box.height * h)
                rect = clip_rect((x, y, bw, bh), w, h)
                crop = crop_rect(frame, rect)
                emb = face_embedding(crop)
                score = cosine_similarity(self.self_embedding, emb) if self.self_embedding is not None else 0.0
                faces.append({
                    'rect': rect,
                    'embedding': emb,
                    'score': score,
                    'is_self': bool(self.self_embedding is not None and score >= Config.SELF_FACE_THRESHOLD),
                })

        mesh = self.mesh.process(rgb)
        if mesh.multi_face_landmarks:
            import numpy as np
            pts = [(int(p.x * w), int(p.y * h)) for p in mesh.multi_face_landmarks[0].landmark]
            ear = (self._eye_ratio(pts, self.left_eye) + self._eye_ratio(pts, self.right_eye)) / 2.0
            if ear < Config.BLINK_THRESHOLD:
                self.blink_counter += 1
            else:
                if self.blink_counter >= Config.BLINK_CONSEC_FRAMES:
                    self.blink_total += 1
                self.blink_counter = 0
        else:
            self.blink_counter = 0

        self.last_faces = faces
        return faces

    def register_self(self):
        if not self.last_faces:
            return False
        best = max(self.last_faces, key=lambda f: f['rect'][2] * f['rect'][3])
        self.self_embedding = best['embedding']
        return self.self_embedding is not None
