from core.config import Config


class OpenVocabularyDetector:
    def __init__(self):
        self.model = None
        self.last_prompts = []
        self.ready = False
        self.error = ''
        self.model_name = Config.DETECTOR_MODEL

    def _ensure_model(self):
        if self.model is not None or self.error:
            return self.model is not None
        try:
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(self.model_name)
            self.ready = True
            self.error = ''
            return True
        except Exception as exc:
            self.error = str(exc)
            self.ready = False
            return False

    def _set_prompts(self, prompts):
        prompts = list(dict.fromkeys([p.strip() for p in prompts if p and p.strip()]))
        if prompts == self.last_prompts:
            return
        self.model.set_classes(prompts)
        self.last_prompts = prompts

    def detect(self, frame, prompts):
        if not prompts:
            return []
        if not self._ensure_model():
            return []
        try:
            self._set_prompts(prompts)
            results = self.model.predict(
                source=frame,
                conf=Config.DETECTOR_CONFIDENCE,
                imgsz=Config.DETECTOR_IMAGE_SIZE,
                max_det=Config.DETECTOR_MAX_RESULTS,
                verbose=False,
            )
        except Exception as exc:
            self.error = str(exc)
            self.ready = False
            return []

        output = []
        if not results:
            return output
        result = results[0]
        names = result.names
        for box in result.boxes:
            cls_idx = int(box.cls[0].item())
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            if isinstance(names, dict):
                prompt_name = names.get(cls_idx, str(cls_idx))
            elif isinstance(names, (list, tuple)) and 0 <= cls_idx < len(names):
                prompt_name = names[cls_idx]
            else:
                prompt_name = str(cls_idx)
            output.append({
                'rect': (x1, y1, max(2, x2 - x1), max(2, y2 - y1)),
                'prompt': prompt_name,
                'confidence': float(box.conf[0].item()),
            })
        self.ready = True
        self.error = ''
        return output
