import json
import time
import threading
from collections import deque
from pathlib import Path
import winsound
from .config import Config, EVENTS_DIR

class EventBus:
    def __init__(self):
        self.logs = deque(maxlen=Config.MAX_EVENTS)
        self.last_emit = {}
        EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = EVENTS_DIR / 'events.json'

    def emit(self, event_type, message, data=None, beep=False, cooldown_key=None):
        now = time.time()
        key = cooldown_key or f'{event_type}:{message}'
        payload = data or {}
        cooldown_sec = float(payload.get('cooldown_override', Config.EVENT_COOLDOWN_SEC))
        if now - self.last_emit.get(key, 0.0) < cooldown_sec:
            return None
        self.last_emit[key] = now
        item = {
            'ts': now,
            'type': event_type,
            'message': message,
            'data': payload,
        }
        self.logs.append(item)
        self._save()
        print(f'[{event_type}] {message}')
        if beep:
            threading.Thread(target=lambda: winsound.PlaySound(Config.BEEP_ALIAS, winsound.SND_ALIAS), daemon=True).start()
        return item

    def _save(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.logs), f, ensure_ascii=False, indent=2)

    def list(self):
        return list(self.logs)
