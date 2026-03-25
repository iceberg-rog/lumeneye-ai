class RuleEngine:
    def process(self, state, profiles, bus):
        if state.get('self_registered') and state.get('self_present'):
            bus.emit('RULE', 'Self detected in frame', cooldown_key='rule:self')
        for name, p in profiles.items():
            label = p.metadata.get('label', name)
            if p.ever_visible and p.lost_frames >= 30:
                bus.emit('RULE', f'{label} missing for a while', beep=True, cooldown_key=f'rule:missing:{name}', data={'cooldown_override': state.get('missing_event_repeat_sec')})
            if p.kind == 'phone' and p.visible and p.phone_state == 'screen_off':
                bus.emit('RULE', f'{label} is visible and screen is off', cooldown_key=f'rule:phone-off:{name}')
