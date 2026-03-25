from flask import Flask, jsonify, Response


def create_app(state):
    app = Flask(__name__)

    html = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LumenEye AI</title>
  <style>
    :root{
      --bg:#06101d;
      --bg-2:#0a1727;
      --panel:#102035;
      --panel-2:#132740;
      --line:rgba(166,193,232,.14);
      --text:#eef5ff;
      --muted:#8da4c7;
      --soft:#cfdcf2;
      --accent:#51d6a7;
      --accent-2:#69a9ff;
      --warn:#ffc76c;
      --danger:#ff878f;
      --shadow:0 30px 90px rgba(0,0,0,.45);
    }
    *{box-sizing:border-box}
    html,body{height:100%}
    body{
      margin:0;
      color:var(--text);
      font-family:"Aptos","Segoe UI Variable Display","Bahnschrift","Trebuchet MS",sans-serif;
      background:
        radial-gradient(circle at top left, rgba(81,214,167,.15), transparent 24%),
        radial-gradient(circle at top right, rgba(105,169,255,.14), transparent 28%),
        linear-gradient(165deg, var(--bg) 0%, #081320 42%, var(--bg-2) 100%);
    }
    .shell{
      max-width:1500px;
      margin:0 auto;
      padding:28px;
    }
    .hero{
      display:grid;
      grid-template-columns:1.5fr .9fr;
      gap:18px;
      margin-bottom:18px;
    }
    .panel{
      background:linear-gradient(180deg, rgba(19,39,64,.94), rgba(11,24,40,.96));
      border:1px solid var(--line);
      border-radius:24px;
      box-shadow:var(--shadow);
      backdrop-filter:blur(18px);
    }
    .hero-main{
      position:relative;
      overflow:hidden;
      padding:24px;
    }
    .hero-main::after{
      content:"";
      position:absolute;
      inset:auto -50px -60px auto;
      width:260px;
      height:260px;
      background:radial-gradient(circle, rgba(81,214,167,.18), transparent 66%);
      pointer-events:none;
    }
    .eyebrow{
      display:inline-flex;
      align-items:center;
      gap:10px;
      padding:7px 12px;
      border-radius:999px;
      border:1px solid rgba(255,255,255,.08);
      background:rgba(255,255,255,.04);
      font-size:12px;
      letter-spacing:.14em;
      text-transform:uppercase;
      color:#dbe8fc;
    }
    .dot{
      width:8px;
      height:8px;
      border-radius:50%;
      background:var(--accent);
      box-shadow:0 0 0 8px rgba(81,214,167,.10);
    }
    h1{
      margin:18px 0 8px;
      font-size:46px;
      letter-spacing:-.04em;
      line-height:1;
    }
    .lede{
      margin:0;
      max-width:780px;
      color:#a9bdd9;
      font-size:16px;
      line-height:1.65;
    }
    .hero-stats{
      margin-top:22px;
      display:grid;
      grid-template-columns:repeat(4, minmax(0,1fr));
      gap:14px;
    }
    .metric{
      padding:16px;
      border-radius:18px;
      background:rgba(255,255,255,.04);
      border:1px solid rgba(255,255,255,.06);
    }
    .metric .k{
      color:var(--muted);
      font-size:11px;
      text-transform:uppercase;
      letter-spacing:.14em;
      margin-bottom:8px;
    }
    .metric .v{
      font-size:30px;
      font-weight:700;
      letter-spacing:-.03em;
    }
    .hero-side{
      padding:22px;
      display:flex;
      flex-direction:column;
      gap:16px;
    }
    .stack-title{
      margin-bottom:8px;
      color:var(--muted);
      font-size:11px;
      letter-spacing:.14em;
      text-transform:uppercase;
    }
    .status-card{
      padding:16px 18px;
      border-radius:18px;
      background:rgba(255,255,255,.04);
      border:1px solid rgba(255,255,255,.06);
    }
    .status-line{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:14px;
      margin-bottom:10px;
    }
    .status-name{
      font-size:15px;
      color:#dbe8fd;
    }
    .status-detail{
      color:#aebfda;
      font-size:13px;
      line-height:1.55;
      word-break:break-word;
    }
    .pill{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      min-width:78px;
      padding:8px 12px;
      border-radius:999px;
      font-size:12px;
      font-weight:700;
      letter-spacing:.08em;
      text-transform:uppercase;
      border:1px solid transparent;
    }
    .good{background:rgba(81,214,167,.12); color:#a7f0d4; border-color:rgba(81,214,167,.24)}
    .warn{background:rgba(255,199,108,.12); color:#ffdca0; border-color:rgba(255,199,108,.24)}
    .bad{background:rgba(255,135,143,.12); color:#ffc1c6; border-color:rgba(255,135,143,.24)}
    .prompt-cloud{
      display:flex;
      flex-wrap:wrap;
      gap:8px;
      min-height:44px;
    }
    .chip{
      padding:8px 12px;
      border-radius:999px;
      background:rgba(105,169,255,.11);
      border:1px solid rgba(105,169,255,.18);
      color:#dce8fe;
      font-size:12px;
    }
    .tabs{
      display:flex;
      gap:10px;
      margin-bottom:18px;
      flex-wrap:wrap;
    }
    .tab-btn{
      appearance:none;
      border:none;
      cursor:pointer;
      padding:12px 16px;
      border-radius:999px;
      color:#dbe8fd;
      background:rgba(255,255,255,.04);
      border:1px solid rgba(255,255,255,.07);
      font-size:13px;
      font-weight:700;
      letter-spacing:.08em;
      text-transform:uppercase;
      transition:.2s ease;
    }
    .tab-btn.active{
      background:linear-gradient(135deg, rgba(81,214,167,.18), rgba(105,169,255,.18));
      border-color:rgba(105,169,255,.28);
      box-shadow:inset 0 0 0 1px rgba(255,255,255,.04);
    }
    .tab-panel{display:none}
    .tab-panel.active{display:block}
    .grid{
      display:grid;
      grid-template-columns:1.05fr .95fr;
      gap:18px;
      margin-bottom:18px;
    }
    .section{padding:20px}
    .section-head{
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:14px;
      margin-bottom:16px;
    }
    .section-head h2{
      margin:0 0 4px;
      font-size:24px;
      letter-spacing:-.03em;
    }
    .section-head p{
      margin:0;
      color:var(--muted);
      font-size:14px;
      line-height:1.5;
    }
    .section-kpi{
      font-size:13px;
      color:#dde8fa;
      padding:9px 12px;
      border-radius:999px;
      background:rgba(255,255,255,.04);
      border:1px solid rgba(255,255,255,.06);
      white-space:nowrap;
    }
    .model-grid{
      display:grid;
      grid-template-columns:repeat(auto-fill, minmax(290px,1fr));
      gap:14px;
    }
    .model-card{
      border-radius:20px;
      padding:18px;
      background:linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.025));
      border:1px solid rgba(255,255,255,.06);
      min-height:196px;
      display:flex;
      flex-direction:column;
      gap:14px;
    }
    .model-card.active{
      border-color:rgba(81,214,167,.32);
      box-shadow:inset 0 0 0 1px rgba(81,214,167,.10);
    }
    .model-top{
      display:flex;
      justify-content:space-between;
      gap:12px;
    }
    .model-title{
      margin:0;
      font-size:20px;
      letter-spacing:-.03em;
    }
    .model-sub{
      margin-top:4px;
      color:var(--muted);
      font-size:13px;
    }
    .score-band{
      display:grid;
      grid-template-columns:repeat(3,minmax(0,1fr));
      gap:10px;
    }
    .score-item{
      padding:10px 12px;
      border-radius:14px;
      background:rgba(7,17,31,.30);
      border:1px solid rgba(255,255,255,.05);
    }
    .score-item .k,.meta b{
      color:var(--muted);
      font-size:10px;
      text-transform:uppercase;
      letter-spacing:.14em;
      margin-bottom:6px;
      display:block;
      font-weight:600;
    }
    .score-item .v{
      font-size:18px;
      font-weight:700;
    }
    .meta-list{
      display:grid;
      grid-template-columns:repeat(2,minmax(0,1fr));
      gap:10px;
      color:#dbe8ff;
      font-size:13px;
    }
    .meta{
      padding:10px 12px;
      border-radius:14px;
      background:rgba(255,255,255,.03);
      border:1px solid rgba(255,255,255,.05);
    }
    .table-wrap{
      overflow:auto;
      border-radius:18px;
      border:1px solid rgba(255,255,255,.06);
      background:rgba(255,255,255,.02);
    }
    table{width:100%; border-collapse:collapse}
    th,td{
      text-align:left;
      padding:13px 14px;
      border-bottom:1px solid rgba(255,255,255,.06);
      font-size:13px;
      vertical-align:top;
    }
    th{
      color:#98aed0;
      font-size:11px;
      text-transform:uppercase;
      letter-spacing:.14em;
      position:sticky;
      top:0;
      background:#102137;
    }
    tr:last-child td{border-bottom:none}
    .muted{color:var(--muted)}
    .event-list{
      display:flex;
      flex-direction:column;
      gap:12px;
      max-height:780px;
      overflow:auto;
      padding-right:4px;
    }
    .event-item{
      position:relative;
      padding:16px 16px 16px 20px;
      border-radius:18px;
      background:rgba(255,255,255,.03);
      border:1px solid rgba(255,255,255,.06);
    }
    .event-item::before{
      content:"";
      position:absolute;
      left:0;
      top:14px;
      bottom:14px;
      width:4px;
      border-radius:8px;
      background:linear-gradient(180deg, var(--accent-2), var(--accent));
    }
    .event-meta{
      display:flex;
      justify-content:space-between;
      gap:12px;
      margin-bottom:8px;
      color:var(--muted);
      font-size:12px;
      text-transform:uppercase;
      letter-spacing:.12em;
    }
    .event-msg{
      font-size:15px;
      line-height:1.5;
      color:#eff5ff;
    }
    .overview-grid{
      display:grid;
      grid-template-columns:repeat(3,minmax(0,1fr));
      gap:14px;
    }
    .insight{
      padding:18px;
      border-radius:20px;
      background:linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.025));
      border:1px solid rgba(255,255,255,.06);
      min-height:170px;
    }
    .insight h3{
      margin:0 0 10px;
      font-size:18px;
      letter-spacing:-.03em;
    }
    .insight p{
      margin:0;
      color:#b0c2dd;
      font-size:14px;
      line-height:1.6;
    }
    .big-number{
      margin:12px 0 8px;
      font-size:34px;
      font-weight:700;
      letter-spacing:-.04em;
    }
    .empty{
      border:1px dashed rgba(255,255,255,.12);
      border-radius:18px;
      padding:26px;
      text-align:center;
      color:var(--muted);
      background:rgba(255,255,255,.02);
    }
    .footer-note{
      margin-top:12px;
      color:var(--muted);
      font-size:12px;
      text-align:right;
    }
    @media (max-width:1200px){
      .hero,.grid,.overview-grid{grid-template-columns:1fr}
    }
    @media (max-width:760px){
      .shell{padding:16px}
      .hero-stats{grid-template-columns:repeat(2,minmax(0,1fr))}
      .score-band,.meta-list{grid-template-columns:1fr}
      h1{font-size:34px}
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="panel hero-main">
        <div class="eyebrow"><span class="dot"></span><span>Live Vision Intelligence</span></div>
        <h1>LumenEye AI Dashboard</h1>
        <p class="lede">A presentation-grade control center for open-vocabulary detection, object identity verification, and model behavior across live visual scenes.</p>
        <div class="hero-stats">
          <div class="metric"><div class="k">FPS</div><div class="v" id="fps">0.0</div></div>
          <div class="metric"><div class="k">Frames</div><div class="v" id="frames">0</div></div>
          <div class="metric"><div class="k">Models</div><div class="v" id="profiles_count">0</div></div>
          <div class="metric"><div class="k">Blinks</div><div class="v" id="blinks">0</div></div>
        </div>
      </div>
      <aside class="panel hero-side">
        <div>
          <div class="stack-title">Detector Status</div>
          <div class="status-card">
            <div class="status-line">
              <div class="status-name" id="detector_model">-</div>
              <div class="pill warn" id="detector_ready_badge">Standby</div>
            </div>
            <div class="status-detail" id="detector_error">Awaiting detector response.</div>
          </div>
        </div>
        <div>
          <div class="stack-title">Identity Signals</div>
          <div class="status-card">
            <div class="status-line">
              <div class="status-name">Self Registration</div>
              <div class="pill warn" id="self_registered_badge">No</div>
            </div>
            <div class="status-detail">Presence: <span id="self_present">NO</span> | Similarity score: <span id="self_score">0.00</span></div>
          </div>
        </div>
        <div>
          <div class="stack-title">Prompt Cloud</div>
          <div class="prompt-cloud" id="prompt_cloud"></div>
        </div>
      </aside>
    </section>

    <nav class="tabs">
      <button class="tab-btn active" data-tab="overview">Overview</button>
      <button class="tab-btn" data-tab="models">Models</button>
      <button class="tab-btn" data-tab="events">Events</button>
      <button class="tab-btn" data-tab="registry">Registry</button>
    </nav>

    <section class="tab-panel active" id="tab-overview">
      <div class="panel section">
        <div class="section-head">
          <div>
            <h2>Executive Overview</h2>
            <p>A concise system narrative for presentation, focusing on readiness, active matches, prompt scope, and verification confidence.</p>
          </div>
          <div class="section-kpi">Active model: <span id="active_profile">-</span></div>
        </div>
        <div class="overview-grid">
          <article class="insight">
            <h3>Detector Availability</h3>
            <div class="big-number" id="ov_detector_state">Standby</div>
            <p id="ov_detector_copy">The open-vocabulary detector is preparing to receive prompts.</p>
          </article>
          <article class="insight">
            <h3>Live Matches</h3>
            <div class="big-number" id="ov_live_matches">0</div>
            <p id="ov_live_copy">No verified objects are currently matched in the scene.</p>
          </article>
          <article class="insight">
            <h3>Prompt Coverage</h3>
            <div class="big-number" id="ov_prompt_count">0</div>
            <p id="ov_prompt_copy">Prompt generation expands dynamically from your model metadata and aliases.</p>
          </article>
        </div>
      </div>
    </section>

    <section class="tab-panel" id="tab-models">
      <div class="panel section">
        <div class="section-head">
          <div>
            <h2>Verified Models</h2>
            <p>Registered model identities, their prompt strategies, and the latest verification signals from the live scene.</p>
          </div>
          <div class="section-kpi">Presentation view</div>
        </div>
        <div class="model-grid" id="model_cards"></div>
        <div class="footer-note">Cards highlight detector prompt usage, confidence, and metadata cohesion in one view.</div>
      </div>
    </section>

    <section class="tab-panel" id="tab-events">
      <div class="panel section">
        <div class="section-head">
          <div>
            <h2>Live Event Feed</h2>
            <p>Recent platform activity across detection, motion, color shifts, phone state, and identity verification events.</p>
          </div>
          <div class="section-kpi">Last 120 events</div>
        </div>
        <div class="event-list" id="event_list"></div>
      </div>
    </section>

    <section class="tab-panel" id="tab-registry">
      <div class="panel section">
        <div class="section-head">
          <div>
            <h2>Model Registry</h2>
            <p>A full registry view for prompt strategy, expected color gating, latest detector confidence, and current model state.</p>
          </div>
          <div class="section-kpi">Prompt count: <span id="prompt_count">0</span></div>
        </div>
        <div class="table-wrap">
          <div id="profiles_table"></div>
        </div>
      </div>
    </section>
  </div>

  <script>
    function esc(value){
      return String(value ?? '')
        .replace(/&/g,'&amp;')
        .replace(/</g,'&lt;')
        .replace(/>/g,'&gt;')
        .replace(/"/g,'&quot;')
        .replace(/'/g,'&#39;');
    }
    function badgeClass(ok, warn=false){
      if(ok) return 'pill good';
      return warn ? 'pill warn' : 'pill bad';
    }
    function formatTime(ts){
      return new Date(ts * 1000).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
    }
    function activateTab(name){
      for(const btn of document.querySelectorAll('.tab-btn')){
        btn.classList.toggle('active', btn.dataset.tab === name);
      }
      for(const panel of document.querySelectorAll('.tab-panel')){
        panel.classList.toggle('active', panel.id === `tab-${name}`);
      }
    }
    for(const btn of document.querySelectorAll('.tab-btn')){
      btn.addEventListener('click', () => activateTab(btn.dataset.tab));
    }
    function renderPromptCloud(prompts){
      if(!prompts.length){
        prompt_cloud.innerHTML = '<div class="empty">No open-vocabulary prompts are active yet.</div>';
        return;
      }
      prompt_cloud.innerHTML = prompts.map(p => `<span class="chip">${esc(p)}</span>`).join('');
    }
    function renderModelCards(profiles, activeProfile){
      if(!profiles.length){
        model_cards.innerHTML = '<div class="empty">No models are registered yet.</div>';
        return;
      }
      model_cards.innerHTML = profiles.map(x => {
        const stateClass = x.visible ? 'good' : 'warn';
        const stateLabel = x.visible ? 'Matched' : 'Monitoring';
        const active = x.name === activeProfile ? ' active' : '';
        return `
          <article class="model-card${active}">
            <div class="model-top">
              <div>
                <h3 class="model-title">${esc(x.label)}</h3>
                <div class="model-sub">${esc(x.kind)} · ${esc(x.category || 'uncategorized')}</div>
              </div>
              <div class="${stateClass} pill">${stateLabel}</div>
            </div>
            <div class="score-band">
              <div class="score-item"><div class="k">Match</div><div class="v">${esc(x.confidence)}</div></div>
              <div class="score-item"><div class="k">Detector</div><div class="v">${esc(x.detector_confidence)}</div></div>
              <div class="score-item"><div class="k">Samples</div><div class="v">${esc(x.samples)}</div></div>
            </div>
            <div class="meta-list">
              <div class="meta"><b>Expected</b>${esc(x.expected_color || '-')}</div>
              <div class="meta"><b>Seen</b>${esc(x.color || '-')}</div>
              <div class="meta"><b>Prompt Used</b>${esc(x.prompt_used || '-')}</div>
              <div class="meta"><b>Detect As</b>${esc(x.detect_as || '-')}</div>
            </div>
          </article>
        `;
      }).join('');
    }
    function renderProfilesTable(profiles){
      let p = '<table><tr><th>Label</th><th>Detect As</th><th>Aliases</th><th>Kind</th><th>Expected</th><th>Seen</th><th>Match</th><th>Detect</th><th>Samples</th><th>State</th></tr>';
      for(const x of profiles){
        p += `<tr>
          <td><strong>${esc(x.label)}</strong><div class="muted">${esc(x.category || '-')}</div></td>
          <td>${esc(x.detect_as || '-')}</td>
          <td>${esc(x.aliases || '-')}</td>
          <td>${esc(x.kind)}</td>
          <td>${esc(x.expected_color || '-')}</td>
          <td>${esc(x.color || '-')}</td>
          <td>${esc(x.confidence)}</td>
          <td>${esc(x.detector_confidence)}</td>
          <td>${esc(x.samples)}</td>
          <td>${esc(x.state)}</td>
        </tr>`;
      }
      p += '</table>';
      profiles_table.innerHTML = p;
    }
    function renderEvents(events){
      if(!events.length){
        event_list.innerHTML = '<div class="empty">No events have been emitted yet.</div>';
        return;
      }
      event_list.innerHTML = events.slice().reverse().map(it => `
        <article class="event-item">
          <div class="event-meta">
            <span>${esc(it.type)}</span>
            <span>${esc(formatTime(it.ts))}</span>
          </div>
          <div class="event-msg">${esc(it.message)}</div>
        </article>
      `).join('');
    }
    async function refresh(){
      const res = await fetch('/api/state');
      const s = await res.json();
      const profiles = s.profiles || [];
      const visibleCount = profiles.filter(x => x.visible).length;

      fps.innerText = s.fps.toFixed(1);
      frames.innerText = s.frame_index;
      profiles_count.innerText = s.profiles_count;
      blinks.innerText = s.blink_total;
      active_profile.innerText = s.active_profile || '-';
      self_present.innerText = s.self_present ? 'Yes' : 'No';
      self_score.innerText = s.self_score.toFixed(2);
      prompt_count.innerText = (s.open_vocab_prompts || []).length;
      detector_model.innerText = s.detector_model || '-';
      ov_prompt_count.innerText = (s.open_vocab_prompts || []).length;
      ov_live_matches.innerText = String(visibleCount);

      detector_ready_badge.className = badgeClass(s.detector_ready, true);
      detector_ready_badge.innerText = s.detector_ready ? 'Online' : 'Attention';
      detector_error.innerText = s.detector_error || (s.detector_ready ? `${visibleCount} verified match(es) currently visible.` : 'Detector is initializing.');

      self_registered_badge.className = badgeClass(s.self_registered, true);
      self_registered_badge.innerText = s.self_registered ? 'Ready' : 'Pending';

      ov_detector_state.innerText = s.detector_ready ? 'Online' : 'Standby';
      ov_detector_copy.innerText = s.detector_error || (s.detector_ready ? 'The open-vocabulary detector is active and processing live prompts.' : 'The system is waiting for detector readiness.');
      ov_live_copy.innerText = visibleCount ? `${visibleCount} verified object match(es) are currently active in the scene.` : 'No verified objects are currently matched in the scene.';
      ov_prompt_copy.innerText = (s.open_vocab_prompts || []).length ? 'Prompt generation is active and sourced directly from dynamic model metadata.' : 'Prompt generation will populate as models are defined.';

      renderPromptCloud(s.open_vocab_prompts || []);
      renderModelCards(profiles, s.active_profile);
      renderProfilesTable(profiles);
      renderEvents(s.events || []);
    }
    setInterval(refresh, 700);
    refresh();
  </script>
</body>
</html>'''

    @app.route('/')
    def dashboard():
        return Response(html, mimetype='text/html')

    @app.route('/events')
    def events():
        return jsonify(state['events'][-120:])

    @app.route('/api/state')
    def api_state():
        profiles = list(state['profiles'].values())
        return jsonify({
            'fps': state['fps'],
            'frame_index': state['frame_index'],
            'profiles_count': len(state['profiles']),
            'blink_total': state['blink_total'],
            'self_registered': state['self_registered'],
            'self_present': state['self_present'],
            'self_score': state['self_score'],
            'active_profile': state['active_profile'],
            'detector_ready': state['detector_ready'],
            'detector_error': state['detector_error'],
            'detector_model': state['detector_model'],
            'open_vocab_prompts': state['open_vocab_prompts'],
            'profiles': profiles,
            'events': state['events'][-120:],
        })
    return app
