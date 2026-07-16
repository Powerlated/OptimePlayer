/**
 * optime-ml training dashboard.
 *
 * Polls `/api/state` (the snapshot `src/dashboard.rs` serves) once a second and
 * renders it. No build step: the Vue global build is vendored next to this file and
 * both are baked into the training binary, so a run serves its own dashboard.
 *
 * Charts are hand-rolled SVG rather than a charting library — two line plots need
 * far less than a bundle would cost, and it keeps the offline/no-CDN property.
 * Colors come from the validated categorical palette in index.html: slot 1 (blue)
 * is always the training series, slot 2 (green) always the held-out one.
 */
const { createApp, ref, computed, onMounted, onUnmounted } = Vue;

/** Human-readable duration: `1h 04m`, `4m 12s`, `9.4s`. */
function fmtDuration(s) {
  if (!isFinite(s) || s < 0) return '—';
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60), sec = Math.round(s % 60);
  if (m < 60) return `${m}m ${String(sec).padStart(2, '0')}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${String(m % 60).padStart(2, '0')}m`;
}

/**
 * Elapsed-time axis tick: `45s`, `1m20`, `2h05`.
 *
 * Must stay distinct at the tick spacing actually used — rounding to whole minutes
 * renders 60s and 80s both as "1m", which reads as a broken axis.
 */
function fmtTimeTick(v) {
  if (v < 60) return `${Math.round(v)}s`;
  if (v < 3600) {
    const m = Math.floor(v / 60), s = Math.round(v % 60);
    return s ? `${m}m${String(s).padStart(2, '0')}` : `${m}m`;
  }
  const h = Math.floor(v / 3600), m = Math.round((v % 3600) / 60);
  return m ? `${h}h${String(m).padStart(2, '0')}` : `${h}h`;
}

/** Drop trailing zeros: `0.8000` -> `0.8`, `1.000` -> `1`. Integers pass through. */
function trimZeros(s) {
  return s.includes('.') ? s.replace(/0+$/, '').replace(/\.$/, '') : s;
}

/** Loss axis tick: enough digits to separate ticks, no more. */
function fmtLossTick(v) {
  return Math.abs(v) >= 0.01 || v === 0 ? trimZeros(v.toFixed(3)) : v.toExponential(1);
}

/** Compact count: 1.2 K, 4.6 M, 2.1 B. Used for params and FLOPs. */
function fmtCount(n, unit = '') {
  if (!isFinite(n)) return '—';
  const steps = [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']];
  for (const [mag, suffix] of steps) {
    if (n >= mag) return `${(n / mag).toFixed(n / mag < 10 ? 2 : 1)} ${suffix}${unit}`;
  }
  return `${Math.round(n)} ${unit}`.trim();
}

/** FLOPs read in FLOP/MFLOP/GFLOP, not "B FLOP". */
function fmtFlops(n) {
  if (!isFinite(n)) return '—';
  const steps = [[1e12, 'TFLOP'], [1e9, 'GFLOP'], [1e6, 'MFLOP'], [1e3, 'kFLOP']];
  for (const [mag, suffix] of steps) {
    if (n >= mag) return `${(n / mag).toFixed(n / mag < 10 ? 2 : 1)} ${suffix}`;
  }
  return `${Math.round(n)} FLOP`;
}

/** Config values arrive as raw JSON — render them readably without lying about type. */
function fmtConfigValue(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (typeof v === 'number') {
    if (Number.isInteger(v)) return v.toLocaleString();
    // Small learning rates are unreadable in fixed notation.
    return Math.abs(v) < 1e-3 ? v.toExponential(1) : String(v);
  }
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

/** Flatten a config object to [key, value] rows, nested objects dotted. */
function configRows(obj, prefix = '') {
  if (!obj || typeof obj !== 'object') return [];
  const rows = [];
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === 'object' && !Array.isArray(v)) rows.push(...configRows(v, key));
    else rows.push({ key, value: fmtConfigValue(v) });
  }
  return rows;
}

/** Axis ticks on 1/2/5×10ⁿ steps covering [min, max]. */
function niceTicks(min, max, count = 5) {
  if (!isFinite(min) || !isFinite(max)) return [];
  if (min === max) { const d = Math.abs(min) || 1; min -= d * 0.5; max += d * 0.5; }
  const raw = (max - min) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  const step = (n >= 5 ? 10 : n >= 2 ? 5 : n >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(v);
  return out;
}

/**
 * Line plot with a crosshair + tooltip hover layer.
 *
 * `series`: `[{ name, color, points: [{x, y}], dots }]`. All series share one y
 * axis — only ever pass measures in the same unit (this is why loss and accuracy
 * are two separate plots, never one dual-axis chart).
 */
const LinePlot = {
  props: {
    series: { type: Array, required: true },
    xTitle: { type: String, default: '' },
    fmtX: { type: Function, default: (v) => String(Math.round(v)) },
    /** Y axis ticks — few digits, they repeat down the axis. */
    fmtY: { type: Function, default: (v) => v.toFixed(3) },
    /** Tooltip values — full precision; defaults to the axis formatter. */
    fmtTip: { type: Function, default: null },
  },
  setup(props) {
    const W = 920, H = 250, pad = { l: 58, r: 70, t: 12, b: 30 };
    const svg = ref(null);
    const hover = ref(null); // { x, rows: [{name,color,y}], clientX, clientY }

    const shown = computed(() => props.series.filter((s) => s.points.length > 0));
    const all = computed(() => shown.value.flatMap((s) => s.points));

    const xDom = computed(() => {
      const xs = all.value.map((p) => p.x);
      if (!xs.length) return [0, 1];
      const lo = Math.min(...xs), hi = Math.max(...xs);
      return lo === hi ? [lo, lo + 1] : [lo, hi];
    });
    const yDom = computed(() => {
      const ys = all.value.map((p) => p.y);
      if (!ys.length) return [0, 1];
      let lo = Math.min(...ys), hi = Math.max(...ys);
      const padY = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 0.5;
      lo -= padY; hi += padY;
      return [Math.min(lo, hi), Math.max(lo, hi)];
    });

    const sx = (v) => pad.l + ((v - xDom.value[0]) / (xDom.value[1] - xDom.value[0])) * (W - pad.l - pad.r);
    const sy = (v) => H - pad.b - ((v - yDom.value[0]) / (yDom.value[1] - yDom.value[0])) * (H - pad.t - pad.b);

    const xTicks = computed(() => niceTicks(xDom.value[0], xDom.value[1], 6).map((v) => ({ v, x: sx(v) })));
    const yTicks = computed(() => niceTicks(yDom.value[0], yDom.value[1], 5).map((v) => ({ v, y: sy(v) })));

    const paths = computed(() =>
      shown.value.map((s) => ({
        ...s,
        d: s.points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' '),
        marks: s.dots ? s.points.map((p) => ({ cx: sx(p.x), cy: sy(p.y) })) : [],
        last: s.points[s.points.length - 1],
      }))
    );

    function onMove(e) {
      const box = svg.value?.getBoundingClientRect();
      if (!box || !shown.value.length) return;
      const vx = ((e.clientX - box.left) / box.width) * W;
      const dataX = xDom.value[0] + ((vx - pad.l) / (W - pad.l - pad.r)) * (xDom.value[1] - xDom.value[0]);
      const rows = [];
      let anchor = null;
      for (const s of shown.value) {
        // Nearest point in this series to the cursor's x — every series reports,
        // so the tooltip compares them at one instant.
        let best = null, bestD = Infinity;
        for (const p of s.points) {
          const d = Math.abs(p.x - dataX);
          if (d < bestD) { bestD = d; best = p; }
        }
        if (best) {
          rows.push({ name: s.name, color: s.color, y: best.y });
          if (!anchor || Math.abs(best.x - dataX) < Math.abs(anchor.x - dataX)) anchor = best;
        }
      }
      if (anchor) hover.value = { x: anchor.x, rows, clientX: e.clientX, clientY: e.clientY };
    }
    const onLeave = () => { hover.value = null; };

    const tipStyle = computed(() => {
      if (!hover.value) return {};
      const dx = hover.value.clientX > window.innerWidth - 190 ? -12 - 170 : 14;
      return { left: `${hover.value.clientX + dx}px`, top: `${hover.value.clientY - 12}px` };
    });

    const tip = (v) => (props.fmtTip ?? props.fmtY)(v);

    return { W, H, pad, svg, shown, paths, xTicks, yTicks, sx, sy, hover, onMove, onLeave, tipStyle, tip };
  },
  template: `
    <div>
      <div class="legend" v-if="shown.length > 1">
        <span v-for="s in shown" :key="s.name">
          <i class="swatch" :style="{ background: s.color }"></i>{{ s.name }}
        </span>
      </div>
      <svg ref="svg" class="plot" :viewBox="'0 0 ' + W + ' ' + H"
           @pointermove="onMove" @pointerleave="onLeave">
        <!-- Recessive grid, drawn under the marks. -->
        <line v-for="t in yTicks" :key="'gy' + t.v" class="grid-line"
              :x1="pad.l" :x2="W - pad.r" :y1="t.y" :y2="t.y" />
        <text v-for="t in yTicks" :key="'ly' + t.v" :x="pad.l - 8" :y="t.y + 3" text-anchor="end">{{ fmtY(t.v) }}</text>
        <line class="axis" :x1="pad.l" :x2="W - pad.r" :y1="H - pad.b" :y2="H - pad.b" />
        <text v-for="t in xTicks" :key="'lx' + t.v" :x="t.x" :y="H - pad.b + 15" text-anchor="middle">{{ fmtX(t.v) }}</text>
        <text :x="(W - pad.r + pad.l) / 2" :y="H - 2" text-anchor="middle">{{ xTitle }}</text>

        <g v-if="hover">
          <line class="crosshair" :x1="sx(hover.x)" :x2="sx(hover.x)" :y1="pad.t" :y2="H - pad.b" />
        </g>

        <g v-for="s in paths" :key="s.name">
          <path :d="s.d" :style="{ stroke: s.color }" fill="none" stroke-width="2"
                stroke-linejoin="round" stroke-linecap="round" />
          <!-- 2px surface ring keeps overlapping markers separable. -->
          <circle v-for="(m, i) in s.marks" :key="i" :cx="m.cx" :cy="m.cy" r="4"
                  :style="{ fill: s.color }" stroke="var(--surface-1)" stroke-width="2" />
          <!-- Direct label at the line end: identity never rests on color alone. -->
          <text v-if="s.last" class="direct-label" :x="sx(s.last.x) + 8" :y="sy(s.last.y) + 4"
                :style="{ fill: s.color }">{{ s.name }}</text>
        </g>
      </svg>
      <div v-if="hover" class="tooltip" :style="tipStyle">
        <div class="tt-row" v-for="r in hover.rows" :key="r.name">
          <span class="tt-key"><i class="swatch" :style="{ background: r.color }"></i>{{ r.name }}</span>
          <strong>{{ tip(r.y) }}</strong>
        </div>
      </div>
    </div>`,
};

createApp({
  components: { LinePlot },
  setup() {
    const state = ref(null);
    const error = ref(null);
    let timer = null;

    async function poll() {
      try {
        const r = await fetch('/api/state', { cache: 'no-store' });
        state.value = await r.json();
        error.value = null;
      } catch (e) {
        error.value = String(e);
      }
    }
    onMounted(() => { poll(); timer = setInterval(poll, 1000); });
    onUnmounted(() => clearInterval(timer));

    const meta = computed(() => state.value?.meta ?? null);
    const epochs = computed(() => state.value?.epochs ?? []);
    const batches = computed(() => state.value?.batches ?? []);
    const finished = computed(() => !!state.value?.finished);
    const latest = computed(() => batches.value[batches.value.length - 1] ?? null);
    const lastEpoch = computed(() => epochs.value[epochs.value.length - 1] ?? null);

    /** Fraction of the whole run done, batch-resolution. */
    const progress = computed(() => {
      if (finished.value) return 1;
      if (!meta.value || !latest.value || !latest.value.of) return 0;
      const done = epochs.value.length + latest.value.batch / latest.value.of;
      return Math.min(1, done / meta.value.epochs);
    });

    /** Prefer measured epoch times; fall back to batch throughput before epoch 1. */
    const eta = computed(() => {
      if (finished.value || !meta.value) return null;
      if (epochs.value.length) {
        const mean = epochs.value.reduce((a, e) => a + e.secs, 0) / epochs.value.length;
        const partial = latest.value?.of ? latest.value.batch / latest.value.of : 0;
        return mean * (meta.value.epochs - epochs.value.length - partial);
      }
      if (latest.value?.rate > 0 && latest.value.of) {
        const left = latest.value.of * meta.value.epochs - latest.value.batch;
        return left / latest.value.rate;
      }
      return null;
    });

    const C1 = 'var(--series-1)', C2 = 'var(--series-2)';

    /** Loss vs wall time: the dense in-epoch curve plus each epoch's held-out loss,
     *  same unit, one axis. */
    const lossSeries = computed(() => {
      const s = [{
        name: 'train',
        color: C1,
        dots: false,
        points: batches.value.map((b) => ({ x: b.t, y: b.loss })),
      }];
      const val = epochs.value.filter((e) => e.val_loss != null);
      if (val.length) {
        s.push({ name: 'held-out', color: C2, dots: true, points: val.map((e) => ({ x: e.t, y: e.val_loss })) });
      }
      return s;
    });

    /** Fine-tune only. Accuracy is a different unit from loss, so it gets its own
     *  plot rather than a second y-axis. */
    const accSeries = computed(() => {
      const e = epochs.value.filter((x) => x.chord_acc != null);
      if (!e.length) return [];
      return [
        { name: 'chord', color: C1, dots: true, points: e.map((x) => ({ x: x.epoch, y: x.chord_acc * 100 })) },
        { name: 'key', color: C2, dots: true, points: e.map((x) => ({ x: x.epoch, y: x.key_acc * 100 })) },
      ];
    });

    const isSupervised = computed(() => accSeries.value.length > 0);

    const modelRows = computed(() => configRows(meta.value?.model_config));
    const trainRows = computed(() => configRows(meta.value?.train_config));

    return {
      state, error, meta, epochs, batches, finished, latest, lastEpoch,
      progress, eta, lossSeries, accSeries, isSupervised, modelRows, trainRows,
      fmtDuration, fmtCount, fmtFlops,
      fmtLossTick,
      fmtLoss: (v) => v.toFixed(4),
      fmtPct: (v) => v.toFixed(0) + '%',
      fmtSecs: fmtTimeTick,
      fmtEpochTick: (v) => String(Math.round(v)),
    };
  },
  template: `
  <div>
    <header>
      <h1>optime-ml training</h1>
      <span class="pill" :class="{ live: !finished && meta }">
        <i class="dot"></i>{{ finished ? 'finished' : (meta ? 'running' : 'waiting') }}
      </span>
      <span class="sub" v-if="meta">{{ meta.stage }} · <strong>{{ meta.backbone }}</strong> · <code>{{ meta.backend }}</code></span>
    </header>

    <div v-if="error" class="empty">lost contact with the training process — {{ error }}</div>
    <div v-else-if="!meta" class="empty">waiting for a run to report…</div>

    <template v-else>
      <div class="tiles">
        <div class="tile">
          <div class="label">Epoch</div>
          <div class="value">{{ Math.min(epochs.length + (finished ? 0 : 1), meta.epochs) }}<small> / {{ meta.epochs }}</small></div>
        </div>
        <div class="tile">
          <div class="label">Batch</div>
          <div class="value" v-if="latest">{{ latest.batch }}<small> / {{ latest.of }}</small></div>
          <div class="value" v-else>—</div>
        </div>
        <div class="tile">
          <div class="label">Train loss</div>
          <div class="value">{{ latest ? latest.loss.toFixed(4) : '—' }}</div>
        </div>
        <div class="tile" v-if="!isSupervised">
          <div class="label">Held-out loss</div>
          <div class="value">{{ lastEpoch && lastEpoch.val_loss != null ? lastEpoch.val_loss.toFixed(4) : '—' }}</div>
        </div>
        <div class="tile" v-else>
          <div class="label">Chord acc</div>
          <div class="value">{{ lastEpoch ? (lastEpoch.chord_acc * 100).toFixed(1) + '%' : '—' }}</div>
        </div>
        <div class="tile">
          <div class="label">Throughput</div>
          <div class="value">{{ latest ? latest.rate.toFixed(1) : '—' }}<small> batch/s</small></div>
        </div>
        <div class="tile">
          <div class="label">{{ finished ? 'Took' : 'Elapsed' }}</div>
          <div class="value" style="font-size:18px">{{ fmtDuration(state.elapsed) }}</div>
          <div class="label" v-if="eta != null" style="margin-top:4px">~{{ fmtDuration(eta) }} left</div>
        </div>
      </div>

      <div class="bar"><i :style="{ width: (progress * 100).toFixed(2) + '%' }"></i></div>

      <div class="card">
        <h2>Loss</h2>
        <p class="hint">In-epoch running mean, sampled once a second, against each epoch's held-out loss.</p>
        <line-plot v-if="batches.length" :series="lossSeries" x-title="elapsed"
                   :fmt-x="fmtSecs" :fmt-y="fmtLossTick" :fmt-tip="fmtLoss" />
        <div v-else class="empty">no batches reported yet</div>
      </div>

      <div class="card" v-if="isSupervised">
        <h2>Held-out accuracy</h2>
        <p class="hint">Validation accuracy per epoch. Separate plot from loss — different unit, never a second axis.</p>
        <line-plot :series="accSeries" x-title="epoch" :fmt-x="fmtEpochTick" :fmt-y="fmtPct" />
      </div>

      <div class="card">
        <h2>Epochs</h2>
        <p class="hint">The table view of the plots above.</p>
        <div class="scroll" v-if="epochs.length">
          <table>
            <thead>
              <tr>
                <th>Epoch</th><th>Train loss</th>
                <th v-if="!isSupervised">Held-out loss</th>
                <template v-else><th>Key acc</th><th>Chord acc</th><th>Changes/seq</th></template>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="e in [...epochs].reverse()" :key="e.epoch">
                <td>{{ e.epoch }}</td>
                <td>{{ e.train_loss.toFixed(4) }}</td>
                <td v-if="!isSupervised">{{ e.val_loss != null ? e.val_loss.toFixed(4) : '—' }}</td>
                <template v-else>
                  <td>{{ (e.key_acc * 100).toFixed(1) }}%</td>
                  <td>{{ (e.chord_acc * 100).toFixed(1) }}%</td>
                  <td>{{ e.changes.toFixed(1) }}</td>
                </template>
                <td>{{ fmtDuration(e.secs) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-else class="empty">first epoch still running</div>
      </div>

      <div class="card">
        <h2>Model</h2>
        <p class="hint">Context is the trunk's sequence length. Seconds assume 120 bpm — the model's
          time axis is beats, not wall time.</p>
        <div class="facts">
          <div class="fact">
            <span class="k">Parameters</span>
            <span class="v">{{ fmtCount(meta.params) }}<em>{{ meta.params.toLocaleString() }}</em></span>
          </div>
          <div class="fact">
            <span class="k">Context window</span>
            <span class="v">{{ meta.context.tokens }} tokens<em>{{ meta.context.beats }} beats · {{ meta.context.seconds.toFixed(1) }}s @ 120bpm</em></span>
          </div>
          <div class="fact">
            <span class="k">FLOPs / window</span>
            <span class="v">{{ fmtFlops(meta.flops_per_window) }}<em>forward, matmuls only</em></span>
          </div>
          <div class="fact">
            <span class="k">FLOPs / batch</span>
            <span class="v">{{ fmtFlops(meta.flops_per_window * (meta.train_config.batch_size || 1)) }}<em>× batch of {{ meta.train_config.batch_size }}</em></span>
          </div>
        </div>
      </div>

      <div class="card">
        <h2>Training data</h2>
        <p class="hint">Hours are at 120 bpm. Augmented hours count each transposition as new material —
          more variety, not more information.</p>
        <div class="facts">
          <div class="fact">
            <span class="k">Music (unaugmented)</span>
            <span class="v">{{ meta.data.train_hours.toFixed(1) }} h<em>{{ Math.round(meta.data.train_beats).toLocaleString() }} beats</em></span>
          </div>
          <div class="fact">
            <span class="k">Music (augmented)</span>
            <span class="v">{{ meta.data.augmented_hours.toFixed(0) }} h<em>× {{ meta.data.transpositions }} transpositions</em></span>
          </div>
          <div class="fact">
            <span class="k">Windows</span>
            <span class="v">{{ meta.data.train_windows.toLocaleString() }}<em>+ {{ meta.data.val_windows.toLocaleString() }} val</em></span>
          </div>
          <div class="fact">
            <span class="k">Notes / window</span>
            <span class="v">{{ meta.data.notes_per_window.toFixed(0) }}<em>mean</em></span>
          </div>
        </div>
      </div>

      <div class="card">
        <h2>Hyperparameters</h2>
        <p class="hint">Serialized from the run's own configs, so this can't drift from the model.</p>
        <div class="cfg-cols">
          <div>
            <h3>Architecture</h3>
            <table>
              <tbody>
                <tr v-for="r in modelRows" :key="r.key"><td>{{ r.key }}</td><td>{{ r.value }}</td></tr>
              </tbody>
            </table>
          </div>
          <div>
            <h3>Optimisation</h3>
            <table>
              <tbody>
                <tr v-for="r in trainRows" :key="r.key"><td>{{ r.key }}</td><td>{{ r.value }}</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <p class="meta-line" v-if="state.saved_to">saved to <code>{{ state.saved_to }}</code></p>
    </template>
  </div>`,
}).mount('#app');
