import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Wand2, Play, ImagePlus, Check, UploadCloud, Columns, Square, ZoomIn, RotateCcw,
  MoveHorizontal, Info, Image as ImageIcon, Sliders, Cpu, Sparkles, FlaskConical,
} from "lucide-react";

// ============================================================================
// Image Restoration — single-file interface around a (mock) restoration model.
// Degraded image in -> restored image out. Runs fully client-side on an honest
// classical placeholder; the real model would be wired via a FastAPI seam.
//
// Visual language: retro-futuristic pixel-arcade + brutalist editorial.
// Warm cream / golden yellow / deep navy, thick ink borders, hard offset
// shadows, dot-grid texture, pixel display type + monospace body. The
// restoration/comparison/metrics logic below is unchanged from prior versions.
// ============================================================================

// ---- design tokens ---------------------------------------------------------
const T = {
  cream: "#EBE3CB", cream2: "#F4EDD8", paper: "#FBF7EC", stage: "#141109",
  yellow: "#F0BE2E", yellowSoft: "#F6EAC2", yellowDeep: "#E2AC1C",
  navy: "#173049", navy2: "#0E2136",
  ink: "#211C12", ink2: "#2E2718",
  text: "#211C12", dim: "#5C5342", faint: "#8B8069",
  onNavy: "#EDE6CF", onNavyDim: "#A6B4C2",
  green: "#3E7C4F", blue: "#6AA7CB",
  input: "#C1782E", output: "#3E7C4F", gt: "#3E6E96", baseline: "#8A7F6A",
  good: "#3E7C4F", bad: "#C25438",
};
const LAYER_COLOR = { input: T.input, output: T.output, gt: T.gt, baseline: T.baseline };
const LAYER_LABEL = { input: "degraded input", output: "restored output", gt: "ground truth", baseline: "bicubic baseline" };
const DISPLAY = "'Pixelify Sans', system-ui, sans-serif";
const MONO = "'JetBrains Mono', monospace";
const BORDER = `2px solid ${T.ink}`;
const panel = (bg = T.paper, sh = 5) => ({ background: bg, border: BORDER, borderRadius: 5, boxShadow: `${sh}px ${sh}px 0 ${T.ink}` });

// ---- resampling kernels (real pixel math) ----------------------------------
const clampInt = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
function cubicWeight(x, a = -0.5) {
  x = Math.abs(x);
  if (x <= 1) return (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1;
  if (x < 2) return a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a;
  return 0;
}
function bicubicResize(src, srcW, srcH, dstW, dstH) {
  const dst = new Uint8ClampedArray(dstW * dstH * 4);
  const xR = srcW / dstW, yR = srcH / dstH;
  for (let dy = 0; dy < dstH; dy++) {
    const sy = (dy + 0.5) * yR - 0.5, syInt = Math.floor(sy);
    for (let dx = 0; dx < dstW; dx++) {
      const sx = (dx + 0.5) * xR - 0.5, sxInt = Math.floor(sx);
      let r = 0, g = 0, b = 0, a = 0, wsum = 0;
      for (let m = -1; m <= 2; m++) {
        const yy = clampInt(syInt + m, 0, srcH - 1), wy = cubicWeight(sy - (syInt + m));
        for (let n = -1; n <= 2; n++) {
          const xx = clampInt(sxInt + n, 0, srcW - 1), wx = cubicWeight(sx - (sxInt + n));
          const wgt = wx * wy, idx = (yy * srcW + xx) * 4;
          r += src[idx] * wgt; g += src[idx + 1] * wgt; b += src[idx + 2] * wgt; a += src[idx + 3] * wgt;
          wsum += wgt;
        }
      }
      const di = (dy * dstW + dx) * 4;
      dst[di] = r / wsum; dst[di + 1] = g / wsum; dst[di + 2] = b / wsum; dst[di + 3] = a / wsum;
    }
  }
  return new ImageData(dst, dstW, dstH);
}
function resizeCanvas(srcCanvas, dstW, dstH) {
  const sctx = srcCanvas.getContext("2d", { willReadFrequently: true });
  const sData = sctx.getImageData(0, 0, srcCanvas.width, srcCanvas.height).data;
  const imgData = bicubicResize(sData, srcCanvas.width, srcCanvas.height, dstW, dstH);
  const c = document.createElement("canvas"); c.width = dstW; c.height = dstH;
  c.getContext("2d").putImageData(imgData, 0, 0);
  return c;
}

// ---- image / canvas helpers ------------------------------------------------
function loadImageEl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image(); img.crossOrigin = "anonymous";
    img.onload = () => resolve(img); img.onerror = () => reject(new Error("decode failed")); img.src = url;
  });
}
function canvasFromImageEl(img, w = img.naturalWidth, h = img.naturalHeight) {
  const c = document.createElement("canvas"); c.width = Math.max(1, w); c.height = Math.max(1, h);
  c.getContext("2d").drawImage(img, 0, 0, c.width, c.height); return c;
}
function cloneCanvas(canvas) {
  const c = document.createElement("canvas"); c.width = canvas.width; c.height = canvas.height;
  c.getContext("2d").drawImage(canvas, 0, 0); return c;
}
const toDataURL = (c) => c.toDataURL("image/png");
async function urlToImageData(url, w, h) {
  const img = await loadImageEl(url);
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h); return ctx.getImageData(0, 0, w, h);
}
function gaussianBlur(canvas, passes = 1) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const { width: w, height: h } = canvas;
  let src = ctx.getImageData(0, 0, w, h).data;
  const k = [1, 2, 1], kSum = 4;
  const at = (a, x, y, c) => a[(Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))) * 4 + c];
  for (let p = 0; p < passes; p++) {
    const tmp = new Uint8ClampedArray(src.length);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 4; c++)
      tmp[(y * w + x) * 4 + c] = (k[0] * at(src, x - 1, y, c) + k[1] * at(src, x, y, c) + k[2] * at(src, x + 1, y, c)) / kSum;
    const out = new Uint8ClampedArray(src.length);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 4; c++)
      out[(y * w + x) * 4 + c] = (k[0] * at(tmp, x, y - 1, c) + k[1] * at(tmp, x, y, c) + k[2] * at(tmp, x, y + 1, c)) / kSum;
    src = out;
  }
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  c.getContext("2d").putImageData(new ImageData(src, w, h), 0, 0); return c;
}
function addGaussianNoise(canvas, sigma = 18, seed = 1) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const { width: w, height: h } = canvas;
  const d = ctx.getImageData(0, 0, w, h);
  let s = seed >>> 0;
  const rand = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  for (let i = 0; i < d.data.length; i += 4) {
    const u1 = Math.max(1e-6, rand()), u2 = rand();
    const g = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    for (let c = 0; c < 3; c++) d.data[i + c] = Math.max(0, Math.min(255, d.data[i + c] + g * sigma));
  }
  const out = document.createElement("canvas"); out.width = w; out.height = h;
  out.getContext("2d").putImageData(d, 0, 0); return out;
}
function unsharpMask(canvas, amount = 0.6) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const { width: w, height: h } = canvas;
  const orig = ctx.getImageData(0, 0, w, h).data;
  const blurred = gaussianBlur(canvas, 1).getContext("2d", { willReadFrequently: true }).getImageData(0, 0, w, h).data;
  const out = new Uint8ClampedArray(orig.length);
  for (let i = 0; i < orig.length; i += 4) {
    for (let c = 0; c < 3; c++) out[i + c] = Math.max(0, Math.min(255, orig[i + c] + amount * (orig[i + c] - blurred[i + c])));
    out[i + 3] = orig[i + 3];
  }
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  c.getContext("2d").putImageData(new ImageData(out, w, h), 0, 0); return c;
}

// ---- pure full-reference metrics -------------------------------------------
const MAXV = 255;
function mse(a, b) { let s = 0, n = 0; for (let i = 0; i < a.length; i += 4) for (let c = 0; c < 3; c++) { const d = a[i + c] - b[i + c]; s += d * d; n++; } return n ? s / n : 0; }
function mae(a, b) { let s = 0, n = 0; for (let i = 0; i < a.length; i += 4) for (let c = 0; c < 3; c++) { s += Math.abs(a[i + c] - b[i + c]); n++; } return n ? s / n : 0; }
const rmse = (a, b) => Math.sqrt(mse(a, b));
function psnr(a, b) { const m = mse(a, b); return m === 0 ? Infinity : 10 * Math.log10((MAXV * MAXV) / m); }
const luma = (arr, i) => 0.299 * arr[i] + 0.587 * arr[i + 1] + 0.114 * arr[i + 2];
function ssim(a, b, width, height, win = 8) {
  const C1 = (0.01 * MAXV) ** 2, C2 = (0.03 * MAXV) ** 2;
  let total = 0, windows = 0;
  for (let wy = 0; wy < height; wy += win) for (let wx = 0; wx < width; wx += win) {
    const w = Math.min(win, width - wx), h = Math.min(win, height - wy), n = w * h;
    if (n === 0) continue;
    let ma = 0, mb = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const i = ((wy + y) * width + (wx + x)) * 4; ma += luma(a, i); mb += luma(b, i); }
    ma /= n; mb /= n;
    let va = 0, vb = 0, cov = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const i = ((wy + y) * width + (wx + x)) * 4; const da = luma(a, i) - ma, db = luma(b, i) - mb; va += da * da; vb += db * db; cov += da * db; }
    va /= n; vb /= n; cov /= n;
    total += ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2)); windows++;
  }
  return windows ? total / windows : 1;
}
const fullReferenceMetrics = (cand, ref, w, h) => ({ psnr: psnr(cand, ref), ssim: ssim(cand, ref, w, h), mae: mae(cand, ref), rmse: rmse(cand, ref) });

// ---- synthetic sample pairs (clearly labeled demo data) --------------------
const GT_SIZE = 256;
function drawCleanScene(seed) {
  const c = document.createElement("canvas"); c.width = GT_SIZE; c.height = GT_SIZE;
  const ctx = c.getContext("2d");
  const g = ctx.createLinearGradient(0, 0, GT_SIZE, GT_SIZE);
  g.addColorStop(0, "#20304a"); g.addColorStop(0.5, "#3a4b66"); g.addColorStop(1, "#161d29");
  ctx.fillStyle = g; ctx.fillRect(0, 0, GT_SIZE, GT_SIZE);
  ctx.strokeStyle = "#d7e2f0"; ctx.lineWidth = 1;
  const cx = 78 + (seed % 3) * 20, cy = 92;
  for (let r = 6; r < 90; r += 5) { ctx.globalAlpha = 0.85 - r / 140; ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke(); }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "#9fb4cf";
  for (let i = 0; i < GT_SIZE; i += 9) { ctx.beginPath(); ctx.moveTo(GT_SIZE - i, 150); ctx.lineTo(GT_SIZE, 150 + i); ctx.stroke(); }
  ctx.fillStyle = "#e0a458"; ctx.fillRect(150, 40, 62, 46);
  ctx.fillStyle = "#45c9b0"; ctx.beginPath(); ctx.moveTo(190, 200); ctx.lineTo(232, 246); ctx.lineTo(148, 246); ctx.closePath(); ctx.fill();
  const cs = 6;
  for (let y = 0; y < 48; y += cs) for (let x = 0; x < 48; x += cs) { ctx.fillStyle = ((x / cs + y / cs) % 2 === 0) ? "#f0f3f8" : "#2a3340"; ctx.fillRect(28 + x, 190 + y, cs, cs); }
  ctx.fillStyle = "#c9d6e6";
  for (let i = 0; i < 20; i++) ctx.fillRect(150 + i * 5, 120, 2, 10 + (i % 4) * 4);
  return c;
}
function buildSamplePairs() {
  const specs = [
    { id: "sample-noise", name: "sample-noise.png", degradation: "additive Gaussian noise", note: "clean + Gaussian noise σ≈22", degrade: (clean) => ({ input: addGaussianNoise(cloneCanvas(clean), 22, 7), scaleFactor: 1 }) },
    { id: "sample-downscale", name: "sample-downscale.png", degradation: "2× downsampling", note: "bicubically downsampled to 128² (2× SR)", degrade: (clean) => ({ input: resizeCanvas(clean, GT_SIZE / 2, GT_SIZE / 2), scaleFactor: 2 }) },
    { id: "sample-combined", name: "sample-combined.png", degradation: "blur + downsample + noise", note: "blurred, downsampled, then noised", degrade: (clean) => { const b = gaussianBlur(cloneCanvas(clean), 2); const s = resizeCanvas(b, GT_SIZE / 2, GT_SIZE / 2); return { input: addGaussianNoise(s, 12, 13), scaleFactor: 2 }; } },
  ];
  return specs.map((s) => {
    const clean = drawCleanScene(s.id.length);
    const { input, scaleFactor } = s.degrade(clean);
    return { id: s.id, name: s.name, url: toDataURL(input), width: input.width, height: input.height, gtUrl: toDataURL(clean), degradation: s.degradation, isSample: true, note: s.note, scaleFactor };
  });
}

// ---- placeholder restorer (NOT a trained model) ----------------------------
const delay = (ms) => new Promise((r) => setTimeout(r, ms));
function referenceSize(srcCanvas, gtImg, scaleFactor) {
  if (gtImg) return { w: gtImg.naturalWidth, h: gtImg.naturalHeight };
  const f = scaleFactor || 1; return { w: srcCanvas.width * f, h: srcCanvas.height * f };
}
async function buildBaseline({ imageUrl, gtUrl, scaleFactor }) {
  const img = await loadImageEl(imageUrl), src = canvasFromImageEl(img);
  const gtImg = gtUrl ? await loadImageEl(gtUrl) : null;
  const { w, h } = referenceSize(src, gtImg, scaleFactor);
  return { url: toDataURL(resizeCanvas(src, w, h)), width: w, height: h };
}
async function runPlaceholderRestore({ imageUrl, gtUrl, scaleFactor }) {
  const t0 = performance.now();
  const img = await loadImageEl(imageUrl), src = canvasFromImageEl(img);
  const gtImg = gtUrl ? await loadImageEl(gtUrl) : null;
  const { w, h } = referenceSize(src, gtImg, scaleFactor);
  const tPre = performance.now();
  const up = resizeCanvas(src, w, h);
  const sharpened = unsharpMask(gaussianBlur(up, 1), 0.5);
  await delay(150);
  const tInf = performance.now();
  const restoredUrl = toDataURL(sharpened);
  const tEnd = performance.now();
  return {
    restoredUrl, outputWidth: w, outputHeight: h, timeMs: tEnd - t0,
    stageTimings: { preprocessMs: tPre - t0, inferenceMs: tInf - tPre, postprocessMs: tEnd - tInf },
    scaleFactor: scaleFactor || 1, isPlaceholder: true, modelName: "classical placeholder (no model connected)",
  };
}

// ---- shared viewport (one zoom/pan state for every panel) ------------------
const MIN_SCALE = 1, MAX_SCALE = 16;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const clampCenter = (o, scale) => (scale <= 1 ? 0.5 : clamp(o, 0.5 / scale, 1 - 0.5 / scale));
function useViewport() {
  const [vp, setVp] = useState({ scale: 1, ox: 0.5, oy: 0.5 });
  const zoomAt = useCallback((cx, cy, w, h, deltaY) => {
    setVp((prev) => {
      const scale = clamp(prev.scale * Math.exp(-deltaY * 0.0016), MIN_SCALE, MAX_SCALE);
      if (scale === prev.scale) return prev;
      const imgX = prev.ox + (cx / w - 0.5) / prev.scale, imgY = prev.oy + (cy / h - 0.5) / prev.scale;
      return { scale, ox: clampCenter(imgX - (cx / w - 0.5) / scale, scale), oy: clampCenter(imgY - (cy / h - 0.5) / scale, scale) };
    });
  }, []);
  const panBy = useCallback((dx, dy, w, h) => {
    setVp((prev) => prev.scale <= 1 ? prev : { ...prev, ox: clampCenter(prev.ox - dx / (prev.scale * w), prev.scale), oy: clampCenter(prev.oy - dy / (prev.scale * h), prev.scale) });
  }, []);
  const reset = useCallback(() => setVp({ scale: 1, ox: 0.5, oy: 0.5 }), []);
  const transformFor = useCallback((w, h) => `translate(${w * (0.5 - vp.scale * vp.ox)}px, ${h * (0.5 - vp.scale * vp.oy)}px) scale(${vp.scale})`, [vp]);
  return { vp, zoomAt, panBy, reset, transformFor };
}

// ---- primitives ------------------------------------------------------------
const Mono = ({ children, style }) => <span style={{ fontFamily: MONO, ...style }}>{children}</span>;
function Card({ title, sub, right, children, style, bg }) {
  return (
    <div style={{ ...panel(bg || T.paper), padding: 16, ...style }}>
      {(title || right) && (
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 10, marginBottom: 13 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
            <span style={{ width: 10, height: 10, background: T.yellow, border: BORDER, flex: "0 0 auto" }} />
            <div>
              {title && <div style={{ fontFamily: DISPLAY, fontSize: 15, fontWeight: 600, letterSpacing: 0.4, textTransform: "uppercase", color: T.ink }}>{title}</div>}
              {sub && <div style={{ fontSize: 11, color: T.dim, marginTop: 2, fontFamily: MONO }}>{sub}</div>}
            </div>
          </div>
          {right}
        </div>
      )}
      {children}
    </div>
  );
}
// A role chip: a pixel color-swatch + monospace label. Mirrors the little
// colored square markers used across the layout.
const Tag = ({ color, children }) => (
  <span style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "3px 8px", fontSize: 10.5, fontFamily: MONO, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.4, whiteSpace: "nowrap", color: T.ink, background: T.paper, border: BORDER, borderRadius: 3 }}>
    <span style={{ width: 8, height: 8, background: color, boxShadow: `0 0 0 1px ${T.ink}` }} />
    {children}
  </span>
);
const Demo = () => <span style={{ fontFamily: MONO, fontSize: 9, fontWeight: 700, letterSpacing: 1, color: T.cream2, background: T.ink, padding: "2px 5px", borderRadius: 2 }}>DEMO</span>;
const ErrorNote = ({ children }) => <div style={{ ...panel("#F3DCCF", 4), padding: "10px 12px", color: "#7A2E17", fontFamily: MONO, fontSize: 12.5, lineHeight: 1.5 }}>{children}</div>;

// ---- one image panel -------------------------------------------------------
function ViewerStage({ layer, url, transformFor, onZoom, onPan, onReset, clip, showLabel = true, height = 300, interactive = true, frame = true }) {
  const ref = useRef(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const drag = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(ref.current); return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const el = ref.current; if (!el || !interactive) return;
    const onWheel = (e) => { e.preventDefault(); const r = el.getBoundingClientRect(); onZoom(e.clientX - r.left, e.clientY - r.top, r.width, r.height, e.deltaY); };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [onZoom, interactive]);
  const down = (e) => { drag.current = { x: e.clientX, y: e.clientY }; e.currentTarget.setPointerCapture?.(e.pointerId); };
  const move = (e) => { if (!drag.current) return; const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y; drag.current = { x: e.clientX, y: e.clientY }; onPan(dx, dy, size.w, size.h); };
  const end = () => { drag.current = null; };
  const transform = size.w ? transformFor(size.w, size.h) : "none";
  return (
    <div ref={ref}
      onPointerDown={interactive ? down : undefined} onPointerMove={interactive ? move : undefined}
      onPointerUp={interactive ? end : undefined} onPointerLeave={interactive ? end : undefined}
      onDoubleClick={interactive ? () => onReset?.() : undefined}
      style={{ position: "relative", height, background: frame ? T.stage : "transparent", overflow: "hidden", borderRadius: frame ? 4 : 0, border: frame ? BORDER : "none", boxShadow: frame ? `4px 4px 0 ${T.ink}` : "none", cursor: interactive ? "grab" : "default", touchAction: "none", pointerEvents: interactive ? "auto" : "none", clipPath: clip, WebkitClipPath: clip }}>
      {url ? (
        <img src={url} alt={LAYER_LABEL[layer]} draggable={false}
          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", objectFit: "contain", transform, transformOrigin: "0 0", userSelect: "none", imageRendering: "auto" }} />
      ) : (
        <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", color: T.faint, fontSize: 11, fontFamily: MONO }}>not available</div>
      )}
      {showLabel && <div style={{ position: "absolute", top: 8, left: 8, zIndex: 3 }}><Tag color={LAYER_COLOR[layer]}>{LAYER_LABEL[layer]}</Tag></div>}
    </div>
  );
}

// ---- comparison viewer (the centerpiece) -----------------------------------
const MODES = [
  { id: "slider", label: "Slider", icon: MoveHorizontal },
  { id: "sideBySide", label: "Side-by-side", icon: Columns },
  { id: "single", label: "Single", icon: Square },
];
function LayerPicker({ label, value, onChange, options }) {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 11, fontFamily: MONO, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.4, color: T.dim }}>
      {label}
      <select className="pixel-select" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((k) => <option key={k} value={k}>{LAYER_LABEL[k]}</option>)}
      </select>
    </label>
  );
}
function CompareViewer({ layers }) {
  const available = layers.filter((l) => l.url);
  const keys = available.map((l) => l.key);
  const urlOf = (k) => available.find((l) => l.key === k)?.url;
  const [mode, setMode] = useState("slider");
  const { zoomAt, panBy, reset, transformFor } = useViewport();
  const [leftKey, setLeftKey] = useState("input");
  const [rightKey, setRightKey] = useState("output");
  const [singleKey, setSingleKey] = useState("output");
  const [sideKeys, setSideKeys] = useState(["input", "output"]);
  const keySig = keys.join(",");
  useEffect(() => {
    if (!keys.length) return;
    const has = (k) => keys.includes(k);
    if (!has(leftKey)) setLeftKey(keys[0]);
    if (!has(rightKey)) setRightKey(keys.includes("output") ? "output" : keys[keys.length - 1]);
    if (!has(singleKey)) setSingleKey(keys.includes("output") ? "output" : keys[0]);
    setSideKeys((prev) => { const next = prev.filter(has); while (next.length < 2 && keys.length >= 2) { const add = keys.find((k) => !next.includes(k)); if (add) next.push(add); else break; } return next.length ? next : keys.slice(0, 2); });
  }, [keySig]); // eslint-disable-line
  const [divider, setDivider] = useState(50);
  const dragging = useRef(false);
  const frameRef = useRef(null);
  const setFromClientX = useCallback((clientX) => { if (!frameRef.current) return; const r = frameRef.current.getBoundingClientRect(); setDivider(Math.max(2, Math.min(98, ((clientX - r.left) / r.width) * 100))); }, []);
  useEffect(() => {
    const m = (e) => { if (dragging.current) setFromClientX(e.touches ? e.touches[0].clientX : e.clientX); };
    const u = () => { dragging.current = false; };
    window.addEventListener("mousemove", m); window.addEventListener("mouseup", u); window.addEventListener("touchmove", m); window.addEventListener("touchend", u);
    return () => { window.removeEventListener("mousemove", m); window.removeEventListener("mouseup", u); window.removeEventListener("touchmove", m); window.removeEventListener("touchend", u); };
  }, [setFromClientX]);
  const sp = { transformFor, onZoom: zoomAt, onPan: panBy, onReset: reset };
  const H = 360;
  const tabStyle = (active) => ({ padding: "7px 12px", fontSize: 11.5, background: active ? T.yellow : T.cream2, color: T.ink, ...(active ? { transform: "translate(2px,2px)", boxShadow: `1px 1px 0 ${T.ink}` } : null) });
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12, flexWrap: "wrap" }}>
        <div style={{ display: "flex", gap: 8 }}>
          {MODES.map((m) => { const Icon = m.icon; const active = mode === m.id; return (
            <button key={m.id} className="pixel-tab" onClick={() => setMode(m.id)} style={tabStyle(active)}>
              <Icon size={13} /> {m.label}
            </button>
          ); })}
        </div>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 12, color: T.dim, fontSize: 11, fontFamily: MONO }}>
          <span style={{ display: "flex", alignItems: "center", gap: 5 }}><ZoomIn size={13} /> scroll to zoom · drag to pan</span>
          <button className="pixel-tab" onClick={reset} title="Reset view" style={{ padding: "6px 10px", fontSize: 11, background: T.cream2, color: T.ink }}><RotateCcw size={12} /> Reset</button>
        </div>
      </div>

      {mode === "slider" && (
        <div>
          <div ref={frameRef} style={{ position: "relative", ...panel(T.stage, 5), overflow: "hidden", borderRadius: 4 }}>
            <ViewerStage layer={rightKey} url={urlOf(rightKey)} {...sp} height={H} showLabel={false} frame={false} />
            <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
              <ViewerStage layer={leftKey} url={urlOf(leftKey)} {...sp} height={H} showLabel={false} clip={`inset(0 ${100 - divider}% 0 0)`} interactive={false} frame={false} />
            </div>
            <div style={{ position: "absolute", top: 10, left: 10, zIndex: 4 }}><Tag color={LAYER_COLOR[leftKey]}>{LAYER_LABEL[leftKey]}</Tag></div>
            <div style={{ position: "absolute", top: 10, right: 10, zIndex: 4 }}><Tag color={LAYER_COLOR[rightKey]}>{LAYER_LABEL[rightKey]}</Tag></div>
            <div onMouseDown={() => (dragging.current = true)} onTouchStart={() => (dragging.current = true)} style={{ position: "absolute", top: 0, bottom: 0, left: `${divider}%`, width: 0, zIndex: 5, cursor: "ew-resize" }}>
              <div style={{ position: "absolute", top: 0, bottom: 0, left: -1.5, width: 3, background: T.yellow }} />
              <div style={{ position: "absolute", top: "50%", left: -18, transform: "translateY(-50%)", width: 34, height: 34, background: T.yellow, border: BORDER, borderRadius: 4, display: "grid", placeItems: "center", boxShadow: `2px 2px 0 ${T.ink}` }}><MoveHorizontal size={16} color={T.ink} /></div>
            </div>
          </div>
          <div style={{ display: "flex", gap: 14, marginTop: 12 }}>
            <LayerPicker label="Left" value={leftKey} onChange={setLeftKey} options={keys} />
            <LayerPicker label="Right" value={rightKey} onChange={setRightKey} options={keys} />
          </div>
        </div>
      )}

      {mode === "sideBySide" && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${sideKeys.length}, 1fr)`, gap: 14 }}>
            {sideKeys.map((k, i) => <ViewerStage key={i} layer={k} url={urlOf(k)} {...sp} height={H} />)}
          </div>
          <div style={{ display: "flex", gap: 12, marginTop: 12, flexWrap: "wrap", alignItems: "center" }}>
            {sideKeys.map((k, i) => <LayerPicker key={i} label={`Panel ${i + 1}`} value={k} onChange={(v) => setSideKeys((prev) => prev.map((x, j) => (j === i ? v : x)))} options={keys} />)}
            {sideKeys.length < 3 && keys.length > sideKeys.length && <button className="pixel-tab" onClick={() => setSideKeys((prev) => [...prev, keys.find((k) => !prev.includes(k)) || keys[0]])} style={{ padding: "6px 10px", fontSize: 11, background: T.cream2, color: T.ink }}>+ panel</button>}
            {sideKeys.length > 2 && <button className="pixel-tab" onClick={() => setSideKeys((prev) => prev.slice(0, -1))} style={{ padding: "6px 10px", fontSize: 11, background: T.cream2, color: T.ink }}>− panel</button>}
          </div>
        </div>
      )}

      {mode === "single" && (
        <div>
          <ViewerStage layer={singleKey} url={urlOf(singleKey)} {...sp} height={H} />
          <div style={{ display: "flex", gap: 10, marginTop: 12, flexWrap: "wrap" }}>
            {keys.map((k) => (
              <button key={k} className="pixel-tab" onClick={() => setSingleKey(k)} style={{ padding: "6px 11px", fontSize: 11, textTransform: "none", background: singleKey === k ? T.yellow : T.cream2, color: T.ink, ...(singleKey === k ? { transform: "translate(2px,2px)", boxShadow: `1px 1px 0 ${T.ink}` } : null) }}>
                <span style={{ width: 8, height: 8, background: LAYER_COLOR[k], boxShadow: `0 0 0 1px ${T.ink}`, marginRight: 6, display: "inline-block" }} />{LAYER_LABEL[k]}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ---- results ---------------------------------------------------------------
const fmt = (v, d = 3) => (v === Infinity ? "∞" : Number.isFinite(v) ? v.toFixed(d) : "—");
function deltaOf(v, b, higherBetter) {
  if (b == null || !Number.isFinite(v) || !Number.isFinite(b)) return null;
  const d = v - b;
  return { text: `${d >= 0 ? "+" : ""}${d.toFixed(2)} vs bicubic`, better: higherBetter ? d > 0 : d < 0 };
}
function MetricCard({ label, value, unit, delta, accent, big }) {
  return (
    <div style={{ ...panel(T.paper, big ? 5 : 4), padding: big ? "13px 14px 15px" : "11px 12px" }}>
      <div style={{ fontSize: 10, fontFamily: MONO, fontWeight: 600, letterSpacing: 0.7, textTransform: "uppercase", color: T.dim }}>{label}</div>
      <div style={{ marginTop: 6, display: "flex", alignItems: "baseline", gap: 5 }}>
        <span style={{ fontFamily: DISPLAY, fontSize: big ? 34 : 22, fontWeight: 600, color: accent || T.ink, lineHeight: 1 }}>{value}</span>
        {unit && <span style={{ fontSize: 12, fontFamily: MONO, color: T.faint }}>{unit}</span>}
      </div>
      {delta && <div style={{ marginTop: 7, fontSize: 10.5, fontFamily: MONO, fontWeight: 600, color: delta.better ? T.good : T.bad }}>{delta.text}</div>}
    </div>
  );
}
// The metrics that matter for restoration (PSNR/SSIM when a reference exists,
// plus inference time). Bicubic baseline is a secondary toggle; MAE/RMSE demoted.
function ResultsPanel({ result, modelMetrics, baselineMetrics, hasGT, baselineEnabled, onToggleBaseline, busy }) {
  const showDelta = baselineEnabled && baselineMetrics;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14, position: "sticky", top: 18 }}>
      <Card title="Results" sub={result ? (hasGT ? "quality vs. ground truth" : "restored output") : "shown after you restore"}>
        {busy && !result ? (
          <div style={{ color: T.dim, fontSize: 12.5, fontFamily: MONO }}>Restoring…</div>
        ) : !result ? (
          <div style={{ color: T.dim, fontSize: 12.5, fontFamily: MONO, lineHeight: 1.6 }}>Restore an image to see PSNR, SSIM, and inference time.</div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
            {hasGT ? (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 11 }}>
                <MetricCard big label="PSNR" unit="dB" accent={T.output}
                  value={modelMetrics ? fmt(modelMetrics.psnr, 2) : "…"}
                  delta={showDelta && modelMetrics ? deltaOf(modelMetrics.psnr, baselineMetrics.psnr, true) : null} />
                <MetricCard big label="SSIM"
                  value={modelMetrics ? fmt(modelMetrics.ssim, 3) : "…"}
                  delta={showDelta && modelMetrics ? deltaOf(modelMetrics.ssim, baselineMetrics.ssim, true) : null} />
              </div>
            ) : (
              <div style={{ display: "flex", gap: 9, alignItems: "flex-start", color: T.dim, fontSize: 12, fontFamily: MONO, lineHeight: 1.6 }}>
                <Info size={14} color={T.faint} style={{ flex: "0 0 auto", marginTop: 1 }} />
                <span>PSNR and SSIM need a clean reference image. Add a ground-truth image to measure restoration quality.</span>
              </div>
            )}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 11 }}>
              <MetricCard label="Inference" unit="ms" value={result.timeMs.toFixed(0)} />
              <MetricCard label="Output" value={`${result.outputWidth}×${result.outputHeight}`} />
            </div>
            {hasGT && modelMetrics && (
              <div style={{ fontSize: 10, color: T.faint, fontFamily: MONO }}>
                MAE {fmt(modelMetrics.mae, 2)} · RMSE {fmt(modelMetrics.rmse, 2)} · {modelMetrics.source === "backend" ? "from model" : "measured in-browser"}
              </div>
            )}
          </div>
        )}
      </Card>

      <Card>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
          <div>
            <div style={{ fontSize: 12.5, fontFamily: MONO, fontWeight: 600, color: T.ink }}>Bicubic baseline</div>
            <div style={{ fontSize: 10.5, color: T.faint, fontFamily: MONO, marginTop: 2 }}>Classical reference for comparison</div>
          </div>
          <button role="switch" aria-checked={baselineEnabled} onClick={onToggleBaseline}
            style={{ width: 46, height: 24, border: BORDER, borderRadius: 4, background: baselineEnabled ? T.green : T.cream2, position: "relative", cursor: "pointer", flex: "0 0 auto", padding: 0, boxShadow: `2px 2px 0 ${T.ink}` }}>
            <span style={{ position: "absolute", top: 2, left: baselineEnabled ? 24 : 2, width: 16, height: 16, background: T.ink, borderRadius: 2, transition: "left .12s" }} />
          </button>
        </div>
      </Card>
    </div>
  );
}

// ---- pipeline bar ----------------------------------------------------------
const STEPS = [
  { id: "input", label: "Input", icon: ImageIcon },
  { id: "preprocess", label: "Preprocess", icon: Sliders },
  { id: "infer", label: "Model inference", icon: Cpu },
  { id: "output", label: "Restored output", icon: Sparkles },
];
const REACHED = { idle: 0, preprocess: 1, infer: 2, metrics: 3, baseline: 3, done: 4 };
function PipelineBar({ stage, result }) {
  const reached = REACHED[stage] ?? 0;
  const active = stage !== "idle" && stage !== "done";
  const timings = result?.stageTimings;
  const stepTiming = (id) => { if (!timings) return null; if (id === "preprocess" && timings.preprocessMs != null) return `${timings.preprocessMs.toFixed(0)} ms`; if (id === "infer") return `${timings.inferenceMs.toFixed(0)} ms`; if (id === "output" && timings.postprocessMs != null) return `${timings.postprocessMs.toFixed(0)} ms`; return null; };
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4, ...panel(T.cream2, 4), padding: "11px 14px" }}>
      {STEPS.map((s, i) => {
        const Icon = s.icon; const done = reached > i + 1 || stage === "done"; const isActive = active && reached === i + 1;
        const bg = done ? T.green : isActive ? T.yellow : T.paper; const t = stepTiming(s.id);
        return (
          <React.Fragment key={s.id}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
              <div style={{ width: 24, height: 24, borderRadius: 3, display: "grid", placeItems: "center", background: bg, border: BORDER }}>
                {done ? <Check size={13} color={T.cream2} /> : <Icon size={12} color={T.ink} />}
              </div>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 11.5, fontFamily: MONO, fontWeight: 600, color: done || isActive ? T.ink : T.dim, whiteSpace: "nowrap" }}>{s.label}</div>
                {t && <Mono style={{ fontSize: 10, color: T.faint }}>{t}</Mono>}
              </div>
            </div>
            {i < STEPS.length - 1 && <div style={{ flex: 1, height: 2, minWidth: 14, background: reached > i + 1 ? T.green : T.ink, opacity: reached > i + 1 ? 1 : 0.25, margin: "0 6px" }} />}
          </React.Fragment>
        );
      })}
    </div>
  );
}

// ---- placeholder banner ----------------------------------------------------
const PlaceholderBanner = ({ result }) => (!result || !result.isPlaceholder) ? null : (
  <div style={{ display: "flex", alignItems: "center", gap: 10, ...panel(T.yellow, 4), padding: "10px 13px", color: T.ink, fontSize: 12.5, fontFamily: MONO, lineHeight: 1.5 }}>
    <FlaskConical size={16} color={T.ink} style={{ flex: "0 0 auto" }} />
    <span><strong>No model connected.</strong> Showing a classical placeholder (bicubic upsample + denoise), not a trained model. Wire up the backend to see real restoration.</span>
  </div>
);

// ---- dropzone --------------------------------------------------------------
function Dropzone({ onImage, size = "default" }) {
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);
  const handle = (files) => { const f = Array.from(files || []).find((x) => x.type.startsWith("image/")); if (f) onImage(f); };
  const pad = size === "compact" ? "14px 12px" : size === "big" ? "40px 24px" : "22px 16px";
  const tile = size === "big" ? 54 : 40;
  const label = size === "big" ? "Drop a degraded image here, or click to browse" : size === "compact" ? "Drop image" : "Drop a degraded image, or click to choose";
  return (
    <div onClick={() => inputRef.current?.click()} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={(e) => { e.preventDefault(); setOver(false); handle(e.dataTransfer.files); }}
      style={{ border: `2px dashed ${over ? T.green : T.ink}`, background: over ? T.yellowSoft : T.paper, borderRadius: 5, boxShadow: `5px 5px 0 ${T.ink}`, padding: pad, textAlign: "center", cursor: "pointer", transition: "background .12s, border-color .12s" }}>
      <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => handle(e.target.files)} />
      <div style={{ display: "inline-grid", placeItems: "center", width: tile, height: tile, background: T.yellow, border: BORDER, borderRadius: 4, marginBottom: 12, boxShadow: `2px 2px 0 ${T.ink}` }}>
        <UploadCloud size={tile - 24} color={T.ink} />
      </div>
      <div style={{ fontFamily: MONO, fontWeight: 600, fontSize: size === "big" ? 14.5 : 12.5, color: T.ink }}>{label}</div>
      {size === "big" && <div style={{ fontSize: 11.5, fontFamily: MONO, color: T.faint, marginTop: 5 }}>PNG or JPG — noisy, blurry, or low-resolution</div>}
    </div>
  );
}

// ---- product chrome --------------------------------------------------------
function Header({ hasSource, onNew }) {
  return (
    <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "13px 22px", borderBottom: `3px solid ${T.ink}`, gap: 14, flexWrap: "wrap", background: T.cream2 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{ width: 34, height: 34, borderRadius: 4, background: T.yellow, border: BORDER, display: "grid", placeItems: "center", boxShadow: `2px 2px 0 ${T.ink}` }}><Wand2 size={17} color={T.ink} /></div>
        <div>
          <div style={{ fontFamily: DISPLAY, fontSize: 17, fontWeight: 700, letterSpacing: 0.5, textTransform: "uppercase", color: T.ink }}>Image Restoration</div>
          <div style={{ fontSize: 11, color: T.dim, fontFamily: MONO }}>Recover clean images from degraded input</div>
        </div>
      </div>
      {hasSource && (
        <button className="pixel-btn" onClick={onNew} style={{ padding: "8px 14px", fontSize: 12, background: T.cream, color: T.ink }}>
          <ImagePlus size={14} /> New image
        </button>
      )}
    </header>
  );
}

function SampleChip({ s, onClick }) {
  return (
    <button className="pixel-btn" onClick={onClick} style={{ display: "flex", width: "100%", justifyContent: "flex-start", textTransform: "none", letterSpacing: 0, alignItems: "center", gap: 11, padding: 9, background: T.paper, color: T.ink }}>
      <img src={s.url} alt="" style={{ width: 48, height: 48, objectFit: "cover", border: BORDER, borderRadius: 3, background: T.stage, flex: "0 0 auto" }} />
      <div style={{ minWidth: 0, textAlign: "left" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span style={{ fontSize: 12, fontFamily: MONO, fontWeight: 600, color: T.ink }}>{s.degradation}</span>
          <Demo />
        </div>
        <div style={{ fontSize: 10.5, color: T.faint, fontFamily: MONO, marginTop: 3 }}>{s.width}×{s.height}{s.scaleFactor > 1 ? ` · ${s.scaleFactor}× SR` : ""}</div>
      </div>
    </button>
  );
}

// Upload-first landing. Navy hero states the job; the upload card overlaps the
// seam as the centerpiece; demo samples sit below, clearly labeled.
function Hero({ onUploadInput, samples, onPickSample }) {
  return (
    <div>
      <div className="grid-dots-navy" style={{ background: T.navy, padding: "56px 20px 92px", borderBottom: `3px solid ${T.ink}` }}>
        <div style={{ maxWidth: 780, margin: "0 auto", textAlign: "center" }}>
          <div style={{ fontFamily: MONO, fontSize: 12, fontWeight: 600, letterSpacing: 2, textTransform: "uppercase", color: T.yellow, marginBottom: 16 }}>AI Image Restoration</div>
          <h1 className="hero-h1">Restore degraded images</h1>
          <p style={{ fontFamily: MONO, fontSize: 14, lineHeight: 1.7, color: T.onNavyDim, maxWidth: 520, margin: "18px auto 0" }}>
            Upload a noisy, blurry, or low-resolution image and recover a clean version — then compare before and after, pixel for pixel.
          </p>
        </div>
      </div>
      <div style={{ maxWidth: 640, margin: "-66px auto 0", padding: "0 20px 48px", position: "relative", zIndex: 2 }}>
        <Dropzone onImage={onUploadInput} size="big" />
        <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "30px 0 16px" }}>
          <div style={{ flex: 1, height: 2, background: T.ink, opacity: 0.25 }} />
          <span style={{ fontFamily: MONO, fontSize: 11, fontWeight: 600, letterSpacing: 1, textTransform: "uppercase", color: T.dim }}>or try a demo sample</span>
          <div style={{ flex: 1, height: 2, background: T.ink, opacity: 0.25 }} />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: 12 }}>
          {samples.map((s) => <SampleChip key={s.id} s={s} onClick={() => onPickSample(s)} />)}
        </div>
        <div style={{ textAlign: "center", marginTop: 16, fontSize: 11, fontFamily: MONO, color: T.faint }}>
          Demo samples are synthetic test images with known ground truth — for trying the tool.
        </div>
      </div>
    </div>
  );
}

// Slim summary of the loaded image, with an inline way to attach ground truth.
function SourceBar({ source, result, busy, onUploadGT }) {
  const gtInput = useRef(null);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, ...panel(T.paper, 4), padding: "10px 12px" }}>
      <img src={source.url} alt="" style={{ width: 44, height: 44, objectFit: "cover", border: BORDER, borderRadius: 3, background: T.stage, flex: "0 0 auto" }} />
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 13, fontFamily: MONO, fontWeight: 600, color: T.ink, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxWidth: 240 }}>{source.name}</div>
        <div style={{ display: "flex", alignItems: "center", gap: 7, marginTop: 4, flexWrap: "wrap" }}>
          <span style={{ fontSize: 10.5, color: T.faint, fontFamily: MONO }}>{source.width}×{source.height}</span>
          {source.degradation && <Tag color={T.input}>{source.degradation}</Tag>}
          {source.isSample && <Demo />}
        </div>
      </div>
      <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 12 }}>
        {source.gtUrl ? (
          <Tag color={T.gt}>ground truth ✓</Tag>
        ) : (
          <button className="pixel-btn" onClick={() => gtInput.current?.click()} style={{ padding: "6px 10px", fontSize: 10.5, background: T.cream2, color: T.ink }}>
            <ImagePlus size={12} /> add ground truth
            <input ref={gtInput} type="file" accept="image/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) onUploadGT(f); }} />
          </button>
        )}
        {result ? (
          <span style={{ display: "flex", alignItems: "center", gap: 7, color: T.green, fontFamily: MONO, fontWeight: 600, fontSize: 12 }}><Check size={15} /> Restored · <Mono style={{ color: T.dim, fontWeight: 400 }}>{result.timeMs.toFixed(0)} ms</Mono></span>
        ) : busy ? (
          <span style={{ color: T.input, fontFamily: MONO, fontWeight: 600, fontSize: 12 }}>Restoring…</span>
        ) : null}
      </div>
    </div>
  );
}

// Pre-restore center: the degraded image with the primary Restore action.
function InputPreview({ source, busy, onRestore }) {
  return (
    <div>
      <div style={{ position: "relative", height: 360, ...panel(T.stage, 5), overflow: "hidden", borderRadius: 4, display: "grid", placeItems: "center" }}>
        <img src={source.url} alt="degraded input" style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }} />
        <div style={{ position: "absolute", top: 10, left: 10 }}><Tag color={T.input}>{LAYER_LABEL.input}</Tag></div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10, marginTop: 18 }}>
        <button className="pixel-btn" onClick={onRestore} disabled={busy} style={{ padding: "13px 28px", fontSize: 15, background: T.yellow, color: T.ink }}>
          <Play size={17} /> {busy ? "Restoring…" : "Restore image"}
        </button>
        <div style={{ fontSize: 11.5, fontFamily: MONO, color: T.faint }}>Recover a clean image from this degraded input.</div>
      </div>
    </div>
  );
}

// ---- app -------------------------------------------------------------------
async function computeMetricsVsGT(candidateUrl, gtUrl, w, h) {
  const [cand, ref] = await Promise.all([urlToImageData(candidateUrl, w, h), urlToImageData(gtUrl, w, h)]);
  return { ...fullReferenceMetrics(cand.data, ref.data, w, h), source: "client" };
}

function SectionHead({ children, hint }) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
      <h2 style={{ fontFamily: DISPLAY, fontSize: 21, fontWeight: 700, letterSpacing: 0.5, textTransform: "uppercase", color: T.ink, margin: 0 }}>{children}</h2>
      {hint && <span style={{ fontFamily: MONO, fontSize: 11, color: T.dim }}>{hint}</span>}
    </div>
  );
}

export default function App() {
  const [samples, setSamples] = useState([]);
  const [source, setSource] = useState(null);
  const [baselineEnabled, setBaselineEnabled] = useState(true);
  const [result, setResult] = useState(null);
  const [baseline, setBaseline] = useState(null);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [baselineMetrics, setBaselineMetrics] = useState(null);
  const [stage, setStage] = useState("idle");
  const [error, setError] = useState(null);
  const runId = useRef(0);

  useEffect(() => { setSamples(buildSamplePairs()); }, []);

  const resetRun = useCallback(() => { setResult(null); setBaseline(null); setModelMetrics(null); setBaselineMetrics(null); setStage("idle"); setError(null); }, []);
  const selectSource = useCallback((src) => { resetRun(); setSource(src); }, [resetRun]);

  const onUploadInput = useCallback(async (file) => {
    const url = URL.createObjectURL(file);
    const img = await loadImageEl(url);
    selectSource({ id: `upload-${Date.now()}`, name: file.name, url, width: img.naturalWidth, height: img.naturalHeight, gtUrl: null, degradation: null, isSample: false, file });
  }, [selectSource]);

  const onUploadGT = useCallback(async (file) => { const gtUrl = URL.createObjectURL(file); setSource((prev) => prev ? { ...prev, gtUrl } : prev); resetRun(); }, [resetRun]);

  const run = useCallback(async (src, withBaseline) => {
    const id = ++runId.current; const stale = () => id !== runId.current;
    setError(null); setResult(null); setBaseline(null); setModelMetrics(null); setBaselineMetrics(null);
    try {
      setStage("preprocess");
      const res = await runPlaceholderRestore({ imageUrl: src.url, gtUrl: src.gtUrl, scaleFactor: src.scaleFactor });
      if (stale()) return;
      setStage("infer"); setResult(res);
      const w = res.outputWidth, h = res.outputHeight;
      if (res.metrics) setModelMetrics({ ...res.metrics, source: res.metrics.source || "backend" });
      else if (src.gtUrl) { setStage("metrics"); const m = await computeMetricsVsGT(res.restoredUrl, src.gtUrl, w, h); if (stale()) return; setModelMetrics(m); }
      if (withBaseline) {
        setStage("baseline");
        const base = res.baselineUrl ? { url: res.baselineUrl, width: w, height: h } : await buildBaseline({ imageUrl: src.url, gtUrl: src.gtUrl, scaleFactor: src.scaleFactor });
        if (stale()) return; setBaseline(base);
        if (src.gtUrl) { const bm = await computeMetricsVsGT(base.url, src.gtUrl, w, h); if (stale()) return; setBaselineMetrics(bm); }
      }
      if (!stale()) setStage("done");
    } catch (e) { if (!stale()) { setError(e); setStage("idle"); } }
  }, []);

  // If baseline is toggled on after a run, compute it live.
  useEffect(() => {
    if (!baselineEnabled || !result || baseline || !source) return;
    let alive = true;
    (async () => {
      const base = await buildBaseline({ imageUrl: source.url, gtUrl: source.gtUrl, scaleFactor: source.scaleFactor });
      if (!alive) return; setBaseline(base);
      if (source.gtUrl) { const bm = await computeMetricsVsGT(base.url, source.gtUrl, result.outputWidth, result.outputHeight); if (alive) setBaselineMetrics(bm); }
    })();
    return () => { alive = false; };
  }, [baselineEnabled, result, baseline, source]);

  const doRun = useCallback(() => { if (source) run(source, baselineEnabled); }, [source, baselineEnabled, run]);
  const busy = stage !== "idle" && stage !== "done";
  const hasGT = !!source?.gtUrl;
  const layers = [
    { key: "input", url: source?.url },
    { key: "output", url: result?.restoredUrl },
    { key: "gt", url: source?.gtUrl },
    ...(baselineEnabled ? [{ key: "baseline", url: baseline?.url }] : []),
  ];

  return (
    <div className="grid-dots" style={{ minHeight: "100vh", background: T.cream, color: T.ink, fontFamily: MONO }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Pixelify+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
        *{box-sizing:border-box} html,body{margin:0}
        .grid-dots{background-image:radial-gradient(rgba(33,28,18,.05) 1px, transparent 0);background-size:6px 6px}
        .grid-dots-navy{background-image:radial-gradient(rgba(255,255,255,.06) 1px, transparent 0);background-size:6px 6px}
        .hero-h1{font-family:${DISPLAY};font-weight:700;font-size:clamp(30px,6vw,54px);line-height:1.05;letter-spacing:1px;margin:0;color:${T.onNavy};text-transform:uppercase}
        .pixel-btn{border:${BORDER};border-radius:4px;box-shadow:4px 4px 0 ${T.ink};font-family:${MONO};font-weight:600;cursor:pointer;transition:transform .08s ease,box-shadow .08s ease;text-transform:uppercase;letter-spacing:.5px;display:inline-flex;align-items:center;justify-content:center;gap:8px}
        .pixel-btn:hover:not(:disabled){transform:translate(-1px,-1px);box-shadow:5px 5px 0 ${T.ink}}
        .pixel-btn:active:not(:disabled){transform:translate(3px,3px);box-shadow:1px 1px 0 ${T.ink}}
        .pixel-btn:disabled{opacity:.5;box-shadow:2px 2px 0 ${T.ink};cursor:not-allowed}
        .pixel-tab{border:${BORDER};border-radius:4px;box-shadow:3px 3px 0 ${T.ink};font-family:${MONO};font-weight:600;cursor:pointer;transition:transform .08s,box-shadow .08s;display:inline-flex;align-items:center;gap:6px;text-transform:uppercase;letter-spacing:.4px}
        .pixel-tab:hover{transform:translate(-1px,-1px);box-shadow:4px 4px 0 ${T.ink}}
        .pixel-tab:active{transform:translate(2px,2px);box-shadow:1px 1px 0 ${T.ink}}
        select.pixel-select{border:${BORDER};border-radius:4px;font-family:${MONO};font-weight:600;background:${T.paper};color:${T.ink};padding:6px 8px;font-size:11px;cursor:pointer;text-transform:none}
        ::-webkit-scrollbar{width:12px;height:12px}
        ::-webkit-scrollbar-track{background:${T.cream}}
        ::-webkit-scrollbar-thumb{background:${T.ink};border:3px solid ${T.cream};border-radius:2px}
        :focus-visible{outline:3px solid ${T.navy};outline-offset:2px}
        @media (prefers-reduced-motion:reduce){.pixel-btn,.pixel-tab{transition:none}}
        @media (max-width:900px){.rest-main{grid-template-columns:1fr !important}}`}</style>

      <Header hasSource={!!source} onNew={() => selectSource(null)} />

      {!source ? (
        <Hero onUploadInput={onUploadInput} samples={samples} onPickSample={selectSource} />
      ) : (
        <main className="rest-main" style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 330px", gap: 18, padding: 20, alignItems: "start", maxWidth: 1200, margin: "0 auto" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 16, minWidth: 0 }}>
            <SourceBar source={source} result={result} busy={busy} onUploadGT={onUploadGT} />
            <PlaceholderBanner result={result} />
            {error && <ErrorNote>Restoration failed: {error.message}</ErrorNote>}
            {result ? (
              <>
                <SectionHead hint="before / after · zoom to inspect">Restored result</SectionHead>
                <CompareViewer layers={layers} />
                <PipelineBar stage={stage} result={result} />
              </>
            ) : (
              <>
                <SectionHead hint={busy ? "running…" : "ready to restore"}>Degraded input</SectionHead>
                <InputPreview source={source} busy={busy} onRestore={doRun} />
                {busy && <PipelineBar stage={stage} result={result} />}
              </>
            )}
          </div>
          <ResultsPanel result={result} modelMetrics={modelMetrics} baselineMetrics={baselineMetrics} hasGT={hasGT} baselineEnabled={baselineEnabled} onToggleBaseline={() => setBaselineEnabled((v) => !v)} busy={busy} />
        </main>
      )}
    </div>
  );
}
