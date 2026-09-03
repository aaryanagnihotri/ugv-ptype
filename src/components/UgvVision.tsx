import { useEffect, useRef, useState, useCallback } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs";

type Decision = "FORWARD" | "LEFT" | "RIGHT" | "STOP" | "IDLE";

type Zones = { left: number; center: number; right: number };
type Blocked = { left: boolean; center: boolean; right: boolean };

// Hysteresis: a corridor becomes blocked above BLOCK_ON and only clears below
// BLOCK_OFF, so a flickering detection can't make the rover twitch.
const BLOCK_ON = 0.1;
const BLOCK_OFF = 0.05;
const SMOOTHING = 0.55; // exponential smoothing on the per-zone risk score
const MIN_SCORE = 0.4;
const MAX_BOXES = 20;
const STOP_LATCH_MS = 900; // once stopped, hold before resuming

// Objects that occupy real floor space in front of a ground vehicle.
const HARD_OBSTACLES = new Set([
  "person","bicycle","car","motorcycle","bus","truck","train","dog","cat","horse","sheep","cow",
  "bench","chair","couch","bed","dining table","potted plant","toilet","tv","refrigerator","suitcase",
  "backpack","sports ball","skateboard","fire hydrant","stop sign","traffic light","vase","laptop","book","bottle",
]);

/** Per-detection risk: bigger + lower in frame (closer to the rover) = riskier. */
function riskOf(p: cocoSsd.DetectedObject, w: number, h: number) {
  const [, by, bw, bh] = p.bbox;
  const baseY = (by + bh) / h; // 1 = at the rover's wheels
  const proximity = Math.pow(Math.max(0, Math.min(1, baseY)), 1.6);
  const size = (bw * bh) / (w * h);
  const height = bh / h; // apparent height is the strongest distance cue
  const classWeight = HARD_OBSTACLES.has(p.class) ? 1 : 0.55;
  return { risk: (size * 0.6 + height * 0.4) * proximity * classWeight * p.score, proximity };
}

function decide(b: Blocked): Decision {
  if (b.left && b.center && b.right) return "STOP";
  if (!b.left && !b.center && !b.right) return "FORWARD";
  if (b.center) {
    if (!b.right) return "RIGHT";
    if (!b.left) return "LEFT";
    return "STOP";
  }
  if (b.left) return "RIGHT";
  if (b.right) return "LEFT";
  return "FORWARD";
}

export default function UgvVision() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const stateRef = useRef({ decision: "IDLE" as Decision, heading: 0, x: 0.5, y: 0.82, trail: [] as [number, number][] });
  const smoothRef = useRef<Zones>({ left: 0, center: 0, right: 0 });
  const blockedRef = useRef<Blocked>({ left: false, center: false, right: false });
  const stopUntilRef = useRef(0);

  const [status, setStatus] = useState("Loading detection model…");
  const [running, setRunning] = useState(false);
  const [decision, setDecision] = useState<Decision>("IDLE");
  const [zones, setZones] = useState<Zones>({ left: 0, center: 0, right: 0 });
  const [blocked, setBlocked] = useState<Blocked>({ left: false, center: false, right: false });
  const [objects, setObjects] = useState<{ label: string; score: number; proximity: number }[]>([]);
  const [fps, setFps] = useState(0);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);

  const [modelReady, setModelReady] = useState(false);
  const [loadAttempt, setLoadAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        for (const b of ["webgl", "cpu"]) {
          try {
            await tf.setBackend(b);
            break;
          } catch {
            /* try next backend */
          }
        }
        await tf.ready();
        setStatus("Loading detection model weights…");
        // Try the higher-accuracy net first, fall back to the lite net if its
        // weights can't be fetched (blocked CDN / flaky network).
        let m: cocoSsd.ObjectDetection | null = null;
        let lastErr: unknown = null;
        for (const base of ["mobilenet_v2", "lite_mobilenet_v2"] as const) {
          try {
            m = await cocoSsd.load({ base });
            break;
          } catch (e) {
            lastErr = e;
          }
        }
        if (!m) throw lastErr ?? new Error("weights unavailable");
        if (cancelled) return;
        // warm up so the first real frame isn't a multi-second stall
        const warm = document.createElement("canvas");
        warm.width = warm.height = 300;
        warm.getContext("2d")!.fillRect(0, 0, 300, 300);
        await m.detect(warm);
        if (cancelled) return;
        modelRef.current = m;
        setModelReady(true);
        setStatus(`Model ready (${tf.getBackend()}) — start the camera`);
      } catch (e) {
        setModelReady(false);
        setStatus(`Model failed to load: ${(e as Error)?.message ?? String(e)} — tap RETRY`);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loadAttempt]);

  const start = useCallback(async () => {
    const attempts: MediaStreamConstraints[] = [
      { video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false },
      { video: true, audio: false },
    ];
    let lastErr: unknown = null;
    for (const c of attempts) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(c);
        const v = videoRef.current!;
        v.srcObject = stream;
        await v.play();
        smoothRef.current = { left: 0, center: 0, right: 0 };
        blockedRef.current = { left: false, center: false, right: false };
        setRunning(true);
        setStatus("Live — analysing path");
        return;
      } catch (e) {
        lastErr = e;
      }
    }
    setStatus(`Camera unavailable: ${(lastErr as Error)?.message ?? String(lastErr)}`);
  }, []);


  const stop = useCallback(() => {
    const v = videoRef.current;
    const s = v?.srcObject as MediaStream | null;
    s?.getTracks().forEach((t) => t.stop());
    if (v) v.srcObject = null;
    setRunning(false);
    setDecision("IDLE");
    stateRef.current.decision = "IDLE";
    setStatus("Stopped");
  }, []);

  useEffect(() => () => stop(), [stop]);

  // Detection + render loop
  useEffect(() => {
    if (!running) return;
    let lastDetect = 0;
    let frames = 0;
    let fpsMark = performance.now();
    let alive = true;
    let busy = false;

    const loop = async (t: number) => {
      if (!alive) return;
      const v = videoRef.current;
      const cvs = overlayRef.current;
      if (v && cvs && v.videoWidth) {
        const w = (cvs.width = v.videoWidth);
        const h = (cvs.height = v.videoHeight);
        const ctx = cvs.getContext("2d")!;
        ctx.clearRect(0, 0, w, h);

        if (modelRef.current && !busy && t - lastDetect > 120) {
          busy = true;
          lastDetect = t;
          try {
            const preds = await modelRef.current.detect(v, MAX_BOXES, MIN_SCORE);
            const raw: Zones = { left: 0, center: 0, right: 0 };
            const third = w / 3;
            const scored = preds.map((p) => ({ p, ...riskOf(p, w, h) }));
            for (const { p, risk } of scored) {
              const [bx, , bw] = p.bbox;
              const bands: [keyof Zones, number, number][] = [
                ["left", 0, third],
                ["center", third, third * 2],
                ["right", third * 2, w],
              ];
              for (const [key, s, e] of bands) {
                const ov = Math.max(0, Math.min(bx + bw, e) - Math.max(bx, s));
                if (ov > 0) raw[key] += risk * (ov / bw) * 3;
              }
            }

            // temporal smoothing + hysteresis
            const sm = smoothRef.current;
            const bl = blockedRef.current;
            (Object.keys(raw) as (keyof Zones)[]).forEach((k) => {
              sm[k] = sm[k] * SMOOTHING + raw[k] * (1 - SMOOTHING);
              bl[k] = bl[k] ? sm[k] > BLOCK_OFF : sm[k] > BLOCK_ON;
            });

            let d = decide(bl);
            if (d === "STOP") stopUntilRef.current = t + STOP_LATCH_MS;
            else if (t < stopUntilRef.current) d = "STOP";

            setZones({ ...sm });
            setBlocked({ ...bl });
            setObjects(scored.map(({ p, proximity }) => ({ label: p.class, score: p.score, proximity })));
            setDecision(d);
            stateRef.current.decision = d;
            (cvs as HTMLCanvasElement & { _scored?: typeof scored })._scored = scored;
          } finally {
            busy = false;
          }
        }

        // draw zone guides + boxes
        const third = w / 3;
        const bl = blockedRef.current;
        [bl.left, bl.center, bl.right].forEach((isBlocked, i) => {
          ctx.fillStyle = isBlocked ? "rgba(255,64,64,0.16)" : "rgba(64,255,170,0.07)";
          ctx.fillRect(i * third, 0, third, h);
          ctx.strokeStyle = "rgba(255,255,255,0.18)";
          ctx.strokeRect(i * third, 0, third, h);
        });
        const scored =
          (cvs as HTMLCanvasElement & { _scored?: { p: cocoSsd.DetectedObject; risk: number; proximity: number }[] })
            ._scored ?? [];
        ctx.lineWidth = 2;
        ctx.font = "600 14px ui-monospace, monospace";
        for (const { p, risk, proximity } of scored) {
          const [x, y, bw, bh] = p.bbox;
          const hot = risk > BLOCK_ON;
          ctx.strokeStyle = hot ? "#ff4646" : "#ffb800";
          ctx.strokeRect(x, y, bw, bh);
          ctx.fillStyle = hot ? "#ff4646" : "#ffb800";
          ctx.fillText(
            `${p.class} ${(p.score * 100) | 0}% · near ${(proximity * 100) | 0}%`,
            x + 4,
            Math.max(14, y - 6),
          );
        }

        frames++;
        if (t - fpsMark > 1000) {
          setFps(Math.round((frames * 1000) / (t - fpsMark)));
          frames = 0;
          fpsMark = t;
        }
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      alive = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [running]);

  // UGV simulation loop
  useEffect(() => {
    let alive = true;
    let raf = 0;
    let last = performance.now();
    const draw = (t: number) => {
      if (!alive) return;
      const dt = Math.min(0.05, (t - last) / 1000);
      last = t;
      const cvs = simRef.current;
      if (cvs) {
        const w = (cvs.width = cvs.clientWidth * 2);
        const h = (cvs.height = cvs.clientHeight * 2);
        const ctx = cvs.getContext("2d")!;
        const s = stateRef.current;

        const speed = s.decision === "FORWARD" ? 0.16 : s.decision === "STOP" || s.decision === "IDLE" ? 0 : 0.06;
        if (s.decision === "LEFT") s.heading -= dt * 1.6;
        if (s.decision === "RIGHT") s.heading += dt * 1.6;
        s.x += Math.sin(s.heading) * speed * dt;
        s.y -= Math.cos(s.heading) * speed * dt;
        s.x = Math.min(0.94, Math.max(0.06, s.x));
        if (s.y < 0.08) {
          s.y = 0.9;
          s.trail = [];
        }
        s.trail.push([s.x, s.y]);
        if (s.trail.length > 260) s.trail.shift();

        ctx.fillStyle = "#080c0a";
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = "rgba(64,255,170,0.10)";
        ctx.lineWidth = 2;
        for (let i = 0; i <= 12; i++) {
          ctx.beginPath();
          ctx.moveTo((i / 12) * w, 0);
          ctx.lineTo((i / 12) * w, h);
          ctx.moveTo(0, (i / 12) * h);
          ctx.lineTo(w, (i / 12) * h);
          ctx.stroke();
        }

        ctx.strokeStyle = "rgba(64,255,170,0.55)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        s.trail.forEach(([tx, ty], i) => (i ? ctx.lineTo(tx * w, ty * h) : ctx.moveTo(tx * w, ty * h)));
        ctx.stroke();

        const px = s.x * w;
        const py = s.y * h;
        ctx.save();
        ctx.translate(px, py);
        ctx.rotate(s.heading);
        // sensor cone
        ctx.fillStyle = s.decision === "STOP" ? "rgba(255,70,70,0.18)" : "rgba(64,255,170,0.14)";
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.arc(0, 0, h * 0.26, -Math.PI / 2 - 0.5, -Math.PI / 2 + 0.5);
        ctx.closePath();
        ctx.fill();
        // chassis
        ctx.fillStyle = "#e6fff4";
        ctx.fillRect(-14, -20, 28, 40);
        ctx.fillStyle = "#1d2a24";
        ctx.fillRect(-20, -18, 8, 36);
        ctx.fillRect(12, -18, 8, 36);
        ctx.fillStyle = s.decision === "STOP" ? "#ff4646" : "#40ffaa";
        ctx.fillRect(-6, -26, 12, 8);
        ctx.restore();
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => {
      alive = false;
      cancelAnimationFrame(raf);
    };
  }, []);

  const zoneRows: [string, number, boolean][] = [
    ["LEFT", zones.left, blocked.left],
    ["CENTER", zones.center, blocked.center],
    ["RIGHT", zones.right, blocked.right],
  ];

  return (
    <div className="grid gap-4 lg:grid-cols-[1.3fr_1fr]">
      <section className="panel">
        <header className="panel-head">
          <span>01 · WEBCAM / OBJECT DETECTION</span>
          <span className="text-accent">{fps} FPS</span>
        </header>
        <div className="relative aspect-[4/3] w-full overflow-hidden bg-black">
          <video ref={videoRef} playsInline muted className="absolute inset-0 h-full w-full object-cover" />
          <canvas ref={overlayRef} className="absolute inset-0 h-full w-full" />
          {!running && (
            <div className="absolute inset-0 grid place-items-center bg-black/70 text-center text-xs tracking-widest text-muted-foreground">
              <p>{status}</p>
            </div>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-3 border-t border-border p-3">
          <button className="btn-hud" onClick={running ? stop : start} disabled={!modelReady}>
            {running ? "STOP CAMERA" : "START CAMERA"}
          </button>
          {!modelReady && (
            <button
              className="btn-hud"
              onClick={() => {
                setStatus("Retrying model download…");
                setLoadAttempt((n) => n + 1);
              }}
            >
              RETRY MODEL
            </button>
          )}
          <span className="text-xs text-muted-foreground">{status}</span>
        </div>

      </section>

      <div className="grid gap-4">
        <section className="panel">
          <header className="panel-head">
            <span>02 · IS PATH BLOCKED?</span>
          </header>
          <div className="space-y-3 p-4">
            {zoneRows.map(([name, val, isBlocked]) => {
              const blocked = isBlocked;
              return (
                <div key={name}>
                  <div className="flex justify-between text-[11px] tracking-widest">
                    <span>{name}</span>
                    <span className={blocked ? "text-destructive" : "text-accent"}>
                      {blocked ? "BLOCKED" : "CLEAR"}
                    </span>
                  </div>
                  <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-secondary">
                    <div
                      className={`h-full transition-all ${blocked ? "bg-destructive" : "bg-accent"}`}
                      style={{ width: `${Math.min(100, (val / BLOCK_ON) * 60)}%` }}
                    />
                  </div>
                </div>
              );
            })}
            <div className="mt-2 border-t border-border pt-3 text-[11px] leading-relaxed text-muted-foreground">
              {objects.length ? (
                <ul className="flex flex-wrap gap-2">
                  {objects.slice(0, 6).map((o, i) => (
                    <li key={i} className="rounded border border-border px-2 py-0.5">
                      {o.label} · {(o.score * 100) | 0}% · near {(o.proximity * 100) | 0}%
                    </li>
                  ))}
                </ul>
              ) : (
                "No objects in frame"
              )}
            </div>
          </div>
        </section>

        <section className="panel">
          <header className="panel-head">
            <span>03 · DECISION</span>
          </header>
          <div className="grid grid-cols-2 gap-2 p-4">
            {(["FORWARD", "LEFT", "RIGHT", "STOP"] as Decision[]).map((d) => (
              <div key={d} className={`decision-cell ${decision === d ? "is-active" : ""}`}>
                {d}
              </div>
            ))}
          </div>
          <p className="border-t border-border px-4 py-2 text-[11px] text-muted-foreground">
            Clear → FORWARD · Left blocked → RIGHT · Right blocked → LEFT · All blocked → STOP
          </p>
        </section>
      </div>

      <section className="panel lg:col-span-2">
        <header className="panel-head">
          <span>04 · SIMULATED UGV</span>
          <span className={decision === "STOP" ? "text-destructive" : "text-accent"}>CMD: {decision}</span>
        </header>
        <canvas ref={simRef} className="h-[340px] w-full" />
      </section>
    </div>
  );
}
