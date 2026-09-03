import { createFileRoute, ClientOnly } from "@tanstack/react-router";
import { lazy, Suspense } from "react";

const UgvVision = lazy(() => import("@/components/UgvVision"));

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "UGV Vision Pilot — Webcam Obstacle Avoidance" },
      {
        name: "description",
        content:
          "Live webcam object detection drives a simulated UGV: clear path goes forward, blocked zones trigger turns, full block stops the rover.",
      },
      { property: "og:title", content: "UGV Vision Pilot — Webcam Obstacle Avoidance" },
      {
        property: "og:description",
        content:
          "Real-time in-browser object detection, path-blocked analysis, and a simulated unmanned ground vehicle reacting live.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

const PIPELINE = ["WEBCAM", "OBJECT DETECTION", "PATH ANALYSIS", "DECISION", "SIMULATED UGV"];

function Index() {
  return (
    <main className="min-h-screen bg-background px-4 py-8 font-mono text-foreground md:px-8">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="space-y-3">
          <p className="text-[11px] tracking-[0.35em] text-accent">AUTONOMY CONSOLE / V1</p>
          <h1 className="text-3xl font-bold tracking-tight md:text-5xl">UGV Vision Pilot</h1>
          <p className="max-w-2xl text-sm text-muted-foreground">
            Your camera feed is split into three corridors. Detected objects mark corridors as blocked, and the
            decision engine steers a simulated unmanned ground vehicle in real time.
          </p>
          <ol className="flex flex-wrap items-center gap-2 text-[10px] tracking-widest text-muted-foreground">
            {PIPELINE.map((step, i) => (
              <li key={step} className="flex items-center gap-2">
                <span className="rounded border border-border px-2 py-1">{step}</span>
                {i < PIPELINE.length - 1 && <span className="text-accent">↓</span>}
              </li>
            ))}
          </ol>
        </header>

        <ClientOnly fallback={<div className="panel h-[420px] animate-pulse" />}>
          <Suspense fallback={<div className="panel h-[420px] animate-pulse" />}>
            <UgvVision />
          </Suspense>
        </ClientOnly>
      </div>
    </main>
  );
}
