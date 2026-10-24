import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
OUT = "docs/figures/"
C1, C2, C3, GR = "#1f6feb", "#d1481f", "#2a9d5c", "#6e7781"

# ---------- Fig 1: project overview ----------
fig, ax = plt.subplots(figsize=(12, 4.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4.6)
def box(x, y, w, h, title, lines, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=col + "18", ec=col, lw=1.8))
    ax.text(x + w/2, y + h - 0.32, title, ha="center", va="center", weight="bold", color=col, fontsize=12.5)
    for i, l in enumerate(lines):
        ax.text(x + 0.18, y + h - 0.85 - i*0.43, l, ha="left", va="center", fontsize=9.6)
box(0.2, 0.7, 3.5, 3.2, "STEP 1  ·  Delay stage", ["Thorlabs DDS220/M (rack, BBD30x)", "pylablib KinesisMotor", "ps ⇄ mm conversion, t₀", "homing guard, retry lock", "→ delay_stage.py"], C1)
box(4.25, 0.7, 3.5, 3.2, "STEP 2  ·  + Lock-in", ["Zurich Instruments MFLI", "zhinst ziDAQServer (API lvl 6)", "move → settle → acquire", "n scans, running mean", "→ lockin.py, scan_engine.py"], C2)
box(8.3, 0.7, 3.5, 3.2, "STEP 3  ·  + GUI", ["PyQt5 + matplotlib", "ScanWorker on a QThread", "live decay / residual / heatmap", "calibration, metrics, saving", "→ Scan_Engine_Gui.py"], C3)
for x0, x1 in [(3.75, 4.2), (7.8, 8.25)]:
    ax.add_patch(FancyArrowPatch((x0, 2.3), (x1, 2.3), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=GR))
ax.text(6, 4.35, "Building an optical-Kerr-effect (OKE) pump–probe acquisition platform in Python", ha="center", fontsize=13, weight="bold")
ax.text(6, 0.25, "Each step wraps the previous one — nothing below it is re-implemented", ha="center", fontsize=10, color=GR, style="italic")
plt.savefig(OUT + "fig1_overview.png"); plt.close()

# ---------- Fig 2: ps <-> mm ----------
MM_PER_PS = 0.29979/2; MAXMM = 220.0
t0 = MAXMM / MM_PER_PS
t = np.linspace(-150, t0 + 150, 400)
mm = (t0 - t) * MM_PER_PS
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.plot(t, mm, color=C1, lw=2.2, label=r"$\mathrm{mm}=(t_0-t)\cdot 0.149895$")
ax.axhspan(-30, 0, color=C2, alpha=.12); ax.axhspan(MAXMM, 250, color=C2, alpha=.12)
ax.axhline(0, color=C2, lw=1.2, ls="--"); ax.axhline(MAXMM, color=C2, lw=1.2, ls="--")
ax.text(t0 - 330, -22, "MIN_MM = 0 (home)", ha="right", color=C2, fontsize=9.5)
ax.text(t0 + 140, MAXMM + 6, "MAX_MM = 220 (end of travel)", ha="right", color=C2, fontsize=9.5)
ax.plot([0], [MAXMM], "o", color=C3, ms=9); ax.annotate("t = 0 ps  →  220 mm\n(default T0_POSITION_MM)", (0, MAXMM), (180, 168), arrowprops=dict(arrowstyle="->", color=C3), color=C3, fontsize=9.5)
ax.plot([t0], [0], "o", color=C3, ms=9); ax.annotate(f"t = {t0:.1f} ps  →  0 mm", (t0, 0), (t0-760, 90), arrowprops=dict(arrowstyle="->", color=C3), color=C3, fontsize=9.5)
ax.set_xlabel("pump–probe delay  t  (ps)"); ax.set_ylabel("stage position (mm)"); ax.set_ylim(-30, 250)
ax.set_title("Time → position: double-pass geometry, clamped to the physical travel", fontsize=11.5)
ax.legend(loc="upper right", bbox_to_anchor=(1, .86), frameon=False)
plt.savefig(OUT + "fig2_ps_to_mm.png"); plt.close()

# ---------- Fig 3: settle detection (schematic) ----------
rng = np.random.default_rng(3)
ts = np.arange(0, 0.30, 0.008)
pos = 10 + 4*(1 - np.exp(-ts/0.04)) * np.cos(0.0) + 0.0
pos = 10 + 2*(1 - np.exp(-ts/0.02)) + 0.00015*rng.standard_normal(len(ts))
tol = 0.0005
fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.plot(ts*1e3, pos, "o-", color=C1, ms=4, lw=1.4, label="get_position_mm()")
dif = np.abs(np.diff(pos)); run = 0; hit = None
for i, d in enumerate(dif):
    run = run + 1 if d <= tol else 0
    if run >= 3: hit = i + 1; break
ax.axhspan(pos[hit]-tol, pos[hit]+tol, color=C3, alpha=.2, label=f"± tolerance_mm = {tol}")
ax.axvline(ts[hit]*1e3, color=C3, ls="--"); ax.text(ts[hit]*1e3 + 6, 10.7, "3 consecutive reads\nwithin tolerance → settled", color=C3, fontsize=9.5)
ax.text(8, 11.5, "stage still moving:\nposition keeps changing", color=C2, fontsize=9.5)
ax.set_xlabel("time after move command (ms), poll every 8 ms"); ax.set_ylabel("position (mm)")
ax.set_title("ScanEngine.wait_for_settle() — schematic", fontsize=11.5); ax.legend(frameon=False, loc="lower right")
plt.savefig(OUT + "fig3_settle.png"); plt.close()

# ---------- Fig 4: 3-section grid (same logic as generate_section_points) ----------
def gen(start, stop, step, points, scale):
    if scale == "Log":
        return list(np.round(np.logspace(np.log10(start), np.log10(stop), int(points)), 4))
    n = int(round((stop-start)/step)) + 1
    return list(np.round(np.linspace(start, stop, n), 4))
secs = [(-1, 1, 0.1, 0, "Linear"), (1, 10, 1, 0, "Linear"), (10, 1000, 0, 20, "Log")]
sp = [gen(*s) for s in secs]; allt = sorted(set(round(x, 4) for s in sp for x in s))
cols = [C1, C2, C3]
fig, ax = plt.subplots(figsize=(9.5, 2.2))
for i, s in enumerate(sp):
    ax.plot(s, np.zeros(len(s)), "|", ms=26, mew=1.8, color=cols[i], label=f"Section {i+1}: {secs[i][4]} ({len(s)} pts)")
ax.set_xscale("symlog", linthresh=1); ax.set_ylim(-.5, .5); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("delay (ps) — symlog axis"); ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.45), fontsize=9.5)
ax.set_title(f"Example 3-section scan  →  {len(allt)} unique points after dedup", y=1.42, fontsize=11.5)
plt.savefig(OUT + "fig4_sections.png"); plt.close()

# ---------- Fig 5: averaging (synthetic) ----------
tt = np.linspace(-1, 12, 131)
def clean(t):
    g = np.exp(-(t/0.5)**2)
    d = np.where(t > 0, np.exp(-t/2.5), 0.0) * 0.45
    return g + d
sig = clean(tt); nscan = 20
scans = np.array([sig + 0.35*rng.standard_normal(len(tt)) for _ in range(nscan)])
fig = plt.figure(figsize=(12, 4.4)); gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1.0], wspace=.32)
a = fig.add_subplot(gs[0]); a.plot(tt, scans[0], color="#aaaaaa", lw=1, label="single scan (n=1)")
a.plot(tt, scans[:5].mean(0), color=C2, lw=1.3, alpha=.9, label="running mean, n=5")
a.plot(tt, scans.mean(0), color=C1, lw=2, label="running mean, n=20")
a.set_xlabel("delay (ps)"); a.set_ylabel("lock-in X (a.u.)"); a.set_title("Kinetics panel", fontsize=11); a.legend(frameon=False, fontsize=9)
b = fig.add_subplot(gs[1]); b.plot(tt, scans[-1] - scans.mean(0), color=C3, lw=1); b.axhline(0, color="k", lw=.6)
b.set_xlabel("delay (ps)"); b.set_title("Residual = latest − average", fontsize=11)
c = fig.add_subplot(gs[2]); im = c.imshow(scans, aspect="auto", origin="lower", extent=[tt[0], tt[-1], 1, nscan], cmap="RdBu_r")
c.set_xlabel("delay (ps)"); c.set_ylabel("scan number"); c.set_title("2D heatmap (Mode 1)", fontsize=11)
c.spines["top"].set_visible(True); c.spines["right"].set_visible(True); fig.colorbar(im, ax=c, fraction=.05, pad=.03)
fig.suptitle("SYNTHETIC data — illustrates what the three live plots show and why repeated scans are averaged", fontsize=10.5, color=GR, y=1.02)
plt.savefig(OUT + "fig5_averaging.png"); plt.close()

# ---------- Fig 6: per-point acquisition semantics ----------
fig, ax = plt.subplots(figsize=(8.6, 3.5))
k = np.arange(1, 11); vals = 1.0 + 0.06*rng.standard_normal(10)
ax.plot(k, vals, "o", color=GR, ms=7, label="readings 1…N (all stored raw)")
ax.plot([10], [vals[-1]], "o", color=C2, ms=12, mfc="none", mew=2.4, label="'final' = reading N (not an average)")
ax.set_xticks(k); ax.set_xlabel("getSample() call at one settled delay point"); ax.set_ylabel("X (a.u.)")
ax.set_title("ScanEngine.acquire_point(n_readings=10)", fontsize=11.5); ax.legend(frameon=False, fontsize=9.5, loc="lower left")
plt.savefig(OUT + "fig6_acquire_point.png"); plt.close()
print("ok", len(allt), t0)
