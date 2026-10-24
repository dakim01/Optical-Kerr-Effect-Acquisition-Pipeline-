# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:50:19 2026

@author: Lab
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:02:37 2026

@author: Lab
"""

# -*- coding: utf-8 -*-


import sys
import os
import json
import time
import threading

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib as mpl


def _find_and_add_oke_system_root():
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = here
    for _ in range(6):  # don't walk all the way to the drive root
        if os.path.isdir(os.path.join(candidate, "oke_system")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return None

_oke_root = _find_and_add_oke_system_root()
if _oke_root is None:
    print(
        "WARNING: could not find an 'oke_system' folder by walking up "
        f"from {os.path.dirname(os.path.abspath(__file__))}. "
        "Either move this GUI file so oke_system/ is a sibling or "
        "ancestor-sibling folder, or add the oke_system project root to "
        "sys.path manually before this import."
    )
# --------------------------------------------------------------------------

from oke_system.hardware.delay_stage import DelayStage, StageNotHomedError
from oke_system.hardware.lockin import MFLI
from oke_system.acquisition.scan_engine import (
    ScanEngine, ScanAborted, save_scan_ascii, save_full_run,
)




def find_peak(times_ps, values):
    """
    Locate the peak (max |value|) in a trace.

    Args:
        times_ps: array-like, delay axis (ps)
        values: array-like, same length, signal axis (e.g. running-mean R)

    Returns:
        {"peak_time_ps": float, "peak_value": float, "peak_index": int}
        or None if fewer than 1 point is available.
    """
    times_ps = np.asarray(times_ps, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    idx = int(np.argmax(np.abs(values)))
    return {
        "peak_time_ps": float(times_ps[idx]),
        "peak_value": float(values[idx]),
        "peak_index": idx,
    }


def estimate_fwhm(times_ps, values, peak_index=None):
    """
    Estimate Full Width at Half Maximum around a peak, via linear
    interpolation between the two bracketing samples on each side where
    the signal crosses half the peak's (baseline-subtracted) height.

    This is a straightforward, dependency-free FWHM estimator suitable
    for a live-updating display -- it is NOT a fit (no Gaussian/sech^2
    deconvolution), and intentionally so: a live dashboard value should
    update every point without waiting on an iterative fit to converge.
    For a publication-quality time-zero/pulse-width number, fit the
    saved ASCII data offline instead.

    Args:
        times_ps, values: same as find_peak()
        peak_index: index of the peak; computed via find_peak() if None

    Returns:
        float (FWHM in ps) or None if it cannot be estimated (e.g. too
        few points, or the signal never drops to half-max on one side
        within the available data).
    """
    times_ps = np.asarray(times_ps, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 3:
        return None

    if peak_index is None:
        pk = find_peak(times_ps, values)
        if pk is None:
            return None
        peak_index = pk["peak_index"]

    baseline = float(np.min(values))
    peak_height = float(values[peak_index]) - baseline
    if peak_height <= 0:
        return None
    half = baseline + peak_height / 2.0

    def _crossing(i_start, i_end, step):
        i = i_start
        while i != i_end:
            v_here = values[i]
            v_next = values[i + step]
            crosses_down = (v_here >= half) and (v_next < half)
            crosses_up = (v_here < half) and (v_next >= half)
            if crosses_down or crosses_up:
                t_here, t_next = times_ps[i], times_ps[i + step]
                if v_next == v_here:
                    return t_here
                frac = (half - v_here) / (v_next - v_here)
                return t_here + frac * (t_next - t_here)
            i += step
        return None

    t_left = _crossing(peak_index, 0, -1)
    t_right = _crossing(peak_index, n - 1, +1)

    if t_left is None or t_right is None:
        return None

    return float(abs(t_right - t_left))


def estimate_time_zero(times_ps, values):
    
    pk = find_peak(times_ps, values)
    if pk is None:
        return None
    return pk["peak_time_ps"]


def compute_snr(signal_values, noise_values=None):
    
    signal_values = np.asarray(signal_values, dtype=float)
    if len(signal_values) == 0:
        return {"signal": 0.0, "noise": 0.0, "snr": 0.0}

    signal = float(np.mean(signal_values))

    if noise_values is not None:
        noise_values = np.asarray(noise_values, dtype=float)
        noise = float(np.std(noise_values))
    else:
        noise = float(np.std(signal_values))

    snr = float("inf") if noise == 0.0 else abs(signal) / noise
    return {"signal": signal, "noise": noise, "snr": snr}


def compute_drift_pct(reference_value, current_value):
    
    if reference_value == 0:
        return None
    return 100.0 * (current_value - reference_value) / abs(reference_value)


# ==========================================================================
# Multi-section ("t0->t1->t2->t3") scan point generation -- ported
# directly from the user's standalone generate_timefile.py Tkinter tool
# (generate_section() / build_all_times() there), so a 3-section scan
# built here produces IDENTICAL time points to a .tf file generated by
# that tool, and a loaded .tfmeta.json round-trips exactly.
# ==========================================================================

def generate_section_points(start, stop, step, points, scale):
    
    if scale == "Log":
        if start <= 0 or stop <= 0:
            raise ValueError(
                f"Log scale needs Start and Stop both > 0 (got {start} -> {stop})."
            )
        n = int(points)
        if n < 2:
            raise ValueError("Log scale needs at least 2 points.")
        return list(np.round(np.logspace(np.log10(start), np.log10(stop), n), 4))
    else:
        if step <= 0:
            raise ValueError("Step must be > 0 for a Linear section.")
        n_steps = int(round((stop - start) / step)) + 1
        return list(np.round(np.linspace(start, stop, n_steps), 4))


def build_multi_section_points(section_params):
    
    sections = [generate_section_points(*p) for p in section_params]
    all_times = sorted(set(round(t, 4) for sec in sections for t in sec))
    return all_times, sections


# ==========================================================================
class ScanWorker(QObject):
  

    connected = pyqtSignal(bool, str)
    homed = pyqtSignal(bool, str)
    demod_configured = pyqtSignal(bool, str, float)   # success, message, actual_rate_hz

    # One emission per completed delay point (within any scan repeat):
    point_done = pyqtSignal(dict)   # the raw event dict from run_averaged_scan()
    scan_repeat_done = pyqtSignal(int, int)   # scan_index (0-based), scan_total
    averaging_finished = pyqtSignal(bool, str)   # completed, message

    log = pyqtSignal(str)
    save_finished = pyqtSignal(bool, str, str)  # success, message, dest_folder
    t0_changed = pyqtSignal(float)   # new t0_ps, after define_home_here() (jog_time_zero() no longer touches t0_ps -- see its docstring)

    calib_point_done = pyqtSignal(dict)   # {"commanded_ps","actual_mm","index","total"}
    calib_finished = pyqtSignal(bool, str)   # completed, message
    phase_zeroed = pyqtSignal(dict)   # MFLI.zero_phase()'s return dict

    status_update = pyqtSignal(dict)
    

    def __init__(self):
        super().__init__()
        self.stage = None
        self.lockin = None
        self.engine = None
        self._results = []   # flat list of all "point" events from the latest Control-tab run
        self._calib_abort_requested = False
        
        self._stage_busy = False

    # ------------------------------------------------------------------
    @pyqtSlot(str, float, str)
    def connect_hardware(self, serial, t0_ps, device_id):
        try:
            
            self.stage = DelayStage(serial=serial, t0_ps=t0_ps,
                                     is_rack_system=True)
            self.stage.connect()
            self.lockin = MFLI(device_id=device_id, interface="1GbE")
            self.lockin.connect()
            self.engine = ScanEngine(stage=self.stage, lockin=self.lockin, demod_index=0)
            self.connected.emit(True, "Connected: stage + lock-in")
        except Exception as e:
            self.connected.emit(False, f"Connection failed: {e}")

    @pyqtSlot()
    def home_stage(self):
        if self.stage is None:
            self.homed.emit(False, "Not connected")
            return
        self._stage_busy = True
        try:
            self.stage.home()
            self.homed.emit(True, f"Homed. Position: {self.stage.get_position_mm():.4f} mm")
        except Exception as e:
            self.homed.emit(False, f"Homing failed: {e}")
        finally:
            self._stage_busy = False

    @pyqtSlot(float)
    def configure_demod(self, rate_hz):
        if self.lockin is None:
            self.demod_configured.emit(False, "Not connected", 0.0)
            return
        try:
            self.lockin.configure_demod(demod_index=0, rate_hz=rate_hz)
            actual = self.lockin.get_actual_rate(demod_index=0)
            self.demod_configured.emit(True, "Demod configured", actual)
        except Exception as e:
            self.demod_configured.emit(False, f"Demod config failed: {e}", 0.0)

    # ------------------------------------------------------------------
    @pyqtSlot(object, int, int)
    def run_averaged_scan(self, delay_points_ps, n_readings_per_point, n_scans):
        
        if self.engine is None:
            self.averaging_finished.emit(False, "Scan ignored: not connected")
            return

        self._results = []
        self._stage_busy = True
        try:
            for event in self.engine.run_averaged_scan(
                delay_points_ps, n_readings_per_point, n_scans
            ):
                if event["type"] == "point":
                    self._results.append(event)
                    self.point_done.emit(event)
                elif event["type"] == "scan_complete":
                    self.scan_repeat_done.emit(event["scan_index"], event["scan_total"])

            self.averaging_finished.emit(
                True, f"Averaging complete: {n_scans} scan(s)."
            )

        except ScanAborted as e:
            self.averaging_finished.emit(False, f"Stopped: {e}")
        except StageNotHomedError as e:
            self.averaging_finished.emit(False, f"Scan failed: {e}")
        except Exception as e:
            self.averaging_finished.emit(False, f"Scan failed: {e}")
        finally:
            self._stage_busy = False

    def request_stop(self):
        """Thread-safe -- called directly from the GUI thread."""
        if self.engine is not None:
            self.engine.request_abort()

    def request_calib_stop(self):
        
        self._calib_abort_requested = True

    def get_results(self):
        """Flat list of all per-point 'point' events from the most recent
        Control-tab run_averaged_scan() call (full or partial-on-abort)."""
        return list(self._results)

    # ------------------------------------------------------------------
    @pyqtSlot(int, float)
    def run_calibration_scan(self, n_points, dwell_s):
       
        if self.stage is None:
            self.calib_finished.emit(False, "Calibration scan ignored: not connected.")
            return

        self._calib_abort_requested = False
        self._stage_busy = True
        try:
            try:
                t_min, t_max = self.stage.reachable_window_ps()
            except Exception as e:
                self.calib_finished.emit(False, f"Could not determine reachable range: {e}")
                return

            if n_points < 2:
                self.calib_finished.emit(False, "Calibration scan needs at least 2 points.")
                return

            commanded_points = np.linspace(t_min, t_max, n_points)
            for i, commanded_ps in enumerate(commanded_points):
                if self._calib_abort_requested:
                    self.calib_finished.emit(False, f"Stopped after {i}/{n_points} points.")
                    return
                try:
                    actual_mm = self.stage.move_to(float(commanded_ps))
                except Exception as e:
                    self.calib_finished.emit(False, f"Calibration move failed at point {i+1}: {e}")
                    return
                if dwell_s > 0:
                    time.sleep(dwell_s)
                self.calib_point_done.emit({
                    "commanded_ps": float(commanded_ps),
                    "actual_mm": actual_mm,
                    "index": i,
                    "total": n_points,
                })

            self.calib_finished.emit(True, f"Calibration scan complete: {n_points} points.")
        finally:
            self._stage_busy = False

    @pyqtSlot()
    def zero_lockin_phase(self):
        """Phase-zero calibration -- see MFLI.zero_phase()'s docstring
        for the underlying convention (additive phaseshift adjustment)."""
        if self.lockin is None or not self.lockin.is_connected:
            self.log.emit("Zero Phase ignored: lock-in not connected.")
            return
        try:
            result = self.lockin.zero_phase(demod_index=0)
            self.log.emit(
                f"Lock-in phase zeroed: theta was {result['theta_before_deg']:.3f}\u00b0, "
                f"phaseshift {result['phaseshift_before_deg']:.3f}\u00b0 -> "
                f"{result['phaseshift_after_deg']:.3f}\u00b0."
            )
            self.phase_zeroed.emit(result)
        except Exception as e:
            self.log.emit(f"Zero Phase failed: {e}")

    @pyqtSlot()
    def poll_status(self):
       
        if self._stage_busy:
            stage_status = {"connected": self.stage is not None, "homed": None,
                             "is_moving": True, "position_mm": None,
                             "raw_status": ["busy: calibration in progress"]}
        else:
            stage_status = self.stage.get_status() if self.stage is not None else \
                {"connected": False, "homed": False, "is_moving": None,
                 "position_mm": None, "raw_status": None}

        input_status = None
        reference_status = None
        laser_status = None
        sample = None
        phase_offset_deg = None
        if self.lockin is not None and self.lockin.is_connected:
            try:
                input_status = self.lockin.get_input_utilization(in_channel=0)
            except Exception:
                # Range-utilization nodes require LabOne >= 26.04 -- don't
                # let a missing optional health metric break the poll.
                input_status = None
            try:
                reference_status = self.lockin.get_reference_status(demod_index=0)
            except Exception:
                reference_status = None
            try:
                laser_status = self.lockin.get_laser_status()
            except Exception:
                laser_status = None
            try:
                
                sample = self.lockin.read_sample(demod_index=0)
                phase_offset_deg = self.lockin.get_phase_offset(demod_index=0)
            except Exception:
                sample = None
                phase_offset_deg = None

        self.status_update.emit({
            "stage": stage_status,
            "input": input_status,
            "reference": reference_status,
            "laser": laser_status,
            "sample": sample,
            "phase_offset_deg": phase_offset_deg,
        })

    # ------------------------------------------------------------------
    @pyqtSlot(str, dict, object, list)
    def save_results(self, dest_folder, metadata, average_by_point, point_order_ps):
        
        if not self._results:
            self.save_finished.emit(False, "Nothing to save -- no completed points.", "")
            return
        try:
            result = save_full_run(
                self._results, point_order_ps, average_by_point, dest_folder,
                metadata=metadata,
            )
            n_sweeps = len(result["sweep_files"])

            
            parameter_folders = result.get("parameter_folders", {})
            n_param_scan_files = sum(
                len(v["scan_files"]) for v in parameter_folders.values()
            )
            n_param_avg_files = sum(
                1 for v in parameter_folders.values() if v.get("average_file")
            )

            avg_phrase = "an average file" if result["average_file"] else \
                "no average file (no averaged data yet)"
            param_phrase = (
                f", plus X/Y/R folders ({n_param_scan_files} per-parameter scan "
                f"files + {n_param_avg_files} average files)" if parameter_folders
                else " (X/Y/R per-parameter folders were NOT created -- "
                     "scan_engine.py may be an older version; re-copy it from "
                     "oke_system/acquisition/scan_engine.py and restart the kernel)"
            )
            msg = f"Saved {n_sweeps} sweep file(s) and {avg_phrase}{param_phrase}, to: {dest_folder}"
            self.save_finished.emit(True, msg, dest_folder)
        except Exception as e:
            
            self.save_finished.emit(
                False, f"Save failed: {type(e).__name__}: {e}", dest_folder
            )

    # ------------------------------------------------------------------
    @pyqtSlot(float)
    def manual_move_absolute(self, target_ps):
        """
        Single-point move outside of a scan (the sketch's "Move
        absolute" control). Blocks the worker thread until the move
        completes -- this is fine since it's on the background thread,
        not the GUI thread -- and reports the same StageNotHomedError
        path a scan would hit if Home was skipped.
        """
        if self.stage is None:
            self.log.emit("Move ignored: not connected.")
            return
        self._stage_busy = True
        try:
            actual_mm = self.stage.move_to(target_ps)
            self.log.emit(f"Moved to {target_ps:.4f} ps (stage at {actual_mm:.4f} mm).")
        except StageNotHomedError as e:
            self.log.emit(f"Move failed: {e}")
        except Exception as e:
            self.log.emit(f"Move failed: {e}")
        finally:
            self._stage_busy = False

    @pyqtSlot(float)
    def manual_move_relative(self, delta_ps):
        
        if self.stage is None:
            self.log.emit("Jog ignored: not connected.")
            return
        self._stage_busy = True
        try:
            current_mm = self.stage.get_position_mm()
            current_ps = self.stage.mm_to_ps(current_mm)
            actual_mm = self.stage.jog(delta_ps, current_time_ps=current_ps)
            self.log.emit(
                f"Jogged {delta_ps:+.4f} ps from {current_ps:.4f} ps "
                f"(stage now at {actual_mm:.4f} mm)."
            )
        except StageNotHomedError as e:
            self.log.emit(f"Jog failed: {e}")
        except Exception as e:
            self.log.emit(f"Jog failed: {e}")
        finally:
            self._stage_busy = False

    @pyqtSlot()
    def emergency_stop_stage(self):
        
        if self.stage is None:
            self.log.emit("Emergency stop ignored: not connected.")
            return
        try:
            self.stage.stop()
            self.log.emit("Stage motion halted (emergency stop).")
        except Exception as e:
            self.log.emit(f"Emergency stop failed: {e}")

    @pyqtSlot(float)
    def jog_time_zero(self, delta_ps):
        
        if self.stage is None:
            self.log.emit("Time-zero jog ignored: not connected.")
            return
        self._stage_busy = True
        try:
            current_mm = self.stage.get_position_mm()
            current_ps = self.stage.mm_to_ps(current_mm)
            actual_mm = self.stage.jog(delta_ps, current_time_ps=current_ps)
            self.log.emit(
                f"Time-zero jog {delta_ps:+.4f} ps from {current_ps:.4f} ps "
                f"(stage now at {actual_mm:.4f} mm). t0 unchanged at "
                f"{self.stage.t0_ps:.4f} ps -- use Define HOME here to lock "
                f"in the current position as the new zero delay."
            )
        except StageNotHomedError as e:
            self.log.emit(f"Time-zero jog failed: {e}")
        except Exception as e:
            self.log.emit(f"Time-zero jog failed: {e}")
        finally:
            self._stage_busy = False

    @pyqtSlot()
    def goto_t0(self):
        """Moves the stage to time_ps=0 under the CURRENT t0_ps convention."""
        if self.stage is None:
            self.log.emit("Go-to-t0 ignored: not connected.")
            return
        self._stage_busy = True
        try:
            actual_mm = self.stage.move_to(0.0)
            self.log.emit(f"Moved to t0 (0 ps). Stage at {actual_mm:.4f} mm.")
        except StageNotHomedError as e:
            self.log.emit(f"Go-to-t0 failed: {e}")
        except Exception as e:
            self.log.emit(f"Go-to-t0 failed: {e}")
        finally:
            self._stage_busy = False

    @pyqtSlot()
    def define_home_here(self):
        
        if self.stage is None:
            self.log.emit("Define HOME ignored: not connected.")
            return
        try:
            new_t0 = self.stage.define_home_here()
            self.log.emit(f"HOME defined here -> t0 = {new_t0:.4f} ps.")
            self.t0_changed.emit(new_t0)
        except Exception as e:
            self.log.emit(f"Define HOME failed: {e}")

    @pyqtSlot()
    def disconnect_hardware(self):
        if self.stage is not None:
            try:
                self.stage.close()
            except Exception as e:
                self.log.emit(f"Error closing stage: {e}")
            self.stage = None
        if self.lockin is not None:
            try:
                self.lockin.close()
            except Exception as e:
                self.log.emit(f"Error closing lock-in: {e}")
            self.lockin = None
        self.engine = None
        self.log.emit("Disconnected")


# ==========================================================================
def _status_chip(label_widget, text, color):
    """Small helper: set a QLabel to look like a colored status chip."""
    label_widget.setText(text)
    label_widget.setStyleSheet(
        f"background-color: {color}; color: white; border-radius: 4px; "
        f"padding: 2px 6px; font-weight: bold;"
    )


# ==========================================================================
class ScanEngineGUI(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OKE Scan Engine -- Experiment Control")
        self.results = []  # last completed/partial scan's flat point-event list
        # True while the Control tab's scan is running -- gates the
        # Start button so a second scan can't be launched mid-run.
        self._any_scan_running = False

        # Per-scan-index storage for overlay/heatmap, keyed by scan_index:
        # {scan_index: {"times_ps": [...], "x": [...], "y": [...], "r": [...]}}
        self._scans = {}
        self._running_mean_by_point = {}  # point_index -> latest running_mean dict
        self._point_order_ps = []         # time_ps in visiting order of THIS run
        self._drift_reference_r = None    # R at the first delay point, first scan
        self._average_history = []        # one running-average row per completed scan (heatmap Mode 2)
        self._final_average = None        # frozen {"times_ps","x","y"} once a run completes
        self._calib_commanded_ps = []     # accumulated live calibration scan points
        self._calib_actual_mm = []

        # Worker + thread setup.
        self.worker = ScanWorker()
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

        self._build_ui()
        self._connect_signals()
        self._redraw_calibration_plot()

        # Status poll timer -- runs on the GUI thread; each tick invokes
        # the worker's poll_status() slot via a queued connection, so the
        # actual hardware queries happen on the worker thread.
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._request_status_poll)
        self._status_timer.start()

        self._scan_start_time = None
        self._scan_n_points_total = 0
        self._scan_n_scans_total = 1

    # ------------------------------------------------------------------
    def _build_ui(self):
        
        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(4)
        self.setCentralWidget(root)

        tabs = QtWidgets.QTabWidget()
        root_layout.addWidget(tabs, stretch=1)

        control_page = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(control_page)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        
        outer.addWidget(self._build_top_bar())

        
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(self._build_sidebar())
        main_splitter.addWidget(self._build_plots_area())
        main_splitter.setStretchFactor(0, 0)  # sidebar: fixed-ish, don't grab extra space
        main_splitter.setStretchFactor(1, 1)  # plots: claim all extra space
        main_splitter.setSizes([500, 1000])
        outer.addWidget(main_splitter, stretch=1)

        # ---- Bottom: metrics strip (Control-tab-specific -- SNR/FWHM/
        # etc. only ever get computed from the Control tab's own scan,
        # so unlike the Log this one legitimately stays tab-local) ----
        outer.addWidget(self._build_metrics_strip())

        tabs.addTab(control_page, "Control")
        tabs.addTab(self._build_calibration_tab(), "Calibration")

        # ---- Log -- shared across all tabs, see the comment above. ----
        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(90)
        log_group = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        log_layout.setContentsMargins(4, 4, 4, 4)
        log_layout.addWidget(self.log_box)
        root_layout.addWidget(log_group)

        self._build_settings_dialog()

        self.resize(1500, 950)

    # ------------------------------------------------------------------
    def _build_top_bar(self):
        
        bar = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(4, 2, 4, 2)

        
        self.start_scan_btn = QtWidgets.QPushButton("\u25b6 Start")
        self.start_scan_btn.setEnabled(False)
        self.start_scan_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.stop_scan_btn = QtWidgets.QPushButton("\u25a0 Stop")
        self.stop_scan_btn.setEnabled(False)
        self.stop_scan_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")
        h.addWidget(self.start_scan_btn)
        h.addWidget(self.stop_scan_btn)

        h.addSpacing(16)

        # Scan progress -- consolidates what used to be the Averaging
        # group's progress bar/label into the top bar, per the mockup.
        progress_box = QtWidgets.QVBoxLayout()
        self.avg_progress_label = QtWidgets.QLabel("Completed: 0/0")
        self.avg_progress_bar = QtWidgets.QProgressBar()
        self.avg_progress_bar.setMaximumHeight(14)
        self.avg_progress_bar.setMaximumWidth(360)
        progress_box.addWidget(self.avg_progress_label)
        progress_box.addWidget(self.avg_progress_bar)
        h.addLayout(progress_box)

        self.eta_label = QtWidgets.QLabel("Remaining: --")
        h.addWidget(self.eta_label)

        h.addStretch(1)

        # Status lights -- compact single-line chips for the three most
        # at-a-glance-important states. The DETAILED versions (with
        # frequency/power/notes) live in the sidebar's Signal Monitor
        # section and the Settings dialog, unchanged -- these top-bar
        # chips are deliberately terse, matching the mockup's small
        # colored-dot status indicators.
        self.lockin_conn_chip = QtWidgets.QLabel("LOCK-IN: --")
        _status_chip(self.lockin_conn_chip, "LOCK-IN: --", "#9e9e9e")
        self.chopper_chip = QtWidgets.QLabel("REF: not yet read")
        _status_chip(self.chopper_chip, "REF: not yet read", "#9e9e9e")
        self.stage_status_chip_top = QtWidgets.QLabel("STAGE: --")
        _status_chip(self.stage_status_chip_top, "STAGE: --", "#9e9e9e")
        for chip in (self.lockin_conn_chip, self.chopper_chip, self.stage_status_chip_top):
            chip.setMaximumWidth(150)
            h.addWidget(chip)

        h.addSpacing(12)
        self.settings_btn = QtWidgets.QPushButton("\u2699 Settings")
        self.settings_btn.clicked.connect(self._open_settings_dialog)
        h.addWidget(self.settings_btn)

        return bar

    # ------------------------------------------------------------------
    def _build_plots_area(self):
        """The three-plot main area, unchanged in content from before --
        just no longer competing with a tall sidebar for vertical space."""
        container = QtWidgets.QWidget()
        plots_row = QtWidgets.QHBoxLayout(container)
        plots_row.setContentsMargins(0, 0, 0, 0)
        plots_row.addWidget(self._build_heatmap_panel(), stretch=4)

        right_col = QtWidgets.QVBoxLayout()
        right_col.addWidget(self._build_decay_plot_panel(), stretch=3)
        right_col.addWidget(self._build_residual_panel(), stretch=2)
        plots_row.addLayout(right_col, stretch=5)
        return container

    # ------------------------------------------------------------------
    def _build_sidebar(self):
        
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(480)
        scroll.setMaximumWidth(540)

        content = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(content)
        v.setSpacing(6)

        v.addWidget(self._build_delay_stage_group())
        v.addWidget(self._build_scan_parameters_group())
        v.addWidget(self._build_averaging_group())
        v.addWidget(self._build_signal_monitor_group())
        v.addStretch(1)

        scroll.setWidget(content)
        return scroll

    # ------------------------------------------------------------------
    def _build_delay_stage_group(self):
        
        stage_group = QtWidgets.QGroupBox("Delay Stage")
        outer_v = QtWidgets.QVBoxLayout(stage_group)

        sg_widget = QtWidgets.QWidget()
        sg = QtWidgets.QGridLayout(sg_widget)
        sg.setVerticalSpacing(4)
        sg.setContentsMargins(0, 0, 0, 0)

        sg.addWidget(QtWidgets.QLabel("Position:"), 0, 0)
        self.current_pos_label = QtWidgets.QLabel("-- mm")
        self.current_pos_label.setStyleSheet("font-weight: bold;")
        sg.addWidget(self.current_pos_label, 0, 1)

        sg.addWidget(QtWidgets.QLabel("Delay:"), 1, 0)
        self.current_delay_label = QtWidgets.QLabel("-- ps")
        self.current_delay_label.setStyleSheet("font-weight: bold;")
        sg.addWidget(self.current_delay_label, 1, 1)

        self.home_btn = QtWidgets.QPushButton("Home")
        self.home_btn.setEnabled(False)
        sg.addWidget(self.home_btn, 2, 0)

        self.estop_btn = QtWidgets.QPushButton("STOP")
        self.estop_btn.setEnabled(False)
        self.estop_btn.setStyleSheet(
            "background-color: #b71c1c; color: white; font-weight: bold;"
        )
        sg.addWidget(self.estop_btn, 2, 1)

        sg.addWidget(QtWidgets.QLabel("Move to (ps):"), 3, 0)
        self.target_delay_input = QtWidgets.QDoubleSpinBox()
        self.target_delay_input.setRange(-1e6, 1e6)
        self.target_delay_input.setDecimals(3)
        self.target_delay_input.setMaximumWidth(110)
        sg.addWidget(self.target_delay_input, 3, 1)
        self.move_abs_btn = QtWidgets.QPushButton("Move")
        self.move_abs_btn.setMinimumWidth(60)
        self.move_abs_btn.setEnabled(False)
        sg.addWidget(self.move_abs_btn, 3, 2)

        sg.addWidget(QtWidgets.QLabel("Move by (ps):"), 4, 0)
        self.jog_step_input = QtWidgets.QDoubleSpinBox()
        self.jog_step_input.setRange(-1e6, 1e6)
        self.jog_step_input.setDecimals(3)
        self.jog_step_input.setValue(1.0)
        self.jog_step_input.setMaximumWidth(110)
        sg.addWidget(self.jog_step_input, 4, 1)
        self.move_rel_btn = QtWidgets.QPushButton("Move")
        self.move_rel_btn.setMinimumWidth(60)
        self.move_rel_btn.setEnabled(False)
        sg.addWidget(self.move_rel_btn, 4, 2)

        self.goto_t0_btn = QtWidgets.QPushButton("Go to t0 (zero delay)")
        self.goto_t0_btn.setEnabled(False)
        sg.addWidget(self.goto_t0_btn, 5, 0, 1, 3)

        self.stage_status_chip = QtWidgets.QLabel("STAGE: --")
        _status_chip(self.stage_status_chip, "STAGE: --", "#9e9e9e")
        sg.addWidget(self.stage_status_chip, 6, 0, 1, 3)

        outer_v.addWidget(sg_widget)

        
        jog_group = QtWidgets.QGroupBox("Time Zero Jog (cumulative)")
        jg = QtWidgets.QGridLayout(jog_group)
        jg.setSpacing(3)

        # (label, delta_ps) -- the physical delay-jog amount per press,
        # in ps (fs values expressed directly in ps, e.g. 5fs = 0.005).
        jog_specs = [
            ("-5fs", -0.005), ("-10fs", -0.010), ("-100fs", -0.100),
            ("-1ps", -1.0), ("-10ps", -10.0),
        ]
        jog_specs_pos = [
            ("+5fs", 0.005), ("+10fs", 0.010), ("+100fs", 0.100),
            ("+1ps", 1.0), ("+10ps", 10.0),
        ]
        self.tzero_jog_buttons = []
        for col, (label, delta) in enumerate(jog_specs):
            btn = QtWidgets.QPushButton(label)
            btn.setMinimumWidth(70)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, d=delta: self.on_tzero_jog_clicked(d))
            jg.addWidget(btn, 0, col)
            self.tzero_jog_buttons.append(btn)
        for col, (label, delta) in enumerate(jog_specs_pos):
            btn = QtWidgets.QPushButton(label)
            btn.setMinimumWidth(70)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _, d=delta: self.on_tzero_jog_clicked(d))
            jg.addWidget(btn, 1, col)
            self.tzero_jog_buttons.append(btn)

        self.define_home_btn = QtWidgets.QPushButton("Define HOME here")
        self.define_home_btn.setEnabled(False)
        self.define_home_btn.setStyleSheet(
            "background-color: #1565c0; color: white; font-weight: bold;"
        )
        jg.addWidget(self.define_home_btn, 2, 0, 1, 5)

        outer_v.addWidget(jog_group)

        return stage_group

    # ------------------------------------------------------------------
    def _build_scan_parameters_group(self):
        
        scan_group = QtWidgets.QGroupBox("Scan Parameters")
        outer = QtWidgets.QVBoxLayout(scan_group)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Mode:"))
        self.scan_mode_combo = QtWidgets.QComboBox()
        self.scan_mode_combo.addItems(["Simple", "3-Section"])
        mode_row.addWidget(self.scan_mode_combo)
        mode_row.addStretch(1)
        outer.addLayout(mode_row)

        self.scan_mode_stack = QtWidgets.QStackedWidget()
        self.scan_mode_stack.addWidget(self._build_simple_scan_page())
        self.scan_mode_stack.addWidget(self._build_multisection_scan_page())
        outer.addWidget(self.scan_mode_stack)
        self.scan_mode_combo.currentIndexChanged.connect(self.scan_mode_stack.setCurrentIndex)
        self.scan_mode_combo.currentIndexChanged.connect(self._update_scan_summary)

        self.n_points_label = QtWidgets.QLabel("Points: --")
        self.n_points_label.setWordWrap(True)
        outer.addWidget(self.n_points_label)

        self.scan_duration_label = QtWidgets.QLabel("Est. duration: --")
        self.scan_duration_label.setWordWrap(True)
        outer.addWidget(self.scan_duration_label)

        return scan_group

    # ------------------------------------------------------------------
    def _build_simple_scan_page(self):
        """The original single Start/Stop/Step/Readings/Direction controls,
        unchanged -- now living as page 0 of scan_mode_stack."""
        page = QtWidgets.QWidget()
        scg = QtWidgets.QGridLayout(page)
        scg.setContentsMargins(0, 4, 0, 0)
        scg.setVerticalSpacing(4)

        scg.addWidget(QtWidgets.QLabel("Start (ps):"), 0, 0)
        self.start_input = QtWidgets.QDoubleSpinBox()
        self.start_input.setRange(-1e6, 1e6)
        self.start_input.setDecimals(3)
        self.start_input.setValue(-5.0)
        scg.addWidget(self.start_input, 0, 1)

        scg.addWidget(QtWidgets.QLabel("Stop (ps):"), 1, 0)
        self.stop_input = QtWidgets.QDoubleSpinBox()
        self.stop_input.setRange(-1e6, 1e6)
        self.stop_input.setDecimals(3)
        self.stop_input.setValue(100.0)
        scg.addWidget(self.stop_input, 1, 1)

        scg.addWidget(QtWidgets.QLabel("Step (ps):"), 2, 0)
        self.step_input = QtWidgets.QDoubleSpinBox()
        self.step_input.setRange(0.0001, 1e6)
        self.step_input.setDecimals(4)
        self.step_input.setValue(0.05)
        scg.addWidget(self.step_input, 2, 1)

        scg.addWidget(QtWidgets.QLabel("Readings/pt:"), 3, 0)
        self.n_readings_input = QtWidgets.QSpinBox()
        self.n_readings_input.setRange(1, 1000)
        self.n_readings_input.setValue(3)
        scg.addWidget(self.n_readings_input, 3, 1)

        scg.addWidget(QtWidgets.QLabel("Direction:"), 4, 0)
        self.direction_combo = QtWidgets.QComboBox()
        self.direction_combo.addItems(["Forward", "Reverse", "Bidirectional"])
        scg.addWidget(self.direction_combo, 4, 1)

        return page

    # ------------------------------------------------------------------
    def _build_multisection_scan_page(self):
        
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(0, 4, 0, 0)

        table = QtWidgets.QGridLayout()
        table.setVerticalSpacing(3)
        table.setHorizontalSpacing(3)
        headers = ["", "Start", "Stop", "Step", "Points", "Scale"]
        for c, text in enumerate(headers):
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet("font-size: 9px; color: #616161;")
            table.addWidget(lbl, 0, c)

        
        default_sections = [
            (-0.500, 0.020, 0.005, 50, "Linear"),
            (0.020, 500.0, 8.0, 50, "Linear"),
            (500.0, 3000.0, 8.0, 50, "Linear"),
        ]
        section_colors = ["#2a7ae2", "#e2702a", "#2ab37a"]

        self.section_start_inputs = []
        self.section_stop_inputs = []
        self.section_step_inputs = []
        self.section_points_inputs = []
        self.section_scale_combos = []

        for i, (start, stop, step, points, scale) in enumerate(default_sections):
            row = i + 1
            label = QtWidgets.QLabel(f"Sec {i+1}:")
            label.setStyleSheet(f"color: {section_colors[i]}; font-weight: bold; font-size: 10px;")
            table.addWidget(label, row, 0)

            start_in = QtWidgets.QDoubleSpinBox()
            start_in.setRange(-1e6, 1e6)
            start_in.setDecimals(4)
            start_in.setValue(start)
            start_in.setMaximumWidth(85)
            table.addWidget(start_in, row, 1)
            self.section_start_inputs.append(start_in)

            stop_in = QtWidgets.QDoubleSpinBox()
            stop_in.setRange(-1e6, 1e6)
            stop_in.setDecimals(4)
            stop_in.setValue(stop)
            stop_in.setMaximumWidth(85)
            table.addWidget(stop_in, row, 2)
            self.section_stop_inputs.append(stop_in)

            step_in = QtWidgets.QDoubleSpinBox()
            step_in.setRange(0.0001, 1e6)
            step_in.setDecimals(4)
            step_in.setValue(step)
            step_in.setMaximumWidth(75)
            table.addWidget(step_in, row, 3)
            self.section_step_inputs.append(step_in)

            points_in = QtWidgets.QSpinBox()
            points_in.setRange(2, 100000)
            points_in.setValue(int(points))
            points_in.setMaximumWidth(65)
            points_in.setEnabled(False)  # enabled only when this row's scale == Log
            table.addWidget(points_in, row, 4)
            self.section_points_inputs.append(points_in)

            scale_combo = QtWidgets.QComboBox()
            scale_combo.addItems(["Linear", "Log"])
            scale_combo.setCurrentText(scale)
            scale_combo.setMaximumWidth(75)
            table.addWidget(scale_combo, row, 5)
            self.section_scale_combos.append(scale_combo)

            # Toggle step vs points enabled-state exactly like the
            # Tkinter tool's _make_toggle()/_toggle() pair.
            def _make_toggle(step_widget, points_widget, combo=scale_combo):
                def _toggle():
                    is_log = combo.currentText() == "Log"
                    step_widget.setEnabled(not is_log)
                    points_widget.setEnabled(is_log)
                return _toggle
            toggle = _make_toggle(step_in, points_in)
            scale_combo.currentIndexChanged.connect(lambda _, t=toggle: t())
            toggle()

        v.addLayout(table)

        help_label = QtWidgets.QLabel(
            "Step is used for Linear sections; Points for Log sections.\n"
            "(Log needs Start and Stop both > 0.)"
        )
        help_label.setStyleSheet("color: #757575; font-size: 9px;")
        help_label.setWordWrap(True)
        v.addWidget(help_label)

        self.load_timefile_btn = QtWidgets.QPushButton("Load Timefile...")
        v.addWidget(self.load_timefile_btn)

        # Recompute the live summary whenever any section field changes
        # -- mirrors the Tkinter tool's debounced live-update behaviour,
        # though here it's driven through the same _update_scan_summary
        # already used by the Simple page rather than a separate timer.
        for w in (self.section_start_inputs + self.section_stop_inputs +
                  self.section_step_inputs + self.section_points_inputs):
            w.valueChanged.connect(self._update_scan_summary)
        for c in self.section_scale_combos:
            c.currentIndexChanged.connect(self._update_scan_summary)

        return page

    # ------------------------------------------------------------------
    def _build_averaging_group(self):
        
        avg_group = QtWidgets.QGroupBox("Averaging")
        ag = QtWidgets.QGridLayout(avg_group)
        ag.setVerticalSpacing(4)

        ag.addWidget(QtWidgets.QLabel("Total scans:"), 0, 0)
        self.n_scans_input = QtWidgets.QSpinBox()
        self.n_scans_input.setRange(1, 100000)
        self.n_scans_input.setValue(10)
        ag.addWidget(self.n_scans_input, 0, 1)

        self.save_btn = QtWidgets.QPushButton("Save Results...")
        self.save_btn.setEnabled(False)
        ag.addWidget(self.save_btn, 1, 0, 1, 2)

        # Recompute scan-summary labels whenever a relevant field changes.
        for w in (self.start_input, self.stop_input, self.step_input,
                  self.direction_combo, self.n_readings_input, self.n_scans_input):
            if isinstance(w, QtWidgets.QComboBox):
                w.currentIndexChanged.connect(self._update_scan_summary)
            else:
                w.valueChanged.connect(self._update_scan_summary)
        self._update_scan_summary()

        return avg_group

    # ------------------------------------------------------------------
    def _build_signal_monitor_group(self):
        
        group = QtWidgets.QGroupBox("Signal Monitor")
        g = QtWidgets.QGridLayout(group)
        g.setVerticalSpacing(3)

        g.addWidget(QtWidgets.QLabel("X:"), 0, 0)
        self.x_label = QtWidgets.QLabel("-- V")
        g.addWidget(self.x_label, 0, 1)
        g.addWidget(QtWidgets.QLabel("Y:"), 1, 0)
        self.y_label = QtWidgets.QLabel("-- V")
        g.addWidget(self.y_label, 1, 1)
        g.addWidget(QtWidgets.QLabel("R:"), 2, 0)
        self.r_label = QtWidgets.QLabel("-- V")
        g.addWidget(self.r_label, 2, 1)
        g.addWidget(QtWidgets.QLabel("\u03b8:"), 3, 0)
        self.theta_label = QtWidgets.QLabel("-- \u00b0")
        g.addWidget(self.theta_label, 3, 1)

        g.addWidget(QtWidgets.QLabel("Input:"), 4, 0)
        self.input_level_label = QtWidgets.QLabel("--")
        self.input_level_label.setWordWrap(True)
        g.addWidget(self.input_level_label, 4, 1)
        self.overload_chip = QtWidgets.QLabel("INPUT: --")
        _status_chip(self.overload_chip, "INPUT: --", "#9e9e9e")
        g.addWidget(self.overload_chip, 5, 0, 1, 2)

        g.addWidget(QtWidgets.QLabel("Ref. freq:"), 6, 0)
        self.chopper_freq_label = QtWidgets.QLabel("-- Hz")
        g.addWidget(self.chopper_freq_label, 6, 1)

        return group

    # ------------------------------------------------------------------
    def _build_settings_dialog(self):
        
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Settings")
        dlg.setMinimumWidth(480)
        v = QtWidgets.QVBoxLayout(dlg)

        # --- Connection ---
        conn_group = QtWidgets.QGroupBox("Connection")
        conn_layout = QtWidgets.QGridLayout(conn_group)
        conn_layout.addWidget(QtWidgets.QLabel("Stage serial no.:"), 0, 0)
        self.serial_input = QtWidgets.QLineEdit("103537224")
        conn_layout.addWidget(self.serial_input, 0, 1)
        conn_layout.addWidget(QtWidgets.QLabel("t0 (ps):"), 0, 2)
        self.t0_input = QtWidgets.QDoubleSpinBox()
        self.t0_input.setRange(-1e6, 1e6)
        self.t0_input.setDecimals(4)
        self.t0_input.setValue(DelayStage.t0_ps_for_position_mm(220.0))
        
        conn_layout.addWidget(self.t0_input, 0, 3)
        conn_layout.addWidget(QtWidgets.QLabel("MFLI device ID:"), 1, 0)
        self.device_id_input = QtWidgets.QLineEdit("dev33006")
        conn_layout.addWidget(self.device_id_input, 1, 1)
        self.connect_btn = QtWidgets.QPushButton("Connect")
        conn_layout.addWidget(self.connect_btn, 1, 2)
        self.disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        conn_layout.addWidget(self.disconnect_btn, 1, 3)
        conn_layout.addWidget(QtWidgets.QLabel("Demod rate (Sa/s):"), 2, 0)
        self.rate_input = QtWidgets.QDoubleSpinBox()
        self.rate_input.setRange(1, 500000)
        self.rate_input.setValue(1000)
        conn_layout.addWidget(self.rate_input, 2, 1)
        self.configure_demod_btn = QtWidgets.QPushButton("Configure Demod")
        self.configure_demod_btn.setEnabled(False)
        conn_layout.addWidget(self.configure_demod_btn, 2, 2, 1, 2)
        self.status_label = QtWidgets.QLabel("Not connected")
        self.status_label.setStyleSheet("color: gray;")
        conn_layout.addWidget(self.status_label, 3, 0, 1, 4)
        v.addWidget(conn_group)

        
        chopper_group = QtWidgets.QGroupBox("Chopper Reference -- Detail")
        cg = QtWidgets.QVBoxLayout(chopper_group)
        chopper_note = QtWidgets.QLabel(
            "Reference-lock read is unconfirmed against this rig's "
            "chopper setup (not exercised during commissioning)."
        )
        chopper_note.setWordWrap(True)
        chopper_note.setStyleSheet("color: #757575; font-size: 10px;")
        cg.addWidget(chopper_note)
        v.addWidget(chopper_group)

        # --- Laser Status ---
        laser_group = QtWidgets.QGroupBox("Laser Status")
        lag = QtWidgets.QGridLayout(laser_group)
        self.laser_chip = QtWidgets.QLabel("LASER MONITOR: not available")
        _status_chip(self.laser_chip, "LASER MONITOR: not available", "#9e9e9e")
        lag.addWidget(self.laser_chip, 0, 0, 1, 2)
        lag.addWidget(QtWidgets.QLabel("Power:"), 1, 0)
        self.laser_power_label = QtWidgets.QLabel("-- mW")
        lag.addWidget(self.laser_power_label, 1, 1)
        lag.addWidget(QtWidgets.QLabel("Stability:"), 2, 0)
        self.laser_stability_label = QtWidgets.QLabel("-- %")
        lag.addWidget(self.laser_stability_label, 2, 1)
        laser_note = QtWidgets.QLabel(
            "No laser power/stability sensor is wired up yet -- add a "
            "real read in MFLI.get_laser_status() to populate this."
        )
        laser_note.setWordWrap(True)
        laser_note.setStyleSheet("color: #757575; font-size: 10px;")
        lag.addWidget(laser_note, 3, 0, 1, 2)
        v.addWidget(laser_group)

        # --- Acquisition Engine status (detail copy; top bar has none,
        # so this is the only place engine_chip lives) ---
        engine_group = QtWidgets.QGroupBox("Acquisition Engine")
        eg = QtWidgets.QGridLayout(engine_group)
        self.engine_chip = QtWidgets.QLabel("ENGINE: idle")
        _status_chip(self.engine_chip, "ENGINE: idle", "#9e9e9e")
        eg.addWidget(self.engine_chip, 0, 0, 1, 2)
        v.addWidget(engine_group)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        v.addWidget(close_btn)

        self._settings_dialog = dlg

    def _open_settings_dialog(self):
        self._settings_dialog.show()
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()

    def _build_calibration_tab(self):
        
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)

        # --- Header info line (t0, slope, reachable range) ---
        self.calib_header_label = QtWidgets.QLabel(
            "t0 = -- ps  |  slope = -- mm/ps  |  reachable delay range: [--, --] ps"
        )
        self.calib_header_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        v.addWidget(self.calib_header_label)

        # --- Delay stage calibration scan controls ---
        scan_box = QtWidgets.QGroupBox(
            "Live calibration scan (sanity check only - re-plots commanded ps vs computed mm)"
        )
        scan_layout = QtWidgets.QGridLayout(scan_box)
        scan_layout.addWidget(QtWidgets.QLabel("Points:"), 0, 0)
        self.calib_points_input = QtWidgets.QSpinBox()
        self.calib_points_input.setRange(2, 10000)
        self.calib_points_input.setValue(41)
        scan_layout.addWidget(self.calib_points_input, 0, 1)
        scan_layout.addWidget(QtWidgets.QLabel("Dwell (s):"), 0, 2)
        self.calib_dwell_input = QtWidgets.QDoubleSpinBox()
        self.calib_dwell_input.setRange(0.0, 60.0)
        self.calib_dwell_input.setDecimals(2)
        self.calib_dwell_input.setValue(0.10)
        scan_layout.addWidget(self.calib_dwell_input, 0, 3)

        self.calib_run_btn = QtWidgets.QPushButton("Run Live Calibration Scan")
        self.calib_run_btn.setEnabled(False)
        self.calib_run_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        scan_layout.addWidget(self.calib_run_btn, 1, 0, 1, 2)
        self.calib_stop_btn = QtWidgets.QPushButton("Stop")
        self.calib_stop_btn.setEnabled(False)
        self.calib_stop_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")
        scan_layout.addWidget(self.calib_stop_btn, 1, 2, 1, 2)

        self.calib_progress_bar = QtWidgets.QProgressBar()
        scan_layout.addWidget(self.calib_progress_bar, 2, 0, 1, 4)

        note = QtWidgets.QLabel(
            "Steps across the full reachable range for the current t0, driving the "
            "real stage at each point and re-reading back the mm value via the same "
            "conversion move_to() uses, then plots it (red circles) against the "
            "theoretical line so a mismatch is visible immediately."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #757575; font-size: 9px;")
        scan_layout.addWidget(note, 3, 0, 1, 4)

        v.addWidget(scan_box)

        # --- Calibration plot ---
        self.calib_figure = Figure(figsize=(7, 5))
        self.calib_figure.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12)
        self.calib_canvas = FigureCanvasQTAgg(self.calib_figure)
        self.calib_ax = self.calib_figure.add_subplot(111)
        self.calib_ax.set_xlabel("Delay (ps)")
        self.calib_ax.set_ylabel("Stage position (mm)")
        self.calib_ax.grid(True, alpha=0.3)

        self.calib_line_theory, = self.calib_ax.plot(
            [], [], "-", color="#1565c0", linewidth=2, label="Calibration (mm vs ps)"
        )
        self.calib_band = None  # axvspan patch for the reachable range, built on first draw
        self.calib_point_current, = self.calib_ax.plot(
            [], [], "o", color="#2e7d32", markersize=12, label="Current position"
        )
        self.calib_scatter_live, = self.calib_ax.plot(
            [], [], "o", color="#c62828", markersize=5, markerfacecolor="none",
            markeredgewidth=1.5, label="Live calibration scan"
        )
        self.calib_ax.legend(loc="upper right", fontsize=8)

        v.addWidget(self.calib_canvas)

        legend_note = QtWidgets.QLabel(
            "Blue line: ps-to-mm calibration for current t0  |  Shaded band: reachable "
            "0-150 mm travel  |  Green dot: current position  |  Orange x: built scan "
            "points  |  Red circles: live calibration scan (commanded ps -> actual mm)"
        )
        legend_note.setWordWrap(True)
        legend_note.setStyleSheet("color: #757575; font-size: 9px;")
        v.addWidget(legend_note)

        # --- Lock-in phase calibration ---
        phase_box = QtWidgets.QGroupBox("Lock-in Phase Calibration")
        pg = QtWidgets.QGridLayout(phase_box)
        pg.addWidget(QtWidgets.QLabel("Current X/Y/R/\u03b8:"), 0, 0)
        self.calib_lockin_readout_label = QtWidgets.QLabel("-- / -- / -- / --")
        pg.addWidget(self.calib_lockin_readout_label, 0, 1, 1, 2)

        pg.addWidget(QtWidgets.QLabel("Phaseshift (deg):"), 1, 0)
        self.calib_phaseshift_label = QtWidgets.QLabel("--")
        pg.addWidget(self.calib_phaseshift_label, 1, 1)

        self.calib_zero_phase_btn = QtWidgets.QPushButton("Zero Phase")
        self.calib_zero_phase_btn.setEnabled(False)
        self.calib_zero_phase_btn.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold;")
        pg.addWidget(self.calib_zero_phase_btn, 1, 2)

        phase_note = QtWidgets.QLabel(
            "Reads the current signal phase \u03b8 and adds it to the demodulator's "
            "phaseshift setting, so a fresh reading on this same signal reports "
            "\u03b8\u2248 0 (all signal on X, Y\u22480) -- the standard lock-in phase-zeroing "
            "procedure. Only meaningful with a STABLE reference signal present "
            "(e.g. during dedicated calibration), not concurrently with a live OKE "
            "scan."
        )
        phase_note.setWordWrap(True)
        phase_note.setStyleSheet("color: #757575; font-size: 9px;")
        pg.addWidget(phase_note, 2, 0, 1, 3)

        v.addWidget(phase_box)
        v.addStretch(1)

        return page

    # ------------------------------------------------------------------
    def _build_metrics_strip(self):
        """SNR | Noise | Drift | Peak | FWHM | ETA, per the sketch's bottom row."""
        group = QtWidgets.QGroupBox("Real-Time Metrics")
        h = QtWidgets.QHBoxLayout(group)

        def _metric(name):
            box = QtWidgets.QVBoxLayout()
            title = QtWidgets.QLabel(name)
            title.setStyleSheet("color: #616161; font-size: 10px;")
            value = QtWidgets.QLabel("--")
            value.setStyleSheet("font-size: 16px; font-weight: bold;")
            box.addWidget(title)
            box.addWidget(value)
            container = QtWidgets.QWidget()
            container.setLayout(box)
            h.addWidget(container)
            return value

        self.snr_value_label = _metric("SNR")
        self.noise_value_label = _metric("Noise (std)")
        self.drift_value_label = _metric("Drift")
        self.peak_value_label = _metric("Peak delay (t0 est.)")
        self.fwhm_value_label = _metric("FWHM")
        self.metrics_eta_label = _metric("ETA")

        return group

    # ------------------------------------------------------------------
    def _connect_signals(self):
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        self.disconnect_btn.clicked.connect(self.on_disconnect_clicked)
        self.home_btn.clicked.connect(self.on_home_clicked)
        self.configure_demod_btn.clicked.connect(self.on_configure_demod_clicked)
        self.move_abs_btn.clicked.connect(self.on_move_abs_clicked)
        self.move_rel_btn.clicked.connect(self.on_move_rel_clicked)
        self.estop_btn.clicked.connect(self.on_estop_clicked)
        self.goto_t0_btn.clicked.connect(self.on_goto_t0_clicked)
        self.define_home_btn.clicked.connect(self.on_define_home_clicked)
        self.load_timefile_btn.clicked.connect(self.on_load_timefile_clicked)
        self.start_scan_btn.clicked.connect(self.on_start_scan_clicked)
        self.stop_scan_btn.clicked.connect(self.on_stop_scan_clicked)
        self.save_btn.clicked.connect(self.on_save_clicked)
        self.calib_run_btn.clicked.connect(self.on_calib_run_clicked)
        self.calib_stop_btn.clicked.connect(self.on_calib_stop_clicked)
        self.calib_zero_phase_btn.clicked.connect(self.on_calib_zero_phase_clicked)

        self.show_current_checkbox.stateChanged.connect(self._redraw_decay_plot)
        self.show_average_checkbox.stateChanged.connect(self._redraw_decay_plot)
        self.show_final_checkbox.stateChanged.connect(self._redraw_decay_plot)
        self.show_overlay_checkbox.stateChanged.connect(self._redraw_decay_plot)
        self.heatmap_mode_combo.currentIndexChanged.connect(self._redraw_heatmap)

        self.worker.connected.connect(self.on_connected)
        self.worker.homed.connect(self.on_homed)
        self.worker.demod_configured.connect(self.on_demod_configured)
        self.worker.point_done.connect(self.on_point_done)
        self.worker.scan_repeat_done.connect(self.on_scan_repeat_done)
        self.worker.averaging_finished.connect(self.on_averaging_finished)
        self.worker.log.connect(self.append_log)
        self.worker.status_update.connect(self.on_status_update)
        self.worker.save_finished.connect(self.on_save_finished)
        self.worker.t0_changed.connect(self.on_t0_changed)
        self.worker.calib_point_done.connect(self.on_calib_point_done)
        self.worker.calib_finished.connect(self.on_calib_finished)
        self.worker.phase_zeroed.connect(self.on_phase_zeroed)

    def append_log(self, message):
        self.log_box.append(message)

    # ------------------------------------------------------------------
    def _build_scan_points(self):
        
        if self.scan_mode_combo.currentText() == "3-Section":
            return self._build_multisection_points()

        start = self.start_input.value()
        stop = self.stop_input.value()
        step = abs(self.step_input.value())
        if step == 0:
            return None, "Step must be nonzero"

        n = int(round(abs(stop - start) / step)) + 1
        sign = 1.0 if stop >= start else -1.0
        forward = [start + sign * step * i for i in range(n)]
        forward[-1] = stop  # land exactly on stop, not a float-accumulated overshoot

        direction = self.direction_combo.currentText()
        if direction == "Forward":
            return forward, None
        elif direction == "Reverse":
            return list(reversed(forward)), None
        else:  # Bidirectional
            return forward + list(reversed(forward)), None

    def _read_section_params(self):
        
        params = []
        for i in range(3):
            start = self.section_start_inputs[i].value()
            stop = self.section_stop_inputs[i].value()
            step = self.section_step_inputs[i].value()
            points = self.section_points_inputs[i].value()
            scale = self.section_scale_combos[i].currentText()
            if stop < start:
                raise ValueError(f"Section {i+1}: stop ({stop}) is before start ({start}).")
            if scale == "Linear" and step <= 0:
                raise ValueError(f"Section {i+1}: step must be > 0 for Linear scale.")
            if scale == "Log" and (start <= 0 or stop <= 0):
                raise ValueError(f"Section {i+1}: Log scale needs Start and Stop both > 0.")
            params.append((start, stop, step, points, scale))
        return params

    def _build_multisection_points(self):
        try:
            params = self._read_section_params()
            all_times, _sections = build_multi_section_points(params)
            return all_times, None
        except ValueError as e:
            return None, str(e)

    def _update_scan_summary(self):
        
        points, error = self._build_scan_points()
        if error or points is None:
            self.n_points_label.setText("Points: --")
            self.scan_duration_label.setText(f"Est. duration: -- ({error})")
            return

        n_points = len(points)
        n_scans = self.n_scans_input.value()
        n_readings = self.n_readings_input.value()
        self.n_points_label.setText(
            f"Points: {n_points} \u00d7 {n_scans} scans  "
            f"(range {points[0]:.4g} \u2192 {points[-1]:.4g} ps)"
        )

        # Use the engine's own estimator if one exists (post-connect, it
        # has a measured per-reading time); otherwise fall back to the
        # same conservative default ScanEngine itself uses internally.
        per_reading_s = 0.05
        engine = self.worker.engine
        if engine is not None and engine._last_reading_duration_s is not None:
            per_reading_s = engine._last_reading_duration_s
        settle_estimate_s = 0.3
        per_point_s = settle_estimate_s + n_readings * per_reading_s
        total_s = n_points * n_scans * per_point_s

        mins, secs = divmod(int(total_s), 60)
        hours, mins = divmod(mins, 60)
        duration_str = f"{hours}h {mins}m {secs}s" if hours else f"{mins}m {secs}s"

        finish_dt = QtCore.QDateTime.currentDateTime().addSecs(int(total_s))
        finish_str = finish_dt.toString("HH:mm:ss")

        self.scan_duration_label.setText(
            f"Est. duration: {duration_str}   |   Est. finish: {finish_str}"
        )

    # ------------------------------------------------------------------
    def _request_status_poll(self):
        QtCore.QMetaObject.invokeMethod(
            self.worker, "poll_status", QtCore.Qt.QueuedConnection
        )

    def on_status_update(self, status):
        stage = status["stage"]
        if not stage["connected"]:
            _status_chip(self.stage_status_chip, "STAGE: not connected", "#9e9e9e")
            _status_chip(self.stage_status_chip_top, "STAGE: --", "#9e9e9e")
            self.current_pos_label.setText("-- mm")
            self.current_delay_label.setText("-- ps")
        elif stage["position_mm"] is None:
            # Busy state from poll_status() -- a calibration scan (or
            # any future direct-stage-command sequence) is in progress
            # and the status poll deliberately skipped querying the
            # stage to avoid desyncing the Thorlabs APT protocol's
            # request/reply pairing (see poll_status()'s docstring).
            # Show a clear "busy" indicator rather than crashing on
            # None where a position was expected, or showing a stale
            # last-known value that could be mistaken for current.
            _status_chip(self.stage_status_chip, "STAGE: busy (calibration)", "#1565c0")
            _status_chip(self.stage_status_chip_top, "STAGE: busy", "#1565c0")
            self.current_pos_label.setText("-- mm (busy)")
            self.current_delay_label.setText("-- ps (busy)")
        else:
            pos_mm = stage["position_mm"]
            self.current_pos_label.setText(f"{pos_mm:.4f} mm")
            # Convert back to ps using DelayStage.mm_to_ps() -- the
            # shared inverse of _ps_to_mm(), rather than re-deriving
            # the formula independently here.
            stage_obj = self.worker.stage
            if stage_obj is not None:
                delay_ps = stage_obj.mm_to_ps(pos_mm)
                self.current_delay_label.setText(f"{delay_ps:.4f} ps")

            if not stage["homed"]:
                _status_chip(self.stage_status_chip,
                             f"STAGE: NOT HOMED ({pos_mm:.4f} mm)", "#f57c00")
                _status_chip(self.stage_status_chip_top, "STAGE: NOT HOMED", "#f57c00")
            elif stage["is_moving"]:
                _status_chip(self.stage_status_chip,
                             f"STAGE: MOVING ({pos_mm:.4f} mm)", "#1565c0")
                _status_chip(self.stage_status_chip_top, "STAGE: MOVING", "#1565c0")
            else:
                _status_chip(self.stage_status_chip,
                             f"STAGE: idle ({pos_mm:.4f} mm)", "#2e7d32")
                _status_chip(self.stage_status_chip_top, "STAGE: idle", "#2e7d32")

        input_status = status["input"]
        if input_status is None:
            _status_chip(self.overload_chip, "INPUT: --", "#9e9e9e")
            self.input_level_label.setText("--")
        elif input_status["near_limit"]:
            _status_chip(self.overload_chip, "INPUT: NEAR OVERLOAD", "#c62828")
            self.input_level_label.setText(
                f"min={input_status['min']:.2f}  max={input_status['max']:.2f}  "
                f"(full scale=\u00b11.0)"
            )
        else:
            _status_chip(self.overload_chip, "INPUT: OK", "#2e7d32")
            self.input_level_label.setText(
                f"min={input_status['min']:.2f}  max={input_status['max']:.2f}"
            )

        if self.worker.lockin is not None and self.worker.lockin.is_connected:
            _status_chip(self.lockin_conn_chip, "LOCK-IN: connected", "#2e7d32")
        else:
            _status_chip(self.lockin_conn_chip, "LOCK-IN: not connected", "#9e9e9e")

        sample = status.get("sample")
        phase_offset_deg = status.get("phase_offset_deg")
        if sample is not None:
            self.x_label.setText(f"{sample['x']:.4e} V")
            self.y_label.setText(f"{sample['y']:.4e} V")
            self.r_label.setText(f"{sample['r']:.4e} V")
            self.theta_label.setText(f"{sample['theta_deg']:.2f} \u00b0")
            self.calib_lockin_readout_label.setText(
                f"{sample['x']:.4e} / {sample['y']:.4e} / "
                f"{sample['r']:.4e} / {sample['theta_deg']:.2f}\u00b0"
            )
        if phase_offset_deg is not None:
            self.calib_phaseshift_label.setText(f"{phase_offset_deg:.3f}")

        reference_status = status["reference"]
        if reference_status is None:
            _status_chip(self.chopper_chip, "REFERENCE: not yet read", "#9e9e9e")
            self.chopper_freq_label.setText("-- Hz")
        else:
            locked = reference_status["locked"]
            freq = reference_status["freq_hz"]
            self.chopper_freq_label.setText(
                f"{freq:.2f} Hz" if freq is not None else "-- Hz"
            )
            if locked is None:
                _status_chip(self.chopper_chip, "REFERENCE: unavailable", "#9e9e9e")
            elif locked:
                _status_chip(self.chopper_chip, "REFERENCE: LOCKED", "#2e7d32")
            else:
                _status_chip(self.chopper_chip, "REFERENCE: LOST", "#c62828")

        laser_status = status["laser"]
        if laser_status is not None and laser_status.get("available"):
            _status_chip(self.laser_chip, "LASER MONITOR: OK", "#2e7d32")
            self.laser_power_label.setText(f"{laser_status['power_mw']:.2f} mW")
            self.laser_stability_label.setText(f"{laser_status['stability_pct']:.2f} %")
        else:
            _status_chip(self.laser_chip, "LASER MONITOR: not available", "#9e9e9e")
            self.laser_power_label.setText("-- mW")
            self.laser_stability_label.setText("-- %")

    # ------------------------------------------------------------------
    def on_connect_clicked(self):
        self.append_log("Connecting...")
        self.connect_btn.setEnabled(False)
        QtCore.QMetaObject.invokeMethod(
            self.worker, "connect_hardware", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, self.serial_input.text()),
            QtCore.Q_ARG(float, self.t0_input.value()),
            QtCore.Q_ARG(str, self.device_id_input.text()),
        )

    def on_connected(self, success, message):
        self.append_log(message)
        if success:
            self.status_label.setText("Connected")
            self.status_label.setStyleSheet("color: green;")
            self.home_btn.setEnabled(True)
            self.configure_demod_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(True)
            self.move_abs_btn.setEnabled(True)
            self.move_rel_btn.setEnabled(True)
            self.estop_btn.setEnabled(True)
            self.goto_t0_btn.setEnabled(True)
            self.define_home_btn.setEnabled(True)
            for btn in self.tzero_jog_buttons:
                btn.setEnabled(True)
            self.calib_run_btn.setEnabled(True)
            self.calib_zero_phase_btn.setEnabled(True)
            self.serial_input.setEnabled(False)
            self.device_id_input.setEnabled(False)
            self._redraw_calibration_plot()
        else:
            self.status_label.setText("Connection failed")
            self.status_label.setStyleSheet("color: red;")
            self.connect_btn.setEnabled(True)

    def on_disconnect_clicked(self):
        QtCore.QMetaObject.invokeMethod(
            self.worker, "disconnect_hardware", QtCore.Qt.QueuedConnection
        )
        self.status_label.setText("Not connected")
        self.status_label.setStyleSheet("color: gray;")
        for w in [self.home_btn, self.configure_demod_btn, self.disconnect_btn,
                  self.start_scan_btn, self.move_abs_btn, self.move_rel_btn,
                  self.estop_btn, self.goto_t0_btn, self.define_home_btn,
                  self.calib_run_btn, self.calib_zero_phase_btn] + self.tzero_jog_buttons:
            w.setEnabled(False)
        self.connect_btn.setEnabled(True)
        self.serial_input.setEnabled(True)
        self.device_id_input.setEnabled(True)
        self._redraw_calibration_plot()

    # ------------------------------------------------------------------
    def on_home_clicked(self):
        self.append_log("Homing...")
        QtCore.QMetaObject.invokeMethod(
            self.worker, "home_stage", QtCore.Qt.QueuedConnection
        )

    def on_homed(self, success, message):
        self.append_log(message)
        self._check_ready_for_scan()

    # ------------------------------------------------------------------
    def on_configure_demod_clicked(self):
        QtCore.QMetaObject.invokeMethod(
            self.worker, "configure_demod", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(float, self.rate_input.value()),
        )

    def on_demod_configured(self, success, message, actual_rate):
        if success:
            self.append_log(f"{message} (actual rate: {actual_rate:.2f} Sa/s)")
        else:
            self.append_log(message)
        self._check_ready_for_scan()

    def _check_ready_for_scan(self):
        """Gates the Control tab's Start button on connection state and
        on self._any_scan_running. Called after connect/home/demod-config,
        and again whenever the scan starts or finishes."""
        ready = (self.status_label.text() == "Connected") and not self._any_scan_running
        self.start_scan_btn.setEnabled(ready)

    # ------------------------------------------------------------------
    def on_move_abs_clicked(self):
        """
        Direct, single-point move -- distinct from running a scan. Uses
        DelayStage.move_to() via a blocking queued call so the GUI thread
        waits for the move to actually finish before re-enabling controls,
        matching how Home already behaves (home_stage() is also blocking
        on the worker thread, with the GUI just waiting on its signal).
        """
        target_ps = self.target_delay_input.value()
        self.append_log(f"Moving to {target_ps:.4f} ps...")
        QtCore.QMetaObject.invokeMethod(
            self.worker, "manual_move_absolute", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(float, target_ps),
        )

    def on_move_rel_clicked(self):
        delta_ps = self.jog_step_input.value()
        self.append_log(f"Jogging by {delta_ps:+.4f} ps...")
        QtCore.QMetaObject.invokeMethod(
            self.worker, "manual_move_relative", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(float, delta_ps),
        )

    def on_estop_clicked(self):
        """
        Emergency stop -- requests abort on the engine (covers an
        in-progress scan) AND directly tells the worker to stop the stage
        motion immediately, rather than waiting for the current move's
        wait_move() to return on its own.
        """
        self.append_log("EMERGENCY STOP requested.")
        self.worker.request_stop()
        QtCore.QMetaObject.invokeMethod(
            self.worker, "emergency_stop_stage", QtCore.Qt.QueuedConnection
        )

    # ------------------------------------------------------------------
    def on_tzero_jog_clicked(self, delta_ps):
        QtCore.QMetaObject.invokeMethod(
            self.worker, "jog_time_zero", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(float, delta_ps),
        )

    def on_goto_t0_clicked(self):
        self.append_log("Moving to t0 (zero delay)...")
        QtCore.QMetaObject.invokeMethod(
            self.worker, "goto_t0", QtCore.Qt.QueuedConnection
        )

    def on_define_home_clicked(self):
        reply = QtWidgets.QMessageBox.question(
            self, "Define HOME here",
            "This redefines t0 (zero delay) to be the stage's CURRENT "
            "position. The stage will NOT move -- only the ps<->mm "
            "reference changes.\n\nContinue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            QtCore.QMetaObject.invokeMethod(
                self.worker, "define_home_here", QtCore.Qt.QueuedConnection
            )

    def on_t0_changed(self, new_t0):
        """
        Keeps every displayed copy of t0 in sync after a jog/define-home
        changes it on the worker's DelayStage -- the Settings dialog's
        t0_input spinbox and the Calibration tab's header/plot both need
        to reflect the new value rather than show a stale one typed in
        at Connect time.
        """
        self.t0_input.blockSignals(True)
        self.t0_input.setValue(new_t0)
        self.t0_input.blockSignals(False)
        if hasattr(self, "calib_header_label"):
            self._redraw_calibration_plot()

    # ------------------------------------------------------------------
    def _redraw_calibration_plot(self):
        
        t0 = self.t0_input.value()
        # DelayStage's own MM_PER_PS/MAX_MM constants -- read off the
        # real class rather than re-deriving them, so this plot can
        # never silently drift out of sync with the actual conversion
        # move_to() uses.
        mm_per_ps = DelayStage.MM_PER_PS
        min_mm = DelayStage.MIN_MM
        max_mm = DelayStage.MAX_MM

        # Theoretical line across the full mechanical travel range.
        mm_line = np.linspace(min_mm, max_mm, 200)
        ps_line = t0 - mm_line / mm_per_ps
        self.calib_line_theory.set_data(ps_line, mm_line)

        t_at_mm0 = t0
        t_at_mm_max = t0 - max_mm / mm_per_ps
        t_min, t_max = min(t_at_mm0, t_at_mm_max), max(t_at_mm0, t_at_mm_max)

        if self.calib_band is not None:
            try:
                self.calib_band.remove()
            except Exception:
                pass
        self.calib_band = self.calib_ax.axvspan(
            t_min, t_max, color="#90caf9", alpha=0.3, label="Reachable range", zorder=0
        )

        stage_obj = self.worker.stage
        if stage_obj is not None and stage_obj.is_connected:
            try:
                current_mm = stage_obj.get_position_mm()
                current_ps = t0 - current_mm / mm_per_ps
                self.calib_point_current.set_data([current_ps], [current_mm])
            except Exception:
                self.calib_point_current.set_data([], [])
        else:
            self.calib_point_current.set_data([], [])

        self.calib_scatter_live.set_data(self._calib_commanded_ps, self._calib_actual_mm)

        self.calib_ax.relim()
        self.calib_ax.autoscale_view()
        self.calib_ax.set_title(f"Time-distance calibration (t0 = {t0:.2f})", fontsize=12)
        self.calib_canvas.draw_idle()

        self.calib_header_label.setText(
            f"t0 = {t0:.4f} ps  |  slope = {mm_per_ps:.6f} mm/ps  |  "
            f"reachable delay range: [{t_min:.4f}, {t_max:.4f}] ps "
            f"({min_mm:.0f}-{max_mm:.0f} mm travel)"
        )

    # ------------------------------------------------------------------
    def on_calib_run_clicked(self):
        n_points = self.calib_points_input.value()
        dwell_s = self.calib_dwell_input.value()
        self._calib_commanded_ps = []
        self._calib_actual_mm = []
        self.calib_progress_bar.setMaximum(n_points)
        self.calib_progress_bar.setValue(0)
        self.calib_run_btn.setEnabled(False)
        self.calib_stop_btn.setEnabled(True)
        self.append_log(f"Starting calibration scan: {n_points} points, dwell={dwell_s}s")
        QtCore.QMetaObject.invokeMethod(
            self.worker, "run_calibration_scan", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(int, n_points),
            QtCore.Q_ARG(float, dwell_s),
        )

    def on_calib_stop_clicked(self):
        self.worker.request_calib_stop()

    def on_calib_point_done(self, event):
        self._calib_commanded_ps.append(event["commanded_ps"])
        self._calib_actual_mm.append(event["actual_mm"])
        self.calib_progress_bar.setValue(event["index"] + 1)
        self._redraw_calibration_plot()

    def on_calib_finished(self, completed, message):
        self.append_log(message)
        self.calib_run_btn.setEnabled(True)
        self.calib_stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    def on_calib_zero_phase_clicked(self):
        QtCore.QMetaObject.invokeMethod(
            self.worker, "zero_lockin_phase", QtCore.Qt.QueuedConnection
        )

    def on_phase_zeroed(self, result):
        self.calib_phaseshift_label.setText(f"{result['phaseshift_after_deg']:.3f}")

    # ------------------------------------------------------------------
    def on_load_timefile_clicked(self):
        
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load timefile metadata (.tfmeta.json)", "",
            "Timefile metadata (*.tfmeta.json);;All files (*.*)",
        )
        if not path:
            return

        try:
            with open(path, "r") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            QtWidgets.QMessageBox.critical(
                self, "Load failed", f"Could not read this file as JSON:\n\n{e}"
            )
            return

        if "sections" not in meta:
            QtWidgets.QMessageBox.critical(
                self, "Load failed",
                "This file doesn't look like a timefile metadata sidecar "
                "-- it has no 'sections' key."
            )
            return

        sections_meta = meta["sections"]
        if len(sections_meta) != 3:
            QtWidgets.QMessageBox.critical(
                self, "Load failed",
                f"This GUI has 3 fixed section rows, but the loaded file "
                f"has {len(sections_meta)} section(s). Loading a file with "
                f"a different section count is not supported -- edit the "
                f"rows manually instead, or regenerate the timefile with "
                f"exactly 3 sections."
            )
            return

        for i, sec in enumerate(sections_meta):
            missing = [k for k in ("start", "stop", "step", "points", "scale") if k not in sec]
            if missing:
                QtWidgets.QMessageBox.critical(
                    self, "Load failed",
                    f"Section {i+1} in the loaded file is missing: {', '.join(missing)}"
                )
                return
            if sec["scale"] not in ("Linear", "Log"):
                QtWidgets.QMessageBox.critical(
                    self, "Load failed",
                    f"Section {i+1}: scale must be 'Linear' or 'Log', got {sec['scale']!r}."
                )
                return

        # Every row validated -- now safe to overwrite the GUI fields.
        for i, sec in enumerate(sections_meta):
            self.section_start_inputs[i].setValue(float(sec["start"]))
            self.section_stop_inputs[i].setValue(float(sec["stop"]))
            self.section_step_inputs[i].setValue(float(sec["step"]))
            self.section_points_inputs[i].setValue(int(sec["points"]))
            self.section_scale_combos[i].setCurrentText(sec["scale"])

        if "short_t0_ps" in meta:
            self.t0_input.setValue(float(meta["short_t0_ps"]))

        # Switch to 3-Section mode so the loaded scan is immediately
        # the one that'll actually run -- loading a 3-section file
        # while still on the Simple page would silently do nothing
        # useful otherwise.
        self.scan_mode_combo.setCurrentText("3-Section")
        self._update_scan_summary()

        self.append_log(f"Loaded timefile sections from: {os.path.basename(path)}")

    # ------------------------------------------------------------------
    def on_start_scan_clicked(self):
        points, error = self._build_scan_points()
        if error:
            self.append_log(f"Scan setup error: {error}")
            return

        n_scans = self.n_scans_input.value()
        n_readings = self.n_readings_input.value()

        self.append_log(
            f"Starting averaged scan: {len(points)} points "
            f"({self.direction_combo.currentText()}) \u00d7 {n_scans} scans, "
            f"{n_readings} readings/point"
        )

        # Reset all accumulated scan/plot state for this new run.
        self._scans = {}
        self._running_mean_by_point = {}
        self._point_order_ps = list(points)
        self._drift_reference_r = None
        # Mode-2 heatmap history: one snapshot of the running-average X
        # trace per completed scan, appended in on_scan_repeat_done() --
        # row i is what the average looked like right after scan i+1
        # was folded in, distinct from Mode 1's raw per-scan rows.
        self._average_history = []
        self._final_average = None
        self.line_final_x.set_data([], [])
        self.line_final_x.set_visible(False)
        self.line_final_y.set_data([], [])
        self.line_final_y.set_visible(False)
        self._redraw_decay_plot()
        self._redraw_heatmap()
        self._reset_metrics()

        self.avg_progress_bar.setMaximum(len(points) * n_scans)
        self.avg_progress_bar.setValue(0)
        self.avg_progress_label.setText(f"Completed: 0/{n_scans}")
        self.start_scan_btn.setEnabled(False)
        self.stop_scan_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        _status_chip(self.engine_chip, "ENGINE: running", "#1565c0")
        self._any_scan_running = True

        self._scan_start_time = time.time()
        self._scan_n_points_total = len(points)
        self._scan_n_scans_total = n_scans
        self.eta_label.setText("Remaining: estimating...")

        QtCore.QMetaObject.invokeMethod(
            self.worker, "run_averaged_scan", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(object, points),
            QtCore.Q_ARG(int, n_readings),
            QtCore.Q_ARG(int, n_scans),
        )

    def on_stop_scan_clicked(self):
        self.append_log("Stop requested...")
        self.worker.request_stop()

    # ------------------------------------------------------------------
    def on_point_done(self, event):
        
        scan_index = event["scan_index"]
        point_index = event["point_index"]
        time_ps = event["time_ps"]
        final = event["final"]
        running_mean = event["running_mean"]

        scan_record = self._scans.setdefault(
            scan_index, {"times_ps": [], "x": [], "y": [], "r": []}
        )
        scan_record["times_ps"].append(time_ps)
        scan_record["x"].append(final["x"])
        scan_record["y"].append(final["y"])
        scan_record["r"].append(final["r"])

        # event["n_averaged"] is a sibling of event["running_mean"], not
        # nested inside it -- fold it in here so everything downstream
        # that reads self._running_mean_by_point (the metrics strip, the
        # saved average file) can find n_averaged on the same dict
        # without needing a second parallel lookup.
        running_mean = dict(running_mean)
        running_mean["n_averaged"] = event.get("n_averaged")
        self._running_mean_by_point[point_index] = running_mean

        if self._drift_reference_r is None and point_index == 0:
            self._drift_reference_r = running_mean["r"]

        # Live X/Y/R/theta readout -- shows the latest single reading,
        # i.e. the most recent thing the lock-in actually measured, not
        # the running average (the average is shown on the plot/metrics
        # instead, where "smoothed" is the expected meaning).
        self.x_label.setText(f"{final['x']:.4e} V")
        self.y_label.setText(f"{final['y']:.4e} V")
        self.r_label.setText(f"{final['r']:.4e} V")
        self.theta_label.setText(f"{final['theta_deg']:.2f} \u00b0")

        overall_index = scan_index * event["point_total"] + point_index + 1
        self.avg_progress_bar.setValue(overall_index)

        self._redraw_decay_plot()
        self._update_heatmap_live(scan_index, point_index, final["x"])
        self._update_metrics(event)

        if self._scan_start_time is not None and overall_index > 0:
            total_steps = self._scan_n_points_total * self._scan_n_scans_total
            elapsed = time.time() - self._scan_start_time
            per_step = elapsed / overall_index
            remaining = per_step * (total_steps - overall_index)
            mins, secs = divmod(int(remaining), 60)
            self.eta_label.setText(f"Remaining: ~{mins}m {secs}s")
            self.metrics_eta_label.setText(f"~{mins}m {secs}s")

    def on_scan_repeat_done(self, scan_index, scan_total):
        self.append_log(
            f"Scan {scan_index + 1}/{scan_total} complete."
        )
        self.avg_progress_label.setText(f"Completed: {scan_index + 1}/{scan_total}")

        # Snapshot the running-average X trace as it stands right now,
        # indexed by delay point position -- this becomes one row of the
        # Mode-2 heatmap. Taken here (once per completed scan) rather
        # than on every point, since Mode 2's "averaging scan" axis is
        # explicitly about evolution scan-to-scan, not point-to-point.
        if self._running_mean_by_point:
            n_points = len(self._point_order_ps)
            row = [np.nan] * n_points
            for point_index, mean in self._running_mean_by_point.items():
                if point_index < n_points:
                    row[point_index] = mean["x"]
            self._average_history.append(row)

        self._redraw_heatmap()

    def on_averaging_finished(self, completed, message):
        self.append_log(message)
        self.stop_scan_btn.setEnabled(False)
        self.eta_label.setText("Remaining: --")
        self.results = self.worker.get_results()
        self.save_btn.setEnabled(len(self.results) > 0)
        _status_chip(
            self.engine_chip,
            "ENGINE: idle" if completed else "ENGINE: stopped",
            "#2e7d32" if completed else "#f57c00",
        )
        self._any_scan_running = False
        self._check_ready_for_scan()

        
        if completed and self._running_mean_by_point:
            pts_sorted = sorted(self._running_mean_by_point.items())
            self._final_average = {
                "times_ps": [self._point_order_ps[i] for i, _ in pts_sorted],
                "x": [m["x"] for _, m in pts_sorted],
                "y": [m["y"] for _, m in pts_sorted],
            }
        self._redraw_decay_plot()
        self._redraw_heatmap()

    # ------------------------------------------------------------------
    def _redraw_decay_plot(self):
        
        
        for ln in self._overlay_lines_x:
            ln.remove()
        self._overlay_lines_x = []
        for ln in self._overlay_lines_y:
            ln.remove()
        self._overlay_lines_y = []

        show_current = self.show_current_checkbox.isChecked()
        show_average = self.show_average_checkbox.isChecked()
        show_final = self.show_final_checkbox.isChecked()
        show_overlay = self.show_overlay_checkbox.isChecked()

        latest_scan_index = max(self._scans.keys()) if self._scans else None

        if show_current and latest_scan_index is not None:
            rec = self._scans[latest_scan_index]
            self.line_current_x.set_data(rec["times_ps"], rec["x"])
            self.line_current_y.set_data(rec["times_ps"], rec["y"])
        else:
            self.line_current_x.set_data([], [])
            self.line_current_y.set_data([], [])
        self.line_current_x.set_visible(show_current)
        self.line_current_y.set_visible(show_current)

        if show_average and self._running_mean_by_point:
            pts_sorted = sorted(self._running_mean_by_point.items())
            times_avg = [self._point_order_ps[i] for i, _ in pts_sorted]
            x_avg = [m["x"] for _, m in pts_sorted]
            y_avg = [m["y"] for _, m in pts_sorted]
            self.line_average_x.set_data(times_avg, x_avg)
            self.line_average_y.set_data(times_avg, y_avg)
        else:
            self.line_average_x.set_data([], [])
            self.line_average_y.set_data([], [])
        self.line_average_x.set_visible(show_average)
        self.line_average_y.set_visible(show_average)

        if show_final and self._final_average is not None:
            self.line_final_x.set_data(
                self._final_average["times_ps"], self._final_average["x"]
            )
            self.line_final_y.set_data(
                self._final_average["times_ps"], self._final_average["y"]
            )
            self.line_final_x.set_visible(True)
            self.line_final_y.set_visible(True)
        else:
            self.line_final_x.set_visible(False)
            self.line_final_y.set_visible(False)

        if show_overlay:
            cmap = mpl.colormaps["viridis"]
            n_scans_so_far = len(self._scans)
            for i, (scan_idx, rec) in enumerate(sorted(self._scans.items())):
                color = cmap(i / max(1, n_scans_so_far - 1)) if n_scans_so_far > 1 else cmap(0.5)
                ln_x, = self.decay_ax_x.plot(
                    rec["times_ps"], rec["x"], "-", color=color, linewidth=0.8, alpha=0.6
                )
                self._overlay_lines_x.append(ln_x)
                ln_y, = self.decay_ax_y.plot(
                    rec["times_ps"], rec["y"], "-", color=color, linewidth=0.8, alpha=0.6
                )
                self._overlay_lines_y.append(ln_y)

        self.decay_ax_x.relim()
        self.decay_ax_x.autoscale_view()
        self.decay_ax_y.relim()
        self.decay_ax_y.autoscale_view()
        self.decay_canvas.draw_idle()

        self._redraw_residual_plot()

    # ------------------------------------------------------------------
    def _redraw_residual_plot(self):
        
        latest_scan_index = max(self._scans.keys()) if self._scans else None

        if latest_scan_index is not None and self._running_mean_by_point:
            rec = self._scans[latest_scan_index]
            residual_times = []
            residual_vals = []
            for point_index, t_ps in enumerate(rec["times_ps"]):
                if point_index in self._running_mean_by_point:
                    avg_x = self._running_mean_by_point[point_index]["x"]
                    residual_times.append(t_ps)
                    residual_vals.append(rec["x"][point_index] - avg_x)
            self.line_residual.set_data(residual_times, residual_vals)
        else:
            self.line_residual.set_data([], [])
        self.residual_ax.relim()
        self.residual_ax.autoscale_view()
        self.residual_canvas.draw_idle()

    # ------------------------------------------------------------------
    def _build_decay_plot_panel(self):
        
        group = QtWidgets.QGroupBox("Kinetics (X / Y Signal)")
        v = QtWidgets.QVBoxLayout(group)

        toolbar = QtWidgets.QHBoxLayout()
        self.show_current_checkbox = QtWidgets.QCheckBox("Current scan")
        self.show_current_checkbox.setChecked(True)
        self.show_average_checkbox = QtWidgets.QCheckBox("Running average")
        self.show_average_checkbox.setChecked(True)
        self.show_final_checkbox = QtWidgets.QCheckBox("Final average")
        self.show_final_checkbox.setChecked(True)
        self.show_overlay_checkbox = QtWidgets.QCheckBox("Overlay all scans")
        self.show_overlay_checkbox.setChecked(False)
        toolbar.addWidget(self.show_current_checkbox)
        toolbar.addWidget(self.show_average_checkbox)
        toolbar.addWidget(self.show_final_checkbox)
        toolbar.addWidget(self.show_overlay_checkbox)
        toolbar.addStretch(1)
        v.addLayout(toolbar)

        
        self.decay_figure = Figure(figsize=(7, 5.6))
        self.decay_figure.subplots_adjust(
            left=0.11, right=0.97, top=0.96, bottom=0.09, hspace=0.08
        )
        self.decay_canvas = FigureCanvasQTAgg(self.decay_figure)

        self.decay_ax_x = self.decay_figure.add_subplot(211)
        self.decay_ax_y = self.decay_figure.add_subplot(212, sharex=self.decay_ax_x)

        self.decay_ax_x.set_ylabel("X (V)")
        self.decay_ax_x.grid(True, alpha=0.3)
        
        for lbl in self.decay_ax_x.get_xticklabels():
            lbl.set_visible(False)

        self.decay_ax_y.set_xlabel("Delay (ps)")
        self.decay_ax_y.set_ylabel("Y (V)")
        self.decay_ax_y.grid(True, alpha=0.3)

        self.line_current_x, = self.decay_ax_x.plot(
            [], [], "-", color="#9e9e9e", linewidth=1, label="Current Scan"
        )
        self.line_average_x, = self.decay_ax_x.plot(
            [], [], "-", color="#1565c0", linewidth=2, label="Running Average"
        )
        self.line_final_x, = self.decay_ax_x.plot(
            [], [], "-", color="#c62828", linewidth=2, label="Final Average"
        )
        self.line_final_x.set_visible(False)  # nothing to show until a run completes
        self.decay_ax_x.legend(loc="best", fontsize=8)

        self.line_current_y, = self.decay_ax_y.plot(
            [], [], "-", color="#9e9e9e", linewidth=1, label="Current Scan"
        )
        self.line_average_y, = self.decay_ax_y.plot(
            [], [], "-", color="#1565c0", linewidth=2, label="Running Average"
        )
        self.line_final_y, = self.decay_ax_y.plot(
            [], [], "-", color="#c62828", linewidth=2, label="Final Average"
        )
        self.line_final_y.set_visible(False)
        self.decay_ax_y.legend(loc="best", fontsize=8)

        self._overlay_lines_x = []  # matplotlib Line2D objects, top (X) subplot
        self._overlay_lines_y = []  # matplotlib Line2D objects, bottom (Y) subplot
        self._final_average = None  # {"times_ps","x","y"} once frozen

        v.addWidget(self.decay_canvas)
        return group

    # ------------------------------------------------------------------
    def _build_residual_panel(self):
        """Residual = current scan - running average, per the mockup's panel 3."""
        group = QtWidgets.QGroupBox("Residual (Current \u2212 Average)")
        v = QtWidgets.QVBoxLayout(group)

        self.residual_figure = Figure(figsize=(7, 2.2))
        # Same fixed-margin rationale as decay_figure above -- a taller
        # bottom margin here since this panel is shorter overall, so the
        # x-label needs proportionally more reserved space to avoid
        # being clipped against the canvas edge.
        self.residual_figure.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.22)
        self.residual_canvas = FigureCanvasQTAgg(self.residual_figure)
        self.residual_ax = self.residual_figure.add_subplot(111)
        self.residual_ax.set_xlabel("Delay (ps)")
        self.residual_ax.set_ylabel("Residual (V)")
        self.residual_ax.grid(True, alpha=0.3)
        self.residual_ax.axhline(0, color="black", linewidth=0.8)
        self.line_residual, = self.residual_ax.plot([], [], "-", color="#7b1fa2",
                                                       linewidth=1)
        v.addWidget(self.residual_canvas)
        return group

    # ------------------------------------------------------------------
    def _build_heatmap_panel(self):
       
        group = QtWidgets.QGroupBox("2D Heatmap (Delay vs Scan Number)")
        v = QtWidgets.QVBoxLayout(group)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Mode:"))
        self.heatmap_mode_combo = QtWidgets.QComboBox()
        self.heatmap_mode_combo.addItems(
            ["Delay vs Scan Number", "Delay vs Averaging Scan"]
        )
        mode_row.addWidget(self.heatmap_mode_combo)
        mode_row.addStretch(1)
        v.addLayout(mode_row)

        self.heatmap_figure = Figure(figsize=(6, 7))
        
        self.heatmap_figure.subplots_adjust(left=0.13, right=0.84, top=0.95, bottom=0.10)
        self.heatmap_canvas = FigureCanvasQTAgg(self.heatmap_figure)
        self.heatmap_ax = self.heatmap_figure.add_subplot(111)
        self.heatmap_ax.set_xlabel("Delay (ps)")
        self.heatmap_ax.set_ylabel("Scan Number")
        self._heatmap_image = None       # the current AxesImage, recreated on redraw
        self._heatmap_colorbar = None    # recreated alongside it
        self._heatmap_cax = None         # dedicated fixed-position colorbar axes (created once)
        self._heatmap_grid = None        # the live-mutable backing array (see _update_heatmap_live)
        self._heatmap_mode_at_build = None
        v.addWidget(self.heatmap_canvas)
        return group

    # ------------------------------------------------------------------
    def _redraw_heatmap(self):
       
        self.heatmap_ax.clear()
        self.heatmap_ax.set_xlabel("Delay (ps)")
        self.heatmap_ax.set_ylabel("Scan Number")

        mode = self.heatmap_mode_combo.currentText()
        n_points = len(self._point_order_ps)

        if mode == "Delay vs Averaging Scan":
            if not self._average_history or n_points == 0:
                self._heatmap_image = None
                self._heatmap_grid = None
                self.heatmap_canvas.draw_idle()
                return
            grid = np.array(self._average_history, dtype=float)
            n_rows = grid.shape[0]
            self.heatmap_ax.set_ylabel("Averaging Scan Number")
        else:
            if not self._scans or n_points == 0:
                self._heatmap_image = None
                self._heatmap_grid = None
                self.heatmap_canvas.draw_idle()
                return
            scan_indices = sorted(self._scans.keys())
            n_rows = len(scan_indices)
            grid = np.full((n_rows, n_points), np.nan)
            for row, scan_idx in enumerate(scan_indices):
                rec = self._scans[scan_idx]
                for col, x_val in enumerate(rec["x"]):
                    if col < n_points:
                        grid[row, col] = x_val

        extent = [
            min(self._point_order_ps), max(self._point_order_ps),
            0.5, n_rows + 0.5,
        ]
        img = self.heatmap_ax.imshow(
            grid, aspect="auto", origin="lower", extent=extent, cmap="jet"
        )

        
        if self._heatmap_cax is None:
            # Carve out the colorbar's axes once, in the margin already
            # reserved by subplots_adjust(right=0.84) at construction.
            self._heatmap_cax = self.heatmap_figure.add_axes([0.87, 0.10, 0.03, 0.85])

        if self._heatmap_colorbar is None:
            self._heatmap_colorbar = self.heatmap_figure.colorbar(
                img, cax=self._heatmap_cax, label="Signal (V)"
            )
        else:
            
            self._heatmap_colorbar.update_normal(img)
        self._heatmap_image = img
        self._heatmap_grid = grid          # mutated in place by _update_heatmap_live()
        self._heatmap_mode_at_build = mode  # detect mode changes cheaply elsewhere

        
        self.heatmap_canvas.draw_idle()

    # ------------------------------------------------------------------
    def _update_heatmap_live(self, scan_index, point_index, x_val):
        
        if self.heatmap_mode_combo.currentText() != "Delay vs Scan Number":
            return

        if (self._heatmap_image is None or self._heatmap_grid is None
                or scan_index >= self._heatmap_grid.shape[0]
                or point_index >= self._heatmap_grid.shape[1]):
            # Shape changed (new row needed) -- rebuild once, which will
            # also pick up this point since self._scans was already
            # updated by on_point_done() before this is called.
            self._redraw_heatmap()
            return

        self._heatmap_grid[scan_index, point_index] = x_val
        self._heatmap_image.set_data(self._heatmap_grid)

       
        finite_vals = self._heatmap_grid[np.isfinite(self._heatmap_grid)]
        if finite_vals.size > 0:
            vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
            if vmin == vmax:
                pad = abs(vmin) * 0.05 if vmin != 0 else 1e-9
                vmin, vmax = vmin - pad, vmax + pad
            self._heatmap_image.set_clim(vmin, vmax)

        self.heatmap_canvas.draw_idle()

    # ------------------------------------------------------------------
    def _reset_metrics(self):
        for lbl in (self.snr_value_label, self.noise_value_label,
                    self.drift_value_label, self.peak_value_label,
                    self.fwhm_value_label, self.metrics_eta_label):
            lbl.setText("--")

    def _update_metrics(self, event):
        
        final = event["final"]

        
        r_readings = [r["r"] for r in event["readings"]]
        snr_result = compute_snr(r_readings)
        self.snr_value_label.setText(
            f"{snr_result['snr']:.1f}" if np.isfinite(snr_result["snr"]) else "\u221e"
        )
        self.noise_value_label.setText(f"{snr_result['noise']:.3e} V")

        # Drift: latest running-mean R at the first delay point vs the
        # reference captured at the very start of this run.
        if self._drift_reference_r is not None and 0 in self._running_mean_by_point:
            current_r0 = self._running_mean_by_point[0]["r"]
            drift_pct = compute_drift_pct(self._drift_reference_r, current_r0)
            self.drift_value_label.setText(
                f"{drift_pct:+.2f} %" if drift_pct is not None else "--"
            )

        # Peak / FWHM / time-zero estimate, from the running-average
        # trace accumulated so far (need at least 3 points for FWHM).
        if len(self._running_mean_by_point) >= 1:
            pts_sorted = sorted(self._running_mean_by_point.items())
            times_avg = [self._point_order_ps[i] for i, _ in pts_sorted]
            r_avg = [m["r"] for _, m in pts_sorted]

            pk = find_peak(times_avg, r_avg)
            if pk is not None:
                self.peak_value_label.setText(f"{pk['peak_time_ps']:.4f} ps")

            fwhm = estimate_fwhm(times_avg, r_avg)
            self.fwhm_value_label.setText(
                f"{fwhm:.4f} ps" if fwhm is not None else "--"
            )

    
    SAVE_BASE_FOLDER = r"C:\Users\Lab\Documents\OKE data acquisition"

    def on_save_clicked(self):
        run_name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Run",
            "Run name (used in the subfolder name):",
            QtWidgets.QLineEdit.Normal,
            "scan",
        )
        if not ok or not run_name.strip():
            return
        run_name = run_name.strip()

        timestamp = QtCore.QDateTime.currentDateTime().toString("yyyy-MM-dd_HH-mm-ss")
        # Sanitize the run name for filesystem use -- strip characters
        # that are invalid in Windows folder names, since this becomes
        # part of an actual path on disk.
        safe_run_name = "".join(
            c for c in run_name if c not in '<>:"/\\|?*'
        ).strip() or "scan"
        subfolder_name = f"{safe_run_name}_{timestamp}"
        dest_folder = os.path.join(self.SAVE_BASE_FOLDER, subfolder_name)

        metadata = {
            "t0_ps": self.worker.stage.t0_ps if self.worker.stage else None,
            "n_scans_requested": self._scan_n_scans_total,
            "run_name": run_name,
            "saved_at": timestamp,
        }

        self.append_log(f"Saving to: {dest_folder} ...")
        QtCore.QMetaObject.invokeMethod(
            self.worker, "save_results", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, dest_folder),
            QtCore.Q_ARG(dict, metadata),
            QtCore.Q_ARG(object, dict(self._running_mean_by_point)),
            QtCore.Q_ARG(list, list(self._point_order_ps)),
        )

    def on_save_finished(self, success, message, dest_folder):
        self.append_log(message)
        if success:
            QtWidgets.QMessageBox.information(self, "Save complete", message)
        else:
            QtWidgets.QMessageBox.warning(self, "Save failed", message)

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self._status_timer.stop()
        try:
            self.worker.request_stop()
            QtCore.QMetaObject.invokeMethod(
                self.worker, "disconnect_hardware", QtCore.Qt.BlockingQueuedConnection
            )
        except Exception:
            pass
        self.worker_thread.quit()
        self.worker_thread.wait(2000)
        event.accept()


# ==========================================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ScanEngineGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()