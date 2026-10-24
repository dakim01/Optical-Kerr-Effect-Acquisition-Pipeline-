

import threading
import time

from pylablib.devices.Thorlabs import KinesisMotor

# ThorlabsError's exact import location has moved across pylablib
# versions -- try the common locations and fall back to a bare Exception
# match (by message text only, inside _with_retry) rather than letting
# an ImportError here break the whole module. This mirrors the same
# defensive stance already used elsewhere in this codebase (e.g.
# scan_engine_gui.py's save_full_run() version-mismatch handling).
try:
    from pylablib.devices.Thorlabs import ThorlabsError
except ImportError:
    try:
        from pylablib.devices.Thorlabs.base import ThorlabsError
    except ImportError:
        ThorlabsError = Exception


class StageNotHomedError(RuntimeError):
    """Raised when a move is attempted before the stage has been homed."""
    pass


class StageNotConnectedError(RuntimeError):
    """Raised when an operation is attempted before connect() succeeded."""
    pass


class DelayStage:
   

    MM_PER_PS = 0.29979 / 2.0    
    MIN_MM = 0.0
    MAX_MM = 220.0                
   
    MAX_VELOCITY_MM_S = 20.0
    MAX_ACCELERATION_MM_S2 = 20.0

  
    SCALE = (20000, 134217.73, 13.744)

    @classmethod
    def t0_ps_for_position_mm(cls, position_mm: float, mm_per_ps: float = None) -> float:
        
        factor = cls.MM_PER_PS if mm_per_ps is None else mm_per_ps
        return position_mm / factor

    @classmethod
    def for_t0_position_mm(cls, serial: str, position_mm: float = None,
                            scale=None, is_rack_system: bool = False):
        
        mm = cls.T0_POSITION_MM if position_mm is None else position_mm
        t0_ps = cls.t0_ps_for_position_mm(mm)
        return cls(serial=serial, t0_ps=t0_ps, scale=scale,
                    is_rack_system=is_rack_system)

    def __init__(self, serial: str, t0_ps: float, scale=None,
                 is_rack_system: bool = False):
        
        self.serial = serial
        self.t0_ps = float(t0_ps)
        self.scale = scale if scale is not None else self.SCALE
        self.is_rack_system = is_rack_system

        self._stage = None
        self._homed = False
        # Guards every call that talks to the physical stage over the
        # serial connection, and is the vehicle _with_retry() uses to
        # serialize + retry a call that hits the known APT protocol
        # desync (see _with_retry()'s docstring). A plain
        # threading.Lock rather than anything fancier: every hardware
        # call in this class is already routed through a single
        # ScanWorker background thread by the GUI (queued Qt
        # connections), so this isn't defending against real
        # concurrent Python callers -- it's here so a retry's re-issued
        # command and any other in-flight call to this object can never
        # physically interleave on the wire, which is the condition
        # that produces the desync in the first place.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._stage is not None

    @property
    def is_homed(self) -> bool:
        return self._homed

    # ------------------------------------------------------------------
    def connect(self):
        
        self._stage = KinesisMotor(self.serial, scale=self.scale,
                                    is_rack_system=self.is_rack_system)
        self._homed = False
        return self

    def home(self, timeout: float = 120.0):
        
        self._require_connected()
        self._stage.home(force=True)
        self._stage.wait_for_home(timeout=timeout)
        self._homed = True

    def define_home_here(self) -> float:
        
        self._require_connected()
        current_mm = self.get_position_mm()
        # Invert _ps_to_mm(): mm = (t0_ps - time_ps) * MM_PER_PS, and at
        # the moment this is called time_ps == 0 by definition (this
        # position IS the new zero delay) -- so t0_ps = current_mm / MM_PER_PS.
        self.t0_ps = current_mm / self.MM_PER_PS
        return self.t0_ps

    def close(self):
        """Disconnect from the stage."""
        if self._stage is not None:
            self._stage.close()
        self._stage = None
        self._homed = False

    def stop(self):
        
        self._require_connected()
        self._stage.stop()

    # ------------------------------------------------------------------
    def move_to(self, time_ps: float) -> float:
        
        self._require_homed()

        target_mm = self._ps_to_mm(time_ps)
        target_mm = max(self.MIN_MM, min(self.MAX_MM, target_mm))

        self._stage.move_to(target_mm)
        self._stage.wait_move()

        return self._stage.get_position()

    def jog(self, delta_ps: float, current_time_ps: float) -> float:
      
        return self.move_to(current_time_ps + delta_ps)

    def get_position_mm(self) -> float:
        """Current stage position, in mm, as read back from the controller."""
        self._require_connected()
        return self._stage.get_position()

    # ------------------------------------------------------------------
    def set_velocity_params(self, max_velocity_mm_s: float,
                             acceleration_mm_s2: float = None) -> dict:
        
        self._require_connected()

        requested_velocity = float(max_velocity_mm_s)
        requested_accel = None if acceleration_mm_s2 is None else float(acceleration_mm_s2)

        applied_velocity = requested_velocity
        applied_accel = requested_accel
        clamped = False

        if applied_velocity > self.MAX_VELOCITY_MM_S:
            applied_velocity = self.MAX_VELOCITY_MM_S
            clamped = True
        if applied_velocity <= 0:
            raise ValueError(
                f"max_velocity_mm_s must be > 0 (got {applied_velocity})."
            )

        if applied_accel is not None and applied_accel > self.MAX_ACCELERATION_MM_S2:
            applied_accel = self.MAX_ACCELERATION_MM_S2
            clamped = True

        try:
            result = self._stage.setup_velocity(
                max_velocity=applied_velocity,
                acceleration=applied_accel,
            )
        except TypeError as e:
            raise TypeError(
                "DelayStage.set_velocity_params(): KinesisMotor.setup_velocity() "
                f"rejected these arguments ({e}). Check "
                "help(pylablib.devices.Thorlabs.KinesisMotor.setup_velocity) "
                "for the signature on your installed pylablib version."
            )

        out = self._velocity_result_to_dict(result)
        out["clamped"] = clamped
        out["requested_max_velocity_mm_s"] = requested_velocity
        out["requested_acceleration_mm_s2"] = requested_accel
        return out

    def get_velocity_params(self) -> dict:
        """
        Reads back the stage's current velocity parameters WITHOUT
        changing them -- pylablib's setup_velocity() convention is that
        calling it with no arguments returns current settings in
        query-only mode.
        """
        self._require_connected()
        result = self._stage.setup_velocity()
        return self._velocity_result_to_dict(result)

    @staticmethod
    def _velocity_result_to_dict(result) -> dict:
        """
        Normalizes setup_velocity()'s return value into a plain dict, so
        callers (e.g. the GUI) don't need to import pylablib themselves.
        Handles both the namedtuple-style return (recent pylablib) and a
        plain 3-tuple fallback, in that documented field order:
        (min_velocity, acceleration, max_velocity).
        """
        try:
            return {
                "min_velocity_mm_s": result.min_velocity,
                "acceleration_mm_s2": result.acceleration,
                "max_velocity_mm_s": result.max_velocity,
            }
        except AttributeError:
            min_v, accel, max_v = result
            return {
                "min_velocity_mm_s": min_v,
                "acceleration_mm_s2": accel,
                "max_velocity_mm_s": max_v,
            }

    # ------------------------------------------------------------------
    def get_status(self) -> dict:
        """
        Live stage status, for display in a GUI status indicator.

        Returns:
            {
                "connected": bool,
                "homed": bool,
                "is_moving": bool,        # pylablib KinesisMotor.is_moving()
                "position_mm": float,
                "raw_status": list,       # pylablib KinesisMotor.get_status() --
                    a list of status-bit strings (e.g. "moving_fwd",
                    "homed", limit-switch states); kept raw rather than
                    reinterpreted here, since the exact set of strings is
                    controller/firmware dependent and a GUI can display
                    them directly or check membership (e.g.
                    "moving_fwd" in raw_status) as needed.
            }

        If not connected, returns a minimal dict with connected=False and
        all other fields None, rather than raising -- this is meant to be
        safe to call from a GUI status-poll on every refresh tick, even
        before/after a connection exists.
        """
        if self._stage is None:
            return {"connected": False, "homed": False, "is_moving": None,
                    "position_mm": None, "raw_status": None}

        def _do_get_status():
            is_moving = self._stage.is_moving()
            raw_status = self._stage.get_status()
            position_mm = self._stage.get_position()
            return is_moving, raw_status, position_mm

        is_moving, raw_status, position_mm = self._with_retry(_do_get_status)

        return {
            "connected": True,
            "homed": self._homed,
            "is_moving": is_moving,
            "position_mm": position_mm,
            "raw_status": raw_status,
        }

    # ------------------------------------------------------------------
    def reachable_window_ps(self):
        """
        Returns (t_min_ps, t_max_ps): the delay range reachable without
        exceeding the stage's mechanical travel, given the current t0_ps.
        """
        t_at_mm0 = self.t0_ps
        t_at_mm_max = self.t0_ps - self.MAX_MM / self.MM_PER_PS
        return min(t_at_mm0, t_at_mm_max), max(t_at_mm0, t_at_mm_max)

    def check_time(self, time_ps: float) -> bool:
        """True if a single time point maps to a reachable stage position."""
        mm = self._ps_to_mm(time_ps)
        return self.MIN_MM <= mm <= self.MAX_MM

    def check_times(self, times_ps) -> bool:
        """True if every time point in times_ps is reachable."""
        return all(self.check_time(t) for t in times_ps)

    # ------------------------------------------------------------------
    def _with_retry(self, fn, *args, max_retries=2, retry_delay_s=0.05, **kwargs):
        """
        Runs fn(*args, **kwargs) under self._lock, retrying it up to
        max_retries times if it fails with the specific known APT
        protocol desync: a status-bits reply request crossing paths
        with a move-completed notification arriving on the same serial
        connection (observed on this rig as a ThorlabsError reading
        "unexpected command received: expected 0x042a, got 0x0412").

        Deliberately narrow: only THAT specific message text is treated
        as retryable. Any other exception -- including any other
        ThorlabsError, and definitely a ThorlabsBackendError such as the
        Windows "Access is denied" PermissionError seen when another
        process already holds the COM port -- is re-raised immediately
        on the first attempt. Retrying blindly on every error would
        mask genuine hardware/connection failures behind a delay instead
        of surfacing them, which is worse than just failing fast.

        Between attempts, calls self._stage.flush_comm() if the
        underlying pylablib object exposes it, to drain whatever
        stray/out-of-order message caused the desync before re-issuing
        the request -- flush_comm() is itself best-effort (wrapped in
        its own try/except) since not every backend guarantees it.

        Args:
            fn: a zero-argument callable (typically a small local
                closure, e.g. get_status()'s _do_get_status()) that
                performs the actual pylablib call(s) and returns their
                result. Any positional/keyword args are forwarded to it
                as a convenience, though every current caller in this
                file uses a closure instead.
            max_retries: number of RETRY attempts after the first try
                (so up to max_retries + 1 total attempts).
            retry_delay_s: pause before each retry, giving the
                controller a moment to finish emitting whatever
                notification caused the desync.

        Raises:
            Whatever fn ultimately raises, once retries are exhausted
            or the error isn't the specific retryable one.
        """
        last_error = None
        for attempt in range(max_retries + 1):
            with self._lock:
                try:
                    return fn(*args, **kwargs)
                except ThorlabsError as e:
                    if "unexpected command received" not in str(e):
                        raise
                    last_error = e
                    if hasattr(self._stage, "flush_comm"):
                        try:
                            self._stage.flush_comm()
                        except Exception:
                            pass
            if attempt < max_retries:
                time.sleep(retry_delay_s)
        raise last_error

    # ------------------------------------------------------------------
    def _ps_to_mm(self, time_ps: float) -> float:
        return (self.t0_ps - float(time_ps)) * self.MM_PER_PS

    def mm_to_ps(self, position_mm: float) -> float:
        
        return self.t0_ps - float(position_mm) / self.MM_PER_PS

    def _require_connected(self):
        if self._stage is None:
            raise StageNotConnectedError(
                "Delay stage is not connected. Call connect() first."
            )

    def _require_homed(self):
        self._require_connected()
        if not self._homed:
            raise StageNotHomedError(
                "Delay stage has not been homed. Call home() before "
                "move_to()/jog() -- without homing, the controller lacks "
                "an absolute position reference and move commands can "
                "silently produce little or no real motion."
            )