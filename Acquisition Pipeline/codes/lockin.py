# -*- coding: utf-8 -*-

    if isinstance(value, dict):
        for key, sub in value.items():
            m = _CH_RE.match(key)
            segment = m.group(1) if m else key
            _flatten_labone_nodes(f"{prefix}/{segment}" if prefix else segment,
                                   sub, out)
    else:
        out.append((prefix, value))


def _coerce_labone_value(raw_value):
    """Numeric node values are stringified in the export; convert back."""
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return raw_value


class LockinNotConnectedError(RuntimeError):
    """Raised when an operation is attempted before connect() succeeded."""
    pass


class MFLI:
    

    def __init__(self, device_id: str, interface: str,
                 server_host: str = "localhost", server_port: int = 8004,
                 api_level: int = 6):
       
        self.device_id = device_id
        self.interface = interface
        self.server_host = server_host
        self.server_port = server_port
        self.api_level = api_level

        self._daq = None
        self._dt_device = None  # seconds per timestamp tick (timebase)

    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._daq is not None

    # ------------------------------------------------------------------
    def connect(self):
        """Open the Data Server session and attach to the device."""
        self._daq = zhinst.core.ziDAQServer(
            self.server_host, self.server_port, self.api_level
        )
        self._daq.connectDevice(self.device_id, self.interface)
        self._dt_device = self._daq.getDouble(
            f"/{self.device_id}/system/properties/timebase"
        )
        return self

    def close(self):
        """Disconnect from the device."""
        if self._daq is not None:
            try:
                self._daq.unsubscribe("*")
            except Exception:
                pass
            try:
                self._daq.disconnectDevice(self.device_id)
            except Exception:
                pass
        self._daq = None
        self._dt_device = None

    # ------------------------------------------------------------------
    def configure_demod(self, demod_index: int, rate_hz: float):
        """
        Enable a demodulator and request a sample rate. The MFLI quantizes
        the requested rate to its nearest supported value -- call
        get_actual_rate() afterward to see what was actually set.
        """
        self._require_connected()
        self._daq.set([
            (f"/{self.device_id}/demods/{demod_index}/rate", rate_hz),
            (f"/{self.device_id}/demods/{demod_index}/enable", 1),
        ])

    def get_actual_rate(self, demod_index: int) -> float:
        """The demod rate actually in effect, in Sa/s (device-quantized)."""
        self._require_connected()
        return self._daq.getDouble(f"/{self.device_id}/demods/{demod_index}/rate")

    # ------------------------------------------------------------------
    def load_labone_config(self, json_path, exclude_categories=(),
                            allow_serial_mismatch=False, dry_run=False):
       
        self._require_connected()

        with open(json_path, "r") as f:
            doc = json.load(f)

        blocks = doc.get("settings", {}).get("deviceSettings", [])
        if not blocks:
            raise ValueError(
                f"{json_path}: no deviceSettings found in this export -- "
                f"is this a LabOne UI settings export?"
            )

        block = next((b for b in blocks if b["deviceSerial"] == self.device_id),
                     None)
        if block is None:
            if not allow_serial_mismatch:
                found = ", ".join(b["deviceSerial"] for b in blocks)
                raise ValueError(
                    f"{json_path}: no block for device {self.device_id} "
                    f"(found: {found}). Pass allow_serial_mismatch=True to "
                    f"apply the first block's settings to {self.device_id} "
                    f"anyway."
                )
            block = blocks[0]

        flat = []
        _flatten_labone_nodes("", block.get("nodes", {}), flat)

        pairs = []
        for rel_path, raw_value in flat:
            category = rel_path.split("/", 1)[0]
            if category in exclude_categories:
                continue
            full_path = f"/{self.device_id}/{rel_path}"
            pairs.append((full_path, _coerce_labone_value(raw_value)))

        if not dry_run:
            self._daq.set(pairs)

        return pairs

    # ------------------------------------------------------------------
    def get_input_range(self, in_channel: int = 0) -> float:
        """Configured voltage input range, in V (e.g. 0.01, 0.1, 1.0...)."""
        self._require_connected()
        return self._daq.getDouble(f"/{self.device_id}/sigins/{in_channel}/range")

    def get_input_utilization(self, in_channel: int = 0) -> dict:
       
        self._require_connected()
        min_val = self._daq.getDouble(f"/{self.device_id}/sigins/{in_channel}/min")
        max_val = self._daq.getDouble(f"/{self.device_id}/sigins/{in_channel}/max")
        near_limit = (abs(min_val) > 0.9) or (abs(max_val) > 0.9)
        return {"min": min_val, "max": max_val, "near_limit": near_limit}

    # ------------------------------------------------------------------
    def read_sample(self, demod_index: int) -> dict:
        
        self._require_connected()
        sample = self._daq.getSample(f"/{self.device_id}/demods/{demod_index}/sample")
        x = float(sample['x'][0])
        y = float(sample['y'][0])
        r = float(np.abs(x + 1j * y))
        theta_deg = float(np.degrees(np.arctan2(y, x)))
        return {"x": x, "y": y, "r": r, "theta_deg": theta_deg}

    def poll(self, duration_s: float, demod_index: int,
             timeout_ms: int = 2000) -> dict:
       
        self._require_connected()

        path = f"/{self.device_id}/demods/{demod_index}/sample"
        self._daq.subscribe(path)
        try:
            data = self._daq.poll(duration_s, timeout_ms, flat=True)
        finally:
            self._daq.unsubscribe("*")

        if path not in data:
            raise RuntimeError(
                f"poll() returned no data for {path}. Check that the demod "
                f"is enabled (configure_demod) and the signal input is "
                f"connected."
            )

        chunk = data[path]
        x = chunk["x"]
        y = chunk["y"]
        ts = chunk["timestamp"]

        r = np.abs(x + 1j * y)
        theta_deg = np.degrees(np.arctan2(y, x))
        time_s = (ts - ts[0]) * self._dt_device

        return {
            "x": x,
            "y": y,
            "r": r,
            "theta_deg": theta_deg,
            "timestamp": ts,
            "time_s": time_s,
            "n_samples": len(x),
            "achieved_rate_hz": len(x) / duration_s,
        }

    # ------------------------------------------------------------------
    def get_reference_status(self, demod_index: int = 0, extref_index: int = 0) -> dict:
       
        self._require_connected()
        try:
            locked = bool(self._daq.getInt(
                f"/{self.device_id}/extrefs/{extref_index}/locked"
            ))
        except Exception:
            locked = None
        try:
            freq_hz = self._daq.getDouble(
                f"/{self.device_id}/demods/{demod_index}/freq"
            )
        except Exception:
            freq_hz = None
        return {"locked": locked, "freq_hz": freq_hz}

    def get_laser_status(self) -> dict:
        
        return {"available": False, "power_mw": None, "stability_pct": None}

    # ------------------------------------------------------------------
    def _require_connected(self):
        if self._daq is None:
            raise LockinNotConnectedError(
                "MFLI is not connected. Call connect() first."
            )