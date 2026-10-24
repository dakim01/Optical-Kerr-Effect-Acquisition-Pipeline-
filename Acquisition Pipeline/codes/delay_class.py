# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:35:26 2026

@author: Lab
"""

#from pylablib.devices.Thorlabs import list_kinesis_devices
from pylablib.devices.Thorlabs import KinesisMotor
from pylablib.devices.Thorlabs import list_kinesis_devices
import pyvisa as visa
import serial
import time as pytime
 

def scan_thorlabs_devices():
    devs = list_kinesis_devices()
    print(devs)
    return devs
devs = list_kinesis_devices()
print(devs) 
class thorlabs_delay_stage:
    
 
    PORT = "COM5"          # ← change if Windows reassigns the COM port
    MAX_MM  = 220.0        # DDS600 has 220 mm stage travel limit in mm
    MIN_MM  = 0.0
  
    def __init__(self, t0):
        self.t0 = t0
        self.initialized = False
        try:
            self.stage = KinesisMotor(self.PORT, scale=(20000, 134217.73, 13.744), is_rack_system=True)
            self.initialized = True
            print(f"Connected to Thorlabs stage on {self.PORT}")
            print(f"  Scale      : {self.stage.get_scale()}")
            print(f"  Scale units: {self.stage.get_scale_units()}")
        except Exception as e:
            print(f"Failed to connect to Thorlabs stage: {e}")
            self.stage = None
 
    # ------------------------------------------------------------------
    def home(self):
        """Send the stage home and wait for completion."""
        if self.stage is None:
            return
        self.stage.home()
        self.stage.wait_for_home(timeout=120)

    def move_to(self, time_point_ps):
        """
        Move to the position corresponding to time_point_ps picoseconds.
        Returns False (kept for API compatibility with the original class).
        """
        if self.stage is None:
            return False
        new_pos_mm = self.convert_ps_to_mm(float(self.t0 - time_point_ps))
        new_pos_mm = max(self.MIN_MM, min(self.MAX_MM, new_pos_mm))
        self.stage.move_to(new_pos_mm)
        self.stage.wait_move()
        return False
 
    # ------------------------------------------------------------------
    def convert_ps_to_mm(self, time_ps):
        """Convert a time delay in picoseconds to a position in mm."""
        # speed of light: 0.29979 mm/ps, factor of 2 for double-pass
        return 0.29979 * time_ps / 2.0
 
    # ------------------------------------------------------------------
    def close(self):
        """Disconnect from the stage."""
        if self.stage is not None:
            self.stage.close()
            self.stage = None
 
    # ------------------------------------------------------------------
    def check_times(self, times):
        """Return True if ALL time points map to valid stage positions."""
        return all(self.check_time(t) for t in times)
 
    def check_time(self, time):
        """Return True if a single time point maps to a valid stage position."""
        pos = self.convert_ps_to_mm(float(self.t0 - time))
        return self.MIN_MM <= pos <= self.MAX_MM
 
 
# ======================================================================
class pink_laser_delay:
    def __init__(self, t0):
        self.gen_gpib_address = 'GPIB::15::INSTR'
        self.rm = visa.ResourceManager()
        self.gen = self.rm.open_resource(self.gen_gpib_address)
        self.t0 = t0
        self.initialized = True
 
    def move_to(self, time_point_ns):
        tau_flip_request = False
        new_time = (self.t0 - time_point_ns) * 1E-9
        if new_time < 0:
            tau_flip_request = True
            new_time = new_time + 0.001
        time_point_string = '%.5e' % new_time
        self.gen.write('DT 2,1,' + time_point_string)
        return tau_flip_request
 
    def check_times(self, times):
        return all(self.check_time(t) for t in times)
 
    def check_time(self, time):
        new_time = (self.t0 - time) * 1E-9
        return -0.001 <= new_time <= 0.001
 
 
# ======================================================================
class disco_laser_delay:
 
    def __init__(self, t0, port, baud):
        self.ser = serial.Serial(port, baud, timeout=1)
        self.t0 = t0
        self.initialized = self.ser.isOpen()
        cmd = ('tl 1; tr po; tr te; td 0; aw 10n; as po; as on \r\n').encode('UTF-8')
        self.ser.write(cmd)
 
    def close(self):
        self.ser.close()
 
    def move_to(self, time_point_ns):
        tau_flip_request = False
        new_time_ns = self.t0 + time_point_ns
        cmd = ('ad ' + str(new_time_ns) + 'n \r\n').encode('UTF-8')
        self.ser.write(cmd)
        self.DG_out = ''
        self.DG_output_state = False
        pytime.sleep(0.05)
        self.DG_out += self.ser.read(100).decode('ascii')
        if self.DG_out != '':
            self.DG_output_state = True
        return tau_flip_request
 
    def check_times(self, times):
        return all(self.check_time(t) for t in times)
 
    def check_time(self, time):
        new_time_ns = self.t0 + time
        return 0 <= new_time_ns <= 1E6