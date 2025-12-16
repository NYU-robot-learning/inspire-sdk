"""
RH56 Dexterous Robot Hand Controller
RS-485 Communication Protocol

This module provides a clean Python API for controlling the RH56 5-finger dexterous robot hand.
Angles are specified as percentages (0-100) for easier use.
"""

import serial
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HandState:
    """Represents the current state of the robot hand"""
    angles: List[float]  # 0-100%
    forces: List[float]  # 0-100%
    temperatures: List[int]  # Celsius
    error_codes: List[int]
    status_codes: List[int]


class RH56RobotHand:
    """
    Controller for RH56 Dexterous Robot Hand via RS-485
    
    All angles, speeds, and forces use 0-100 scale (percentage).
    Register addresses reference: RH56 User Manual page 11, section 2.4
    """
    
    # Register addresses
    REGISTERS = {
        'ID': 1000,
        'BAUDRATE': 1001,
        'CLEAR_ERROR': 1004,
        'FORCE_CALIBRATE': 1009,
        'ANGLE_SET': 1486,
        'FORCE_SET': 1498,
        'SPEED_SET': 1522,
        'ANGLE_ACTUAL': 1546,
        'FORCE_ACTUAL': 1582,
        'ERROR_CODE': 1606,
        'STATUS_CODE': 1612,
        'TEMPERATURE': 1618,
        'ACTION_SEQ': 2320,
        'ACTION_RUN': 2322
    }
    
    CMD_READ = 0x11
    CMD_WRITE = 0x12
    FRAME_HEADER = [0xEB, 0x90]
    NUM_MOTORS = 6
    
    # Hardware uses 0-1000 scale, we expose 0-100 to users
    _HW_SCALE = 1000
    _USER_SCALE = 100
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200, hand_id: int = 1):
        """
        Initialize the robot hand controller
        
        Args:
            port: Serial port path (e.g., '/dev/ttyUSB0' on Linux, 'COM3' on Windows)
            baudrate: Communication baudrate (default: 115200)
            hand_id: Hand ID number (default: 1)
        """
        self.port = port
        self.baudrate = baudrate
        self.hand_id = hand_id
        self.serial: Optional[serial.Serial] = None
    
    @staticmethod
    def _to_hardware_value(user_value: float) -> int:
        """Convert user scale (0-100) to hardware scale (0-1000)"""
        if user_value == -1:  # Skip flag
            return -1
        return int(user_value * RH56RobotHand._HW_SCALE / RH56RobotHand._USER_SCALE)
    
    @staticmethod
    def _from_hardware_value(hw_value: int) -> float:
        """Convert hardware scale (0-1000) to user scale (0-100)"""
        return round(hw_value * RH56RobotHand._USER_SCALE / RH56RobotHand._HW_SCALE, 1)
        
    def connect(self) -> bool:
        """
        Open serial connection to the robot hand
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            print(f"✓ Connected to robot hand on {self.port}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("✓ Disconnected from robot hand")
    
    def _write_register(self, address: int, data: List[int]) -> bool:
        """
        Write data to robot hand registers (internal method)
        
        Args:
            address: Starting register address
            data: List of bytes to write
            
        Returns:
            True if write successful
        """
        if not self.serial or not self.serial.is_open:
            print("✗ Error: Serial port not open")
            return False
        
        frame = self.FRAME_HEADER.copy()
        frame.append(self.hand_id)
        frame.append(len(data) + 3)
        frame.append(self.CMD_WRITE)
        frame.append(address & 0xFF)
        frame.append((address >> 8) & 0xFF)
        frame.extend(data)
        
        checksum = sum(frame[2:]) & 0xFF
        frame.append(checksum)
        
        self.serial.write(bytes(frame))
        time.sleep(0.01)
        self.serial.read_all()
        
        return True
    
    def _read_register(self, address: int, num_bytes: int) -> Optional[List[int]]:
        """
        Read data from robot hand registers (internal method)
        
        Args:
            address: Starting register address
            num_bytes: Number of bytes to read
            
        Returns:
            List of bytes read, or None if read failed
        """
        if not self.serial or not self.serial.is_open:
            print("✗ Error: Serial port not open")
            return None
        
        frame = self.FRAME_HEADER.copy()
        frame.append(self.hand_id)
        frame.append(0x04)
        frame.append(self.CMD_READ)
        frame.append(address & 0xFF)
        frame.append((address >> 8) & 0xFF)
        frame.append(num_bytes)
        
        checksum = sum(frame[2:]) & 0xFF
        frame.append(checksum)
        
        self.serial.write(bytes(frame))
        time.sleep(0.01)
        
        response = self.serial.read_all()
        if len(response) == 0:
            return None
        
        data_length = (response[3] & 0xFF) - 3
        data = [response[7 + i] for i in range(data_length)]
        
        return data
    
    def set_angles(self, angles: List[float]) -> bool:
        """
        Set target angles for all 6 motors
        
        Args:
            angles: List of 6 angles (0-100%). Use -1 to skip a motor.
                   0 = fully open, 100 = fully closed
                   
        Returns:
            True if command sent successfully
            
        Example:
            hand.set_angles([0, 50, 100, 75, 25, 80])
        """
        if len(angles) != self.NUM_MOTORS:
            print(f"✗ Error: Expected {self.NUM_MOTORS} angles, got {len(angles)}")
            return False
        
        # Convert to hardware scale and byte array
        data = []
        for angle in angles:
            hw_value = self._to_hardware_value(angle)
            data.append(hw_value & 0xFF)
            data.append((hw_value >> 8) & 0xFF)
        
        return self._write_register(self.REGISTERS['ANGLE_SET'], data)
    
    def set_speeds(self, speeds: List[float]) -> bool:
        """
        Set movement speeds for all 6 motors
        
        Args:
            speeds: List of 6 speeds (0-100%). Use -1 to skip a motor.
                   0 = no movement, 100 = maximum speed
                   
        Returns:
            True if command sent successfully
            
        Example:
            hand.set_speeds([50, 50, 50, 50, 50, 50])  # Half speed
        """
        if len(speeds) != self.NUM_MOTORS:
            print(f"✗ Error: Expected {self.NUM_MOTORS} speeds, got {len(speeds)}")
            return False
        
        data = []
        for speed in speeds:
            hw_value = self._to_hardware_value(speed)
            data.append(hw_value & 0xFF)
            data.append((hw_value >> 8) & 0xFF)
        
        return self._write_register(self.REGISTERS['SPEED_SET'], data)
    
    def set_forces(self, forces: List[float]) -> bool:
        """
        Set force thresholds for all 6 motors
        
        Args:
            forces: List of 6 force values (0-100%). Use -1 to skip a motor.
                   0 = minimum force, 100 = maximum force
                   
        Returns:
            True if command sent successfully
            
        Example:
            hand.set_forces([50, 50, 50, 50, 50, 50])  # Medium grip strength
        """
        if len(forces) != self.NUM_MOTORS:
            print(f"✗ Error: Expected {self.NUM_MOTORS} forces, got {len(forces)}")
            return False
        
        data = []
        for force in forces:
            hw_value = self._to_hardware_value(force)
            data.append(hw_value & 0xFF)
            data.append((hw_value >> 8) & 0xFF)
        
        return self._write_register(self.REGISTERS['FORCE_SET'], data)
    
    def get_angles(self) -> Optional[List[float]]:
        """
        Get actual angles from all 6 motors
        
        Returns:
            List of 6 angle values (0-100%), or None if read failed
        """
        data = self._read_register(self.REGISTERS['ANGLE_ACTUAL'], 12)
        if not data or len(data) < 12:
            print("✗ Failed to read angles")
            return None
        
        angles = []
        for i in range(self.NUM_MOTORS):
            hw_value = (data[2*i] & 0xFF) | (data[2*i + 1] << 8)
            angles.append(self._from_hardware_value(hw_value))
        
        return angles
    
    def get_forces(self) -> Optional[List[float]]:
        """
        Get actual forces from all 6 fingers
        
        Returns:
            List of 6 force values (0-100%), or None if read failed
        """
        data = self._read_register(self.REGISTERS['FORCE_ACTUAL'], 12)
        if not data or len(data) < 12:
            print("✗ Failed to read forces")
            return None
        
        forces = []
        for i in range(self.NUM_MOTORS):
            hw_value = (data[2*i] & 0xFF) | (data[2*i + 1] << 8)
            forces.append(self._from_hardware_value(hw_value))
        
        return forces
    
    def get_temperatures(self) -> Optional[List[int]]:
        """
        Get temperatures from all 6 motors
        
        Returns:
            List of 6 temperature values (°C), or None if read failed
        """
        data = self._read_register(self.REGISTERS['TEMPERATURE'], 6)
        if not data or len(data) < 6:
            print("✗ Failed to read temperatures")
            return None
        
        return data
    
    def get_state(self) -> Optional[HandState]:
        """
        Get complete state of the robot hand
        
        Returns:
            HandState object with all sensor data, or None if read failed
        """
        angles = self.get_angles()
        forces = self.get_forces()
        temps = self.get_temperatures()
        
        errors = self._read_register(self.REGISTERS['ERROR_CODE'], 6)
        status = self._read_register(self.REGISTERS['STATUS_CODE'], 6)
        
        if angles and forces and temps and errors and status:
            return HandState(
                angles=angles,
                forces=forces,
                temperatures=temps,
                error_codes=errors,
                status_codes=status
            )
        return None
    
    def open_hand(self, speed: float = 50) -> bool:
        """
        Fully open the hand
        
        Args:
            speed: Movement speed (0-100%, default 50%)
            
        Returns:
            True if command sent successfully
        """
        self.set_speeds([speed] * self.NUM_MOTORS)
        return self.set_angles([0] * self.NUM_MOTORS)
    
    def close_hand(self, speed: float = 50, force: float = 50) -> bool:
        """
        Fully close the hand
        
        Args:
            speed: Movement speed (0-100%, default 50%)
            force: Gripping force (0-100%, default 50%)
            
        Returns:
            True if command sent successfully
        """
        self.set_speeds([speed] * self.NUM_MOTORS)
        self.set_forces([force] * self.NUM_MOTORS)
        return self.set_angles([100] * self.NUM_MOTORS)
    
    def set_finger(self, finger_index: int, angle: float, speed: float = 50) -> bool:
        """
        Control a single finger
        
        Args:
            finger_index: Finger number (0-5)
            angle: Target angle (0-100%)
            speed: Movement speed (0-100%)
            
        Returns:
            True if command sent successfully
            
        Example:
            hand.set_finger(0, 75, speed=80)  # Move first finger to 75% closed at 80% speed
        """
        if not 0 <= finger_index < self.NUM_MOTORS:
            print(f"✗ Error: Finger index must be 0-{self.NUM_MOTORS-1}")
            return False
        
        # Use -1 for fingers we don't want to move
        angles = [-1] * self.NUM_MOTORS
        speeds = [-1] * self.NUM_MOTORS
        
        angles[finger_index] = angle
        speeds[finger_index] = speed
        
        self.set_speeds(speeds)
        return self.set_angles(angles)
    
    def execute_action_sequence(self, sequence_id: int) -> bool:
        """
        Execute a pre-programmed action sequence
        
        Args:
            sequence_id: Action sequence number
            
        Returns:
            True if command sent successfully
        """
        if not self._write_register(self.REGISTERS['ACTION_SEQ'], [sequence_id]):
            return False
        
        time.sleep(0.1)
        return self._write_register(self.REGISTERS['ACTION_RUN'], [1])
    
    def clear_errors(self) -> bool:
        """
        Clear all error codes
        
        Returns:
            True if command sent successfully
        """
        return self._write_register(self.REGISTERS['CLEAR_ERROR'], [1])
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


# Example usage
def main():
    """Demo program showing the intuitive 0-100% API"""
    
    with RH56RobotHand(port='/dev/ttyUSB0', baudrate=115200, hand_id=1) as hand:
        hand.set_speeds([100, 100, 100, 100, 100, 100])
        hand.set_angles([100, 100, 100, 100, 100, 100])

if __name__ == '__main__':
    main()  