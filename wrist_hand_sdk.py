import serial
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class WristState:
    pitch_angle: float
    yaw_angle: float
    current1: int
    current2: int
    temperature1: int
    temperature2: int
    error_code1: int
    error_code2: int


@dataclass
class HandState:
    angles: List[float]
    forces: List[float]
    temperatures: List[int]
    error_codes: List[int]
    status_codes: List[int]


@dataclass
class CompleteHandState:
    hand: HandState
    wrist: Optional[WristState]


class RH56RobotHandWithWrist:
    
    HAND_REGISTERS = {
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
    
    WRIST_REGISTERS = {
        'PITCH_ANGLE': 1020,
        'YAW_ANGLE': 1022,
        'CURRENT1': 1024,
        'CURRENT2': 1026,
        'ERROR1': 1028,
        'ERROR2': 1030,
        'TEMP1': 1032,
        'TEMP2': 1033,
        'YAW_ANGLE_SET': 1038,
        'PITCH_ANGLE_SET': 1040,
        'PROFILE_TIME_MS': 1042
    }
    
    CMD_READ = 0x11
    CMD_WRITE = 0x12
    CMD_READ_WRIST = 0x30
    CMD_WRITE_WRIST = 0x31
    FRAME_HEADER = [0xEB, 0x90]
    NUM_MOTORS = 6
    
    _HW_SCALE = 1000
    _USER_SCALE = 100
    _ANGLE_SCALE = 100
    
    PITCH_MIN = -22.66
    PITCH_MAX = 22.12
    YAW_MIN = -25.50
    YAW_MAX = 25.50
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200, hand_id: int = 1):
        self.port = port
        self.baudrate = baudrate
        self.hand_id = hand_id
        self.serial: Optional[serial.Serial] = None
    
    @staticmethod
    def _to_hardware_value(user_value: float) -> int:
        if user_value == -1:
            return -1
        return int(user_value * RH56RobotHandWithWrist._HW_SCALE / RH56RobotHandWithWrist._USER_SCALE)
    
    @staticmethod
    def _from_hardware_value(hw_value: int) -> float:
        return round(hw_value * RH56RobotHandWithWrist._USER_SCALE / RH56RobotHandWithWrist._HW_SCALE, 1)
    
    @staticmethod
    def _angle_to_register(angle_degrees: float) -> int:
        reg_value = int(angle_degrees * RH56RobotHandWithWrist._ANGLE_SCALE)
        if reg_value < 0:
            reg_value = reg_value & 0xFFFF
        return reg_value
    
    @staticmethod
    def _register_to_angle(register_value: int) -> float:
        return round(register_value / RH56RobotHandWithWrist._ANGLE_SCALE, 2)
    
    @staticmethod
    def _cubic_transform(x: float, coeffs: List[float]) -> float:
        a, b, c, d = coeffs
        return a * x**3 + b * x**2 + c * x + d
    
    @staticmethod
    def joint_to_motor_thumb(thumb_a: float) -> float:
        thumb_b = RH56RobotHandWithWrist._cubic_transform(
            thumb_a, 
            RH56RobotHandWithWrist.THUMB_A_TO_B_COEFFS
        )
        thumb_c = RH56RobotHandWithWrist._cubic_transform(
            thumb_b,
            RH56RobotHandWithWrist.THUMB_B_TO_C_COEFFS
        )
        return thumb_c
    
    @staticmethod
    def joint_to_motor_finger(finger_e: float) -> float:
        return RH56RobotHandWithWrist._cubic_transform(
            finger_e,
            RH56RobotHandWithWrist.FINGER_E_TO_F_COEFFS
        )
    
    @staticmethod
    def joint_angles_to_motor_angles(joint_angles: List[float]) -> List[float]:
        if len(joint_angles) != RH56RobotHandWithWrist.NUM_MOTORS:
            raise ValueError(f"Expected {RH56RobotHandWithWrist.NUM_MOTORS} joint angles")
        
        motor_angles = []
        
        motor_angles.append(RH56RobotHandWithWrist.joint_to_motor_thumb(joint_angles[0]))
        
        for i in range(1, RH56RobotHandWithWrist.NUM_MOTORS):
            motor_angles.append(RH56RobotHandWithWrist.joint_to_motor_finger(joint_angles[i]))
        
        return motor_angles
    
    def connect(self) -> bool:
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            print(f"Connected to robot hand on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Disconnected from robot hand")
    
    def _write_register(self, address: int, data: List[int]) -> bool:
        if not self.serial or not self.serial.is_open:
            print("Error: Serial port not open")
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
        if not self.serial or not self.serial.is_open:
            print("Error: Serial port not open")
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
    
    def _read_wrist_register(self, address: int, num_bytes: int) -> Optional[List[int]]:
        if not self.serial or not self.serial.is_open:
            print("Error: Serial port not open")
            return None
        
        frame = self.FRAME_HEADER.copy()
        frame.append(self.hand_id)
        frame.append(0x04)
        frame.append(self.CMD_READ_WRIST)
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
    
    def _write_wrist_register(self, address: int, data: List[int]) -> bool:
        if not self.serial or not self.serial.is_open:
            print("Error: Serial port not open")
            return False
        
        frame = self.FRAME_HEADER.copy()
        frame.append(self.hand_id)
        frame.append(len(data) + 3)
        frame.append(self.CMD_WRITE_WRIST)
        frame.append(address & 0xFF)
        frame.append((address >> 8) & 0xFF)
        frame.extend(data)
        
        checksum = sum(frame[2:]) & 0xFF
        frame.append(checksum)
        
        self.serial.write(bytes(frame))
        time.sleep(0.01)
        self.serial.read_all()
        
        return True
    
    def set_angles(self, angles: List[float]) -> bool:
        if len(angles) != self.NUM_MOTORS:
            print(f"Error: Expected {self.NUM_MOTORS} angles, got {len(angles)}")
            return False
        
        data = []
        for angle in angles:
            hw_value = self._to_hardware_value(angle)
            data.append(hw_value & 0xFF)
            data.append((hw_value >> 8) & 0xFF)
        
        return self._write_register(self.HAND_REGISTERS['ANGLE_SET'], data)
    
    def set_joint_angles(self, joint_angles: List[float]) -> bool:
        if len(joint_angles) != self.NUM_MOTORS:
            print(f"Error: Expected {self.NUM_MOTORS} joint angles, got {len(joint_angles)}")
            return False
        
        motor_angles = []
        for i, joint_angle in enumerate(joint_angles):
            if joint_angle == -1:
                motor_angles.append(-1)
            elif i == 0:
                motor_angles.append(self.joint_to_motor_thumb(joint_angle))
            else:
                motor_angles.append(self.joint_to_motor_finger(joint_angle))
        
        return self.set_angles(motor_angles)
    
    def set_speeds(self, speeds: List[float]) -> bool:
        if len(speeds) != self.NUM_MOTORS:
            print(f"Error: Expected {self.NUM_MOTORS} speeds, got {len(speeds)}")
            return False
        
        data = []
        for speed in speeds:
            hw_value = self._to_hardware_value(speed)
            data.append(hw_value & 0xFF)
            data.append((hw_value >> 8) & 0xFF)
        
        return self._write_register(self.HAND_REGISTERS['SPEED_SET'], data)
    
    def set_forces(self, forces: List[float]) -> bool:
        if len(forces) != self.NUM_MOTORS:
            print(f"Error: Expected {self.NUM_MOTORS} forces, got {len(forces)}")
            return False
        
        data = []
        for force in forces:
            hw_value = self._to_hardware_value(force)
            data.append(force & 0xFF)
            data.append((hw_value >> 8) & 0xFF)
        
        return self._write_register(self.HAND_REGISTERS['FORCE_SET'], data)
    
    def get_angles(self) -> Optional[List[float]]:
        data = self._read_register(self.HAND_REGISTERS['ANGLE_ACTUAL'], 12)
        if not data or len(data) < 12:
            print("Failed to read angles")
            return None
        
        angles = []
        for i in range(self.NUM_MOTORS):
            hw_value = (data[2*i] & 0xFF) | (data[2*i + 1] << 8)
            angles.append(self._from_hardware_value(hw_value))
        
        return angles
    
    def get_forces(self) -> Optional[List[float]]:
        data = self._read_register(self.HAND_REGISTERS['FORCE_ACTUAL'], 12)
        if not data or len(data) < 12:
            print("Failed to read forces")
            return None
        
        forces = []
        for i in range(self.NUM_MOTORS):
            hw_value = (data[2*i] & 0xFF) | (data[2*i + 1] << 8)
            forces.append(self._from_hardware_value(hw_value))
        
        return forces
    
    def get_temperatures(self) -> Optional[List[int]]:
        data = self._read_register(self.HAND_REGISTERS['TEMPERATURE'], 6)
        if not data or len(data) < 6:
            print("Failed to read temperatures")
            return None
        
        return data
    
    def get_hand_state(self) -> Optional[HandState]:
        angles = self.get_angles()
        forces = self.get_forces()
        temps = self.get_temperatures()
        
        errors = self._read_register(self.HAND_REGISTERS['ERROR_CODE'], 6)
        status = self._read_register(self.HAND_REGISTERS['STATUS_CODE'], 6)
        
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
        self.set_speeds([speed] * self.NUM_MOTORS)
        return self.set_angles([0] * self.NUM_MOTORS)
    
    def close_hand(self, speed: float = 50, force: float = 50) -> bool:
        self.set_speeds([speed] * self.NUM_MOTORS)
        self.set_forces([force] * self.NUM_MOTORS)
        return self.set_angles([100] * self.NUM_MOTORS)
    
    def set_finger(self, finger_index: int, angle: float, speed: float = 50) -> bool:
        if not 0 <= finger_index < self.NUM_MOTORS:
            print(f"Error: Finger index must be 0-{self.NUM_MOTORS-1}")
            return False
        
        angles = [-1] * self.NUM_MOTORS
        speeds = [-1] * self.NUM_MOTORS
        
        angles[finger_index] = angle
        speeds[finger_index] = speed
        
        self.set_speeds(speeds)
        return self.set_angles(angles)
    
    def set_finger_joint(self, finger_index: int, joint_angle: float, speed: float = 50) -> bool:
        if not 0 <= finger_index < self.NUM_MOTORS:
            print(f"Error: Finger index must be 0-{self.NUM_MOTORS-1}")
            return False
        
        if finger_index == 0:
            motor_angle = self.joint_to_motor_thumb(joint_angle)
        else:
            motor_angle = self.joint_to_motor_finger(joint_angle)
        
        return self.set_finger(finger_index, motor_angle, speed)
    
    def execute_action_sequence(self, sequence_id: int) -> bool:
        if not self._write_register(self.HAND_REGISTERS['ACTION_SEQ'], [sequence_id]):
            return False
        
        time.sleep(0.1)
        return self._write_register(self.HAND_REGISTERS['ACTION_RUN'], [1])
    
    def clear_errors(self) -> bool:
        return self._write_register(self.HAND_REGISTERS['CLEAR_ERROR'], [1])
    
    def set_wrist_angles(self, pitch: float, yaw: float, movement_time_ms: int = 1000) -> bool:
        if not (self.PITCH_MIN <= pitch <= self.PITCH_MAX):
            print(f"Error: Pitch angle {pitch}° out of range [{self.PITCH_MIN}, {self.PITCH_MAX}]")
            return False
        
        if not (self.YAW_MIN <= yaw <= self.YAW_MAX):
            print(f"Error: Yaw angle {yaw}° out of range [{self.YAW_MIN}, {self.YAW_MAX}]")
            return False
        
        yaw_reg = self._angle_to_register(yaw)
        pitch_reg = self._angle_to_register(pitch)
        
        time_data = [
            movement_time_ms & 0xFF,
            (movement_time_ms >> 8) & 0xFF
        ]
        if not self._write_wrist_register(self.WRIST_REGISTERS['PROFILE_TIME_MS'], time_data):
            return False
        
        pitch_data = [
            pitch_reg & 0xFF,
            (pitch_reg >> 8) & 0xFF
        ]
        if not self._write_wrist_register(self.WRIST_REGISTERS['PITCH_ANGLE_SET'], pitch_data):
            return False
        
        yaw_data = [
            yaw_reg & 0xFF,
            (yaw_reg >> 8) & 0xFF
        ]
        return self._write_wrist_register(self.WRIST_REGISTERS['YAW_ANGLE_SET'], yaw_data)
    
    def set_wrist_pitch(self, pitch: float, movement_time_ms: int = 1000) -> bool:
        if not (self.PITCH_MIN <= pitch <= self.PITCH_MAX):
            print(f"Error: Pitch angle {pitch}° out of range")
            return False
        
        time_data = [
            movement_time_ms & 0xFF,
            (movement_time_ms >> 8) & 0xFF
        ]
        self._write_wrist_register(self.WRIST_REGISTERS['PROFILE_TIME_MS'], time_data)
        
        pitch_reg = self._angle_to_register(pitch)
        data = [
            pitch_reg & 0xFF,
            (pitch_reg >> 8) & 0xFF
        ]
        
        return self._write_wrist_register(self.WRIST_REGISTERS['PITCH_ANGLE_SET'], data)
    
    def set_wrist_yaw(self, yaw: float, movement_time_ms: int = 1000) -> bool:
        if not (self.YAW_MIN <= yaw <= self.YAW_MAX):
            print(f"Error: Yaw angle {yaw}° out of range")
            return False
        
        time_data = [
            movement_time_ms & 0xFF,
            (movement_time_ms >> 8) & 0xFF
        ]
        self._write_wrist_register(self.WRIST_REGISTERS['PROFILE_TIME_MS'], time_data)
        
        yaw_reg = self._angle_to_register(yaw)
        data = [
            yaw_reg & 0xFF,
            (yaw_reg >> 8) & 0xFF
        ]
        
        return self._write_wrist_register(self.WRIST_REGISTERS['YAW_ANGLE_SET'], data)

    
    
    def get_wrist_angles(self) -> Optional[Tuple[float, float]]:
        pitch_data = self._read_wrist_register(self.WRIST_REGISTERS['PITCH_ANGLE'], 2)
        if not pitch_data or len(pitch_data) < 2:
            print("Failed to read wrist pitch angle")
            return None
        
        yaw_data = self._read_wrist_register(self.WRIST_REGISTERS['YAW_ANGLE'], 2)
        if not yaw_data or len(yaw_data) < 2:
            print("Failed to read wrist yaw angle")
            return None
        
        pitch_reg = (pitch_data[0] & 0xFF) | (pitch_data[1] << 8)
        if pitch_reg & 0x8000:
            pitch_reg -= 0x10000
        
        yaw_reg = (yaw_data[0] & 0xFF) | (yaw_data[1] << 8)
        if yaw_reg & 0x8000:
            yaw_reg -= 0x10000
        
        pitch = self._register_to_angle(pitch_reg)
        yaw = self._register_to_angle(yaw_reg)
        
        return (pitch, yaw)
    
    def get_wrist_state(self) -> Optional[WristState]:
        angles = self.get_wrist_angles()
        if not angles:
            return None
        pitch, yaw = angles
        
        current_data = self._read_wrist_register(self.WRIST_REGISTERS['CURRENT1'], 4)
        if not current_data or len(current_data) < 4:
            print("Failed to read wrist currents")
            return None
        
        current1 = (current_data[0] & 0xFF) | (current_data[1] << 8)
        current2 = (current_data[2] & 0xFF) | (current_data[3] << 8)
        
        error1_data = self._read_wrist_register(self.WRIST_REGISTERS['ERROR1'], 1)
        error2_data = self._read_wrist_register(self.WRIST_REGISTERS['ERROR2'], 1)
        
        if not error1_data or not error2_data:
            print("Failed to read wrist error codes")
            return None
        
        error1 = error1_data[0]
        error2 = error2_data[0]
        
        temp_data = self._read_wrist_register(self.WRIST_REGISTERS['TEMP1'], 2)
        if not temp_data or len(temp_data) < 2:
            print("Failed to read wrist temperatures")
            return None
        
        temp1 = temp_data[0]
        temp2 = temp_data[1]
        
        return WristState(
            pitch_angle=pitch,
            yaw_angle=yaw,
            current1=current1,
            current2=current2,
            temperature1=temp1,
            temperature2=temp2,
            error_code1=error1,
            error_code2=error2
        )
    
    def center_wrist(self, movement_time_ms: int = 1000) -> bool:
        return self.set_wrist_angles(pitch=0.0, yaw=0.0, movement_time_ms=movement_time_ms)
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def main():
    hand = RH56RobotHandWithWrist(port='/dev/ttyUSB0', baudrate=115200, hand_id=1)
    
    if not hand.connect():
        return
    
    try:
        # hand.set_speeds([50] * 6)
        # hand.set_angles([100, 100, 100, 100, 100, 70])
        # time.sleep(1)
        
        # hand.set_finger(0, 80, speed=50)
        hand.set_wrist_angles(pitch=0.0, yaw=0, movement_time_ms=1000)

        time.sleep(1.5)
        
        hand.close_hand(speed=50, force=50)
        # hand.set_wrist_angles(pitch=0.0, yaw=0.0, movement_time_ms=1000)
        time.sleep(1.5)
        
    finally:
        hand.disconnect()


if __name__ == '__main__':
    main()
