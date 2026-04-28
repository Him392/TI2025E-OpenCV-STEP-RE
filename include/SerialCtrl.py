import serial
from serial.tools import list_ports


def list_available_ports():
    return [port.device for port in list_ports.comports()]


class SerialComm:
    def __init__(self, port, baudrate, timeout=0.1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.buffer = bytearray()
        self._init_serial()

    def _init_serial(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"Serial initialized: {self.port}")
        except Exception as e:
            print(f"Serial initialization failed: {e}")
            self.ser = None

    def is_open(self):
        return self.ser is not None and self.ser.is_open

    def write(self, cmd):
        try:
            if self.is_open():
                self.ser.write(cmd)
        except Exception as e:
            print(f"Serial write error: {e}")

    def send_data(self, data):
        self.write(data)

    def read_all(self):
        if self.is_open() and self.ser.in_waiting > 0:
            data = self.ser.read(self.ser.in_waiting)
            self.buffer.extend(data)
            return data
        return b""

    def check_signal(self, signal):
        if signal in self.buffer:
            index = self.buffer.find(signal)
            self.buffer = self.buffer[index + len(signal):]
            return True
        return False

    def clear_buffer(self):
        self.buffer.clear()

    def close(self):
        if self.is_open():
            self.ser.close()
            print("Serial closed")
