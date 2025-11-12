import serial.tools.list_ports
import platform

def find_port(device_id=None):
    """
    Scans available serial ports and returns (found, port_name).
    If device_id is provided, matches by serial number or description.
    On Windows, returns a hardcoded COM port (e.g., 'COM3').
    """
    # if platform.system() == "Windows":
    #     # Hardcode the COM port for Windows
    #     return True, "COM9"
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if device_id:
            if device_id in (port.serial_number or '') or device_id in (port.description or ''):
                return True, port.device
        else:
            # Return the first available port if no device_id specified
            return True, port.device
    return False, None 