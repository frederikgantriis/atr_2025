import os
import PyCmdMessenger
import time
import logging
from find_port import find_port

def get_logger(name):
    logging.basicConfig(level=logging.DEBUG)
    return logging.getLogger(name)

log = get_logger(__name__)

class RealMicrocontrollerService:
    """
    The actual microcontroller service for connecting to hardware.
    """

    def __init__(self):
        log.info("Initializing microcontroller service")
        found, comPort = find_port(os.getenv("DEVICE_ID"))

        if found:
            log.info(f"Connected to the device: {comPort}")
        else:
            log.error("No suitable device found.")
            exit()

        self._current_port_id = 1

        ESP32 = PyCmdMessenger.ArduinoBoard(comPort, baud_rate=115200, timeout=3)
        log.debug(f"Using board: {ESP32}")

        commands = [["kWatchdog", "s"],
                    ["kAcknowledge", "s"],
                    ["kError", "s"],
                    ["kStart", "?I?"],
                    ["kStop", ""], ]

        # Initialize the messenger
        self.comm = PyCmdMessenger.CmdMessenger(ESP32, commands)
        log.info("Messenger initialized")
        # Wait for arduino to come up
        msg = self.comm.receive()
        log.info(f"Initial communication: {msg}")

    def stopPumps(self):
        """Function for stopping all pumps."""
        log.info("Sending stop command to all pumps")
        self.comm.send("kStop")
        try:
            msg = self.comm.receive()
            log.info(f"Stop pumps response: {msg[1]}")
            return msg[1]
        except EOFError as e:
            log.warning(f"No or incomplete response to stop command: {e}")
            return "No response"



    def start_pumps(self, stateA: bool = False, speedA: int = 0, dirA: bool = True):
        """
        Set the state of pumps A, B, C and the LEDs
        """
        log.info(f"Setting state: A={stateA},{speedA},{dirA} ")
        try:
            self.comm.send("kStart", stateA, speedA, dirA)
            msg = self.comm.receive()
            log.info(msg)
            log.info(f"State set response: {msg[1]}")
            return True
        except Exception as e:
            log.error(f"Error setting state: {e}")
            return False

    

    def close(self):
        """
        Closes the serial connection.
        """
        try:
            self.ser.close()
            log.info("Serial connection closed")
        except Exception as e:
            log.error(f"Error closing connection: {str(e)}")



def main():
    """Main function for testing the service directly."""
    tile = RealMicrocontrollerService()
    tile.start_pumps(stateA = True, speedA = 4096, dirA = True)
    
    time.sleep(5)

    tile.stopPumps()


if __name__ == "__main__":
    main()