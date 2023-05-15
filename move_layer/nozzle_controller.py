from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusException
from pymodbus.payload import BinaryPayloadDecoder, BinaryPayloadBuilder
from pymodbus.constants import Endian
import logging

_logger = logging.getLogger('nozzle')
_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
_logger.addHandler(console_handler)

class NozzleController:
    """**NozzleController** is a class that controls the nozzle valves on the
    Modbus RTU slave device.
    """
    def __init__(self, port="COM2", baudrate=9600, slave_id=1, nozzles=20):
        """Initialize the NozzleController class
        :param port: The serial port to connect to
        :param baudrate: The baudrate to use
        :param slave_id: The slave ID to use
        :param nozzles: The number of nozzles to control
        """
        # Initialize a Modbus client
        self.client = ModbusSerialClient(
            port=port,
            baudrate=baudrate,
            parity="N",
            stopbits=1,
            bytesize=8,
        )
        self.slave_id = slave_id
        self.nozzles = nozzles


    def connect(self):
        """Connect to the Modbus RTU slave device
        :return: True if the connection was successful
        """
        _logger.info("Connecting to Modbus RTU server")
        if not self.client.connect():
            return False
        else:
            return True

    def disconnect(self):
        # Disconnect from the slave device
        _logger.info("Disconnecting from Modbus RTU server")
        self.client.close()

    def _read_coils(self, address, count):
        '''Read the value of the specified coil
        :param address: The coil address
        :param count: The number of coils to read
        :return: A list of values
        '''
        # Read the coils from the remote device
        _logger.info("Reading %d coils from address %d", count, address)
        result = self.client.read_coils(address, count, self.slave_id)
        # If the read was successful
        if not result.isError():
            status = result.bits[0:count]
            _logger.debug("Read %d coils: %s", len(status), status)
            return status
        else:
            raise ModbusException("Failed to read coils: " + str(result))

    def _write_coil(self, address, value):
        """Write a coil (boolean) value to a modbus device
        :param address: The coil address
        :param value: The value to write
        :return: True if the write was successful
        """
        # Write the coil value to the modbus device
        _logger.info("Writing coil at address %d", address)
        result = self.client.write_coil(address, value, self.slave_id)
        # Check the result of the write
        if not result.isError():
            return True
        else:
            # If there was an error raise an exception
            raise ModbusException("Failed to write coil: " + str(result))

    def _write_multiple_coils(self, address, values):
        """Write multiple coils (boolean) values to a modbus device
        :param address: The coil address
        :param values: The values to write
        :return: True if the write was successful
        """
        _logger.info("Writing %d coils %s at address %d", len(values), values, address)
        result = self.client.write_coils(address, values, self.slave_id, skip_encode=True)
        # Check if the write was successful
        if not result.isError():
            return True
        else:
            raise ModbusException("Failed to write multiple coils")

    def close_valve(self, valve_id):
        """Closes the specified valve.
        :param valve_id: The ID of the valve to close.
        :return: True if the valve was closed successfully.
        """
        # Set coil to false to close valve
        return self._write_coil(valve_id, False)

    def close_all_valves(self):
        """Closes all valves.
        :return: True if all valves were closed successfully.
        """
        # Turn all valves off
        return self._write_multiple_coils(0, [False] * self.nozzles)

    def open_valve(self, valve_id) -> bool:
        """Opens the specified valve.
        :param valve_id: The ID of the valve to open.
        :return: True if the valve was opened successfully.
        """
        return self._write_coil(valve_id, True)

    def open_all_valves(self):
        """Opens all valves.
        :return: True if all valves were opened successfully.
        """
        return self._write_multiple_coils(0, [True] * self.nozzles)

    def get_valve_status(self, valve_id):
        """Gets the status of the specified valve.
        :valve_id: The ID of the valve to get the status of.
        :return: A boolean value representing the status of the valve.
        """
        return self._read_coils(valve_id, 1)[0]

    def get_all_valve_status(self):
        """Gets the status of all valves.
        :return: A list of boolean values representing the status of all valves.
        """
        return self._read_coils(0, self.nozzles)