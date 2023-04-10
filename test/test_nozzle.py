import pytest
from unittest.mock import MagicMock
from move.nozzle.nozzle_controller import NozzleController, ModbusException


class TestNozzleController:

    @pytest.fixture
    def mock_client(self):
        nozzle_client = MagicMock()
        nozzle_controller = NozzleController()
        nozzle_controller.client = nozzle_client
        return nozzle_controller
    
    
    @pytest.fixture
    def serial_client(self):
        nozzle_controller =  NozzleController(port="COM12", nozzles=20)
        nozzle_controller.connect()
        yield nozzle_controller
        nozzle_controller.disconnect()

    @pytest.mark.unit
    def test_connect(self, mock_client):
        mock_client.client.connect.return_value = True
        assert mock_client.connect() is True
        mock_client.client.connect.assert_called_once()
        mock_client.client.connect.reset_mock()
        mock_client.client.connect.return_value = False
        assert mock_client.connect() is False
        mock_client.client.connect.assert_called_with()

    @pytest.mark.unit
    def test_disconnect(self, mock_client):
        mock_client.disconnect()
        mock_client.client.close.assert_called_once()

    @pytest.mark.unit
    def test_read_coils(self, mock_client):
        mock_result = MagicMock()
        mock_result.isError.return_value = False
        mock_result.bits = [True, False, True]
        mock_client.client.read_coils.return_value = mock_result
        status = mock_client._read_coils(1, 3)
        assert status == [True, False, True]
        mock_result.isError.return_value = True
        mock_client.client.read_coils.return_value = mock_result
        with pytest.raises(ModbusException):
            mock_client._read_coils(1, 3)

    @pytest.mark.unit
    def test_write_coil(self, mock_client):
        mock_result = MagicMock()
        mock_result.isError.return_value = False
        mock_client.client.write_coil.return_value = mock_result
        result = mock_client._write_coil(1, True)
        assert result is True
        mock_result.isError.return_value = True
        mock_client.client.write_coil.return_value = mock_result
        with pytest.raises(ModbusException):
            mock_client._write_coil(1, True)

    @pytest.mark.unit
    def test_write_multiple_coils(self, mock_client):
        mock_result = MagicMock()
        mock_result.isError.return_value = False
        mock_client.client.write_coils.return_value = mock_result
        result = mock_client._write_multiple_coils(1, [True, False, True])
        assert result is True
        mock_result.isError.return_value = True
        mock_client.client.write_coils.return_value = mock_result
        with pytest.raises(ModbusException):
            mock_client._write_multiple_coils(1, [True, False, True])

    @pytest.mark.unit
    @pytest.mark.parametrize("id", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    def test_open_valve(self, id, mock_client):
        mock_result = MagicMock()
        mock_result.isError.return_value = False
        mock_client.client.write_coil.return_value = mock_result
        result = mock_client.open_valve(id)
        assert result is True
        mock_result.isError.return_value = True
        mock_client.client.write_coil.return_value = mock_result
        with pytest.raises(ModbusException):
            mock_client.open_valve(id)

    @pytest.mark.unit
    @pytest.mark.parametrize("id", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    def test_close_valve(self, id, mock_client):
        mock_result = MagicMock()
        mock_result.isError.return_value = False
        mock_client.client.write_coil.return_value = mock_result
        result = mock_client.close_valve(id)
        assert result is True
        mock_result.isError.return_value = True
        mock_client.client.write_coil.return_value = mock_result
        with pytest.raises(ModbusException):
            mock_client.close_valve(id)

    @pytest.mark.unit
    @pytest.mark.parametrize("id", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    def test_get_valve_status(self, id, mock_client):
        mock_result = MagicMock()
        mock_result.isError.return_value = False
        mock_result.bits = [True, False, True]
        mock_client.client.read_coils.return_value = mock_result
        result = mock_client.get_valve_status(id)
        assert result is True
        mock_result.isError.return_value = True
        mock_client.client.read_coils.return_value = mock_result
        with pytest.raises(ModbusException):
            mock_client.get_valve_status(id)

    @pytest.mark.unit
    def test_open_all_valves(self, mock_client):
        mock_result = MagicMock()
        mock_result.isError.return_value = False
        mock_client.client.write_coils.return_value = mock_result
        result = mock_client.open_all_valves()
        assert result is True
        mock_result.isError.return_value = True
        mock_client.client.write_coils.return_value = mock_result
        with pytest.raises(ModbusException):
            mock_client.open_all_valves()

    @pytest.mark.unit
    def test_close_all_valves(self, mock_client):
        mock_result = MagicMock()
        mock_result.isError.return_value = False
        mock_client.client.write_coils.return_value = mock_result
        result = mock_client.close_all_valves()
        assert result is True
        mock_result.isError.return_value = True
        mock_client.client.write_coils.return_value = mock_result
        with pytest.raises(ModbusException):
            mock_client.close_all_valves()

    @pytest.mark.unit
    def test_get_all_valve_status(self, mock_client):
        mock_result = MagicMock()
        mock_result.isError.return_value = False
        mock_result.bits = [True, False, True]
        mock_client.client.read_coils.return_value = mock_result
        result = mock_client.get_all_valve_status()
        assert result == [True, False, True]
        mock_result.isError.return_value = True
        mock_client.client.read_coils.return_value = mock_result
        with pytest.raises(ModbusException):
            mock_client.get_all_valve_status()

    @pytest.mark.integration
    @pytest.mark.parametrize("id", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    def test_single_valve(self, id, serial_client):
        assert serial_client.open_valve(id) is True
        assert serial_client.get_valve_status(id) is True
        all_status = serial_client.get_all_valve_status()
        assert all_status[id] is True
        assert serial_client.close_valve(id) is True
        assert serial_client.get_valve_status(id) is False
        all_status = serial_client.get_all_valve_status()
        assert all_status[id] is False

    @pytest.mark.integration
    def test_all_valves(self, serial_client):
        assert serial_client.open_all_valves() is True
        all_status = serial_client.get_all_valve_status()
        assert all_status == [True] * 20
        assert serial_client.close_all_valves() is True
        all_status = serial_client.get_all_valve_status()
        assert all_status == [False] * 20


    


if __name__ == "__main__":
    nozzle = NozzleController(port="COM12", nozzles=20)
    nozzle.connect()
    print(nozzle.open_all_valves())
    status = nozzle.get_all_valve_status()
    print(len(status))
    print(status)
    print(nozzle.close_all_valves())

    status = nozzle.get_all_valve_status()
    print(len(status))
    print(status)

    print(nozzle.open_valve(3))
    print(nozzle.get_valve_status(2))
    print(nozzle.get_valve_status(3))
    print(nozzle.get_valve_status(4))
    status = nozzle.get_all_valve_status()
    print(len(status))
    print(status)
    print(nozzle.close_valve(3))
    status = nozzle.get_all_valve_status()
    print(len(status))
    print(status)
