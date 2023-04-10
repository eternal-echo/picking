# 测试用例文档

## move

### NozzleController

#### 单元测试

在工程的根目录下输入下面的命令行，运行所有单元测试。单元测试通过`Mock`打桩替换了`pymodbus`中serial相关的底层接口，对`NozzleController`类的所有接口进行了验证。

```bash
pytest -m unit
```

#### 集成测试

修改`test\test_nozzle.py`文件中TestNozzleController测试用例类的夹具中选择的串口设备，需要和设备管理器中显示的串口号保持一致。

```python
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
```

运行下面的命令行，运行所有集成测试。集成测试会与实际串口连接，需要准备好Modbus客户端设备，`slave_id`、波特率、电磁阀通道数都需要与实际设备保持一致。其中`slave_id`没设置对将无法正常通信。

```bash
pytest -m integration
```



当连接好多路PLC交流放大板后，开始集成测试，会观察到放大板上指示灯从0到19开始依次点亮，最后再全体亮灭一次。





