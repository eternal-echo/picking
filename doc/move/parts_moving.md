# NozzleMoving

接收`PartsDetecting`模块检测到的零件坐标。统计出同一零件在指定区域框内的坐标和类型数据，当零件移动出指定区域后，开启定时器延时启动电磁阀。

## 依赖

通过`NozzleController`对象来控制多路电磁阀，从而开启或关闭对应的喷嘴。需要为`NozzleController`对象加锁，避免多线程同时调用。

```python
from move.nozzle.nozzle_controller import NozzleController
```

## 接口

```python
    def move(self, part_id: int, pos: tuple[float, float]) -> bool:
        """
        Move the part to the specified position by controlling the nozzle valves
        :param part_id: The part ID
        :param pos: The detected position of the part
        :return: True if the part was moved successfully
        """
        pass
```



## 成员

- `parts_settings`: `dict`类型，键为`int`类型的零件类型，值为`PartSetting`类的对象。`PartSetting`类包含以下属性：
  - `nozzle_id`: 喷嘴号，`int`类型。
  - `move_time`: 从相机到喷嘴需要的时间，以毫秒为单位，`float`类型。
- `nozzle_settings`: 长度为20的`list`类型，数组元素是`NozzleSetting`类的对象，包含以下属性：
  - `nozzle_id`: 喷嘴编号，`int`类型。
  - `move_time`: 零件从相机到喷嘴需要的时间，以毫秒为单位，`float`类型。
- `time_range`设置了喷嘴吹送走一个零件需要打开的时长，单位为ms。

### NozzleInfo

`NozzleInfo`类包含以下属性：

- `nozzle_number`：喷嘴编号，`int`类型。
- `move_time`：零件从相机到喷嘴需要的时间，以毫秒为单位，`float`类型。
- `is_busy`：该喷嘴是否正在使用，`bool`类型。

### MoveRequest

零件检测模块检测到了新的零件后，会调用`NozzleMoving`的`def move(self, name, pos) -> bool`接口。`move`接口会创建一个`MoveRequest`类，添加到`NozzleMoving`类的请求链表`request_list`中。

`MoveRequest`类中包含了零件的类型`name`和坐标`pos`、第一次出现的时间`time1`和最后一次出现的时间`time2`、协程。

`MoveRequest`类包含以下属性：

- `part_info`：零件的坐标和类型
  - `pos`：零件的坐标，`(float, float)`元组类型。
  - `id`：零件的类型，`int`类型。
- `start_time`：零件第一次出现的时间，以毫秒为单位，`float`类型。
- `end_time`：零件最后一次出现的时间，以毫秒为单位，`float`类型。
- `task`：协程对象。

`MoveRequest`的协程对象会根据`part_info.id`，查找`parts_info`中对应的喷嘴号，然后根据喷嘴号获取`nozzle_infos`中零件运动到喷嘴所需时间。然后在延时`move_time - time_range/2`毫秒后启动该喷嘴，并且在`time_range`毫秒后关闭该喷嘴。

### MoveRequests

包含`MoveRequest`的deque，

使用双端队列（deque）和二分查找来解决。由于deque是逆序排列的，所以可以使用二分查找在O(log n)的时间复杂度内找到第一个小于等于当前零件横坐标的位置，然后将这个零件信息更新到deque中。如果找不到这样的位置，那么这个数就应该插入到deque的末尾。具体步骤如下：

1. 初始化一个空的deque。

2. 依次读入所有的`MoveRequest`，对于每个数执行以下操作：

   a. 在deque中使用二分查找找到第一个小于等于当前零件横坐标的位置。

   b. 如果这个位置后面还有元素，则用当前数更新这个位置的后一个元素。

   c. 否则，将当前数插入到deque的末尾。