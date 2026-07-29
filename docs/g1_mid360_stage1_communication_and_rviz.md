# Unitree G1 Mid-360 第一阶段：通信、原始点云与 RViz 验证

> 日期：2026-07-29  
> 分支：`test/lidar-blindzone-heightmap`  
> 阶段目标：不接入 FAST-LIO，先确认 G1、Livox Mid-360、ROS 2、DDS、原始点云与雷达内部 IMU 的数据链路能够工作。

## 1. 本阶段结论

第一阶段已经完成以下链路验证：

```text
Livox Mid-360 (192.168.123.120)
        ↓ G1 内部 eth0
G1 机载计算机 (192.168.123.164)
        ↓ livox_ros_driver2
/livox/lidar  sensor_msgs/msg/PointCloud2
/livox/imu    sensor_msgs/msg/Imu
        ↓ CycloneDDS / Wi-Fi
外部 Ubuntu 22.04 + ROS 2 Humble 电脑
        ↓
RViz2 显示原始点云
```

实测结果：

- Mid-360 地址 `192.168.123.120` 可从 G1 的 `eth0` 正常 ping 通。
- G1 上官方 `livox_ros_driver2` 可以成功连接雷达。
- `/livox/lidar` 有真实、非空的 `PointCloud2` 数据。
- 单帧点云约 `20064` 个点，数据量约 `521664` bytes。
- 点云字段完整：`x`、`y`、`z`、`intensity`、`tag`、`line`、`timestamp`。
- `/livox/imu` 约 `200 Hz`。
- G1 本机长期测得点云频率逐渐收敛到约 `7.6 Hz`，目标配置为 `10 Hz`。
- 外部电脑经 Wi-Fi 接收完整点云时只有约 `0.5–0.6 Hz`，可以调试和看 RViz，但不适合最终实时 FAST-LIO。
- RViz 中能够看到点云，但点云上下颠倒，符合 G1 上 Mid-360 倒置安装且当前使用官方驱动、尚未做完整倒装外参修正的现象。

本阶段还没有运行 FAST-LIO，也没有生成里程计、轨迹或累积地图。

---

## 2. 本阶段到底修改了什么

### 2.1 没有修改的内容

本阶段没有修改以下内容：

- 没有修改 `~/test_livox/src/livox_ros_driver2` 中的源码。
- 没有修改该驱动原有的 `MID360_config.json`。
- 没有覆盖或删除现有的 `~/test_livox` 工作空间。
- 没有修改 G1 的 `~/.bashrc`。
- 没有修改已有的 `~/cyclonedds_eth0.xml`，只读取和使用了它。
- 没有创建、停止或重启 systemd 服务。
- 没有执行 `systemctl restart/stop`、`killall` 或影响其他用户节点的全局操作。
- 没有向 `/lowcmd`、`/user_lowcmd`、`/api/sport/request` 等控制话题发布命令。
- 没有修改机器人策略部署目录。
- 没有修改机器人固件或雷达固件。

### 2.2 产生的临时文件

G1 上创建过以下临时文件：

```text
/tmp/g1_topics.txt
/tmp/MID360_g1_qiaomu.json
/tmp/cyclonedds_livox_wifi.xml
/tmp/check_livox_cloud.py
```

外部电脑上创建过：

```text
/tmp/cyclonedds_g1_wifi.xml
```

这些文件用于本次测试，不属于驱动源码。`/tmp` 通常会在重启后清理，也可以手动删除：

```bash
rm -f \
  /tmp/g1_topics.txt \
  /tmp/MID360_g1_qiaomu.json \
  /tmp/cyclonedds_livox_wifi.xml \
  /tmp/check_livox_cloud.py
```

外部电脑：

```bash
rm -f /tmp/cyclonedds_g1_wifi.xml
```

### 2.3 当前终端内的临时环境变量

本阶段设置过：

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
export ROS_DOMAIN_ID=0
export ROS2CLI_DISABLE_DAEMON=1
export CYCLONEDDS_URI=...
```

这些 `export` 只影响当前 shell。关闭 SSH 或终端后即失效，没有写入 `~/.bashrc`。

### 2.4 唯一明确的系统级改动

外部 Ubuntu 22.04 电脑为了显示点云，安装了 RViz2：

```bash
sudo apt update
sudo apt install -y ros-humble-rviz2
```

这是电脑端的持久软件包安装，不是 G1 端修改。

---

## 3. 实测环境

### 3.1 G1 机载计算机

```text
Hostname: ubuntu
OS: Ubuntu 20.04.6 LTS (Focal Fossa)
Architecture: aarch64
ROS 2: Foxy
RMW: rmw_cyclonedds_cpp
```

网络：

```text
eth0   192.168.123.164/24   G1 内部设备网络，连接 Mid-360
wlan0  172.16.10.18/24      SSH 和跨机器 ROS 2 调试
```

### 3.2 外部电脑

```text
OS: Ubuntu 22.04
ROS 2: Humble
Wi-Fi interface: wlp7s0
Wi-Fi IP during test: 172.16.10.136
```

### 3.3 雷达

```text
Model: Livox Mid-360
IP: 192.168.123.120
Installation: upside down on Unitree G1
```

---

## 4. 安全和多人共用注意事项

本阶段只允许读取和订阅数据。

安全命令：

```bash
ros2 topic list
ros2 topic info <topic> -v
ros2 topic hz <topic>
ros2 topic echo <topic>
ros2 node list
ping <ip>
ip route
ip neigh
ps -ef
```

多人共用 G1 时不要执行：

```bash
ros2 daemon stop
pkill -f _ros2_daemon
killall ...
systemctl restart ...
systemctl stop ...
ros2 topic pub /lowcmd ...
ros2 topic pub /user_lowcmd ...
ros2 topic pub /api/sport/request ...
```

另外，同一时间不要同时启动两套 Livox 主驱动。虽然 Livox 节点不控制机器人关节，但两套 SDK/驱动可能争用雷达命令和数据接口。

---

## 5. 步骤一：SSH 连接并确认系统与网络

SSH 已经通过 Wi-Fi 建立：

```text
外部电脑 172.16.10.136
    ↓ SSH
G1 wlan0 172.16.10.18
```

在 G1 SSH 终端执行：

```bash
echo "===== SSH CONNECTION ====="
echo "$SSH_CONNECTION"

echo
echo "===== SYSTEM ====="
hostname
cat /etc/os-release | grep -E 'NAME=|VERSION='
uname -m

echo
echo "===== NETWORK ====="
ip -br -4 addr
ip route

echo
echo "===== ROS ====="
ls -1 /opt/ros 2>/dev/null
command -v ros2 || true

echo
echo "===== DDS ENV ====="
env | grep -E \
'ROS_DISTRO|ROS_DOMAIN_ID|ROS_LOCALHOST_ONLY|RMW_IMPLEMENTATION|CYCLONEDDS_URI|FASTRTPS|FASTDDS' \
|| true
```

实测：

```text
SSH_CONNECTION=172.16.10.136 ... 172.16.10.18 22
Ubuntu 20.04.6 LTS
aarch64
/opt/ros/foxy
/opt/ros/noetic
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ROS_LOCALHOST_ONLY=0
```

确认 SSH 走 `wlan0`：

```bash
CLIENT_IP=$(echo "$SSH_CONNECTION" | awk '{print $1}')
echo "SSH client IP: $CLIENT_IP"
ip route get "$CLIENT_IP"
```

实测结果类似：

```text
172.16.10.136 dev wlan0 src 172.16.10.18
```

---

## 6. 步骤二：加载 G1 的 ROS 2 与 Unitree 消息环境

不需要进入策略 deploy 目录，也不需要激活 uv Python。

下面这个命令是错误的，不要使用：

```bash
source /home/unitree/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/bin/activate
```

该目录是 uv 管理的 Python 解释器目录，不是虚拟环境，所以没有 `bin/activate`。

正确操作：

```bash
source /opt/ros/foxy/setup.bash

if [ -f "$HOME/cyclonedds_ws/install/setup.bash" ]; then
    source "$HOME/cyclonedds_ws/install/setup.bash"
elif [ -f "$HOME/install/setup.bash" ]; then
    source "$HOME/install/setup.bash"
fi

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
export ROS2CLI_DISABLE_DAEMON=1
export CYCLONEDDS_URI="file://$HOME/cyclonedds_eth0.xml"
```

`ROS2CLI_DISABLE_DAEMON=1` 用来避免操作共享的 ROS 2 CLI daemon，不会停止其他人的节点。

确认：

```bash
echo "ROS_DISTRO=$ROS_DISTRO"
echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "CYCLONEDDS_URI=$CYCLONEDDS_URI"
```

---

## 7. 步骤三：验证 Unitree DDS 通信

列出全部话题：

```bash
timeout 15 ros2 topic list -t | tee /tmp/g1_topics.txt
```

筛选关键话题：

```bash
grep -Ei \
'utlidar|lidar|livox|cloud|lowstate|sportmodestate|wireless' \
/tmp/g1_topics.txt || true
```

实测能够看到：

```text
/lf/lowstate
/lowstate
/lf/sportmodestate
/sportmodestate
/wirelesscontroller
/unitree/slam_mapping/points
/unitree/slam_relocation/points
/utlidar/range_info
```

结论：

- G1 的 ROS 2 Foxy、Unitree 消息包和 CycloneDDS 已正常工作。
- 机器人状态话题可以被发现。
- 当时不存在 `/utlidar/cloud`。
- `/utlidar/range_info` 只有订阅者，没有发布者。

检查：

```bash
ros2 topic info /utlidar/range_info -v
```

实测：

```text
Publisher count: 0
Subscription count: 1
```

这说明不是 DDS 整体断开，而是当时没有机载雷达驱动发布 `/utlidar/cloud`。

---

## 8. 步骤四：确认 Mid-360 网络连通

检查雷达地址：

```bash
ping -I eth0 -c 3 -W 1 192.168.123.120
```

实测：

```text
3 packets transmitted
3 received
0% packet loss
```

检查邻居表：

```bash
ip neigh show dev eth0
```

实测能够看到：

```text
192.168.123.120 lladdr 0c:9a:e6:b6:a7:e8 REACHABLE
```

结论：Mid-360 已开机，并且 G1 的内部 `eth0` 能直接访问雷达。

---

## 9. 步骤五：查找 G1 上已有的 Livox 驱动

检查 ROS 包：

```bash
ros2 pkg list | grep -Ei 'livox|lidar|utlidar|pointcloud' || true
```

实测：

```text
livox_pro
livox_ros_driver2
pointcloud_localization
pointcloud_to_laserscan
```

查找工作空间：

```bash
find "$HOME" \
  -maxdepth 4 \
  -type d \
  \( -iname '*livox*' -o -iname '*lidar*' \) \
  2>/dev/null | sort
```

最终确认已有官方驱动工作空间：

```text
/home/unitree/test_livox
/home/unitree/test_livox/src/livox_ros_driver2
/home/unitree/test_livox/install/livox_ros_driver2
```

检查版本：

```bash
cd ~/test_livox/src/livox_ros_driver2
git remote -v
git branch --show-current
git log -1 --oneline
```

实测：

```text
origin https://github.com/Livox-SDK/livox_ros_driver2.git
branch: master
tag/version: 1.2.6
```

此处只读取仓库信息，没有修改源码。

---

## 10. 步骤六：创建本次测试专用的 Mid-360 配置

加载官方驱动：

```bash
source /opt/ros/foxy/setup.bash
source "$HOME/test_livox/install/setup.bash"
```

将原配置复制到 `/tmp`，不改源码和 install 目录：

```bash
cp \
  "$HOME/test_livox/install/livox_ros_driver2/share/livox_ros_driver2/config/MID360_config.json" \
  /tmp/MID360_g1_qiaomu.json
```

将数据接收主机设置为 G1 的 `eth0` 地址，并设置雷达 IP：

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path('/tmp/MID360_g1_qiaomu.json')
cfg = json.loads(path.read_text())

host_ip = '192.168.123.164'
lidar_ip = '192.168.123.120'

host_info = cfg['MID360']['host_net_info']

if isinstance(host_info, dict):
    for key in (
        'cmd_data_ip',
        'push_msg_ip',
        'point_data_ip',
        'imu_data_ip',
    ):
        host_info[key] = host_ip
elif isinstance(host_info, list):
    for item in host_info:
        item['host_ip'] = host_ip
        if 'lidar_ip' in item:
            item['lidar_ip'] = [lidar_ip]

for lidar in cfg['lidar_configs']:
    lidar['ip'] = lidar_ip

path.write_text(json.dumps(cfg, indent=2) + '\n')
print(path.read_text())
PY
```

本阶段没有在该临时配置中正式应用 G1 的 `roll=180°` 作为最终方案；原因是后续准备直接换用能同时修正点云和雷达内部 IMU 的 deepglint 修改版驱动。

---

## 11. 步骤七：启动官方驱动，输出标准 PointCloud2

先确认端口没有被其他 Livox 驱动占用：

```bash
sudo ss -lunp | grep -E \
':(56101|56201|56301|56401|56501)\b' \
|| echo "Livox 接收端口当前未被占用"
```

设置当前终端 DDS：

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
export ROS2CLI_DISABLE_DAEMON=1
export CYCLONEDDS_URI="file://$HOME/cyclonedds_eth0.xml"
```

启动节点：

```bash
ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \
  -r __node:=livox_g1_test_qiaomu \
  -p xfer_format:=0 \
  -p multi_topic:=0 \
  -p data_src:=0 \
  -p publish_freq:=10.0 \
  -p output_data_type:=0 \
  -p frame_id:=livox_frame \
  -p lvx_file_path:=/tmp/unused.lvx \
  -p user_config_path:=/tmp/MID360_g1_qiaomu.json \
  -p cmdline_input_bd_code:=livox0000000001
```

参数说明：

```text
xfer_format=0       输出 sensor_msgs/msg/PointCloud2
publish_freq=10.0   目标点云发布频率 10 Hz
frame_id=livox_frame
```

该终端需要保持运行。按 `Ctrl+C` 只停止本次临时节点。

---

## 12. 步骤八：验证点云和 IMU 话题

新开一个 G1 SSH 终端：

```bash
source /opt/ros/foxy/setup.bash
source "$HOME/test_livox/install/setup.bash"

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
export ROS2CLI_DISABLE_DAEMON=1
export CYCLONEDDS_URI="file://$HOME/cyclonedds_eth0.xml"
```

列出话题：

```bash
ros2 topic list -t | grep -Ei 'livox|lidar|imu'
```

实测新增：

```text
/livox/lidar [sensor_msgs/msg/PointCloud2]
/livox/imu [sensor_msgs/msg/Imu]
```

检查点云发布者：

```bash
ros2 topic info /livox/lidar -v
```

实测：

```text
Publisher count: 1
Node name: livox_g1_test_qiaomu
Reliability: RELIABLE
Durability: VOLATILE
```

检查 IMU：

```bash
ros2 topic info /livox/imu -v
```

实测同样有一个发布者：

```text
Node name: livox_g1_test_qiaomu
```

检查频率：

```bash
timeout 30 ros2 topic hz /livox/lidar
```

长期测试逐渐收敛到约：

```text
average rate: 7.646 Hz
```

短时间隔约 `0.097 s`，说明部分帧能够接近 10 Hz；但存在最长约 `1.3 s` 的偶发间隔。

IMU：

```bash
timeout 10 ros2 topic hz /livox/imu
```

实测稳定约：

```text
200 Hz
```

---

## 13. 步骤九：轻量检查一帧 PointCloud2

Foxy 的 `ros2 topic echo` 对大型 PointCloud2 不方便，而且该版本 CLI 不支持当前 Humble 常用的部分 `--once` 用法。因此使用轻量 Python 订阅器。

创建脚本：

```bash
cat > /tmp/check_livox_cloud.py <<'PY'
#!/usr/bin/env python3

import math
import struct
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)
from sensor_msgs.msg import PointCloud2


class CloudChecker(Node):
    def __init__(self):
        super().__init__('livox_cloud_checker')

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.received = False
        self.subscription = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.callback,
            qos,
        )

    def callback(self, msg: PointCloud2):
        if self.received:
            return

        self.received = True
        point_count = int(msg.width) * int(msg.height)

        print('===== POINTCLOUD METADATA =====')
        print(f'frame_id       : {msg.header.frame_id}')
        print(f'timestamp      : {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}')
        print(f'height         : {msg.height}')
        print(f'width          : {msg.width}')
        print(f'point_count    : {point_count}')
        print(f'point_step     : {msg.point_step}')
        print(f'row_step       : {msg.row_step}')
        print(f'data_bytes     : {len(msg.data)}')
        print(f'is_bigendian   : {msg.is_bigendian}')
        print(f'is_dense       : {msg.is_dense}')

        print('\n===== FIELDS =====')
        offsets = {}

        for field in msg.fields:
            offsets[field.name] = field.offset
            print(
                f'name={field.name:<12} '
                f'offset={field.offset:<3} '
                f'datatype={field.datatype} '
                f'count={field.count}'
            )

        if not {'x', 'y', 'z'}.issubset(offsets):
            print('\n没有找到完整的 x/y/z 字段')
            return

        endian = '>' if msg.is_bigendian else '<'
        print('\n===== FIRST VALID XYZ POINTS =====')
        valid = 0

        for index in range(min(point_count, 100)):
            row = index // msg.width
            column = index % msg.width
            base = row * msg.row_step + column * msg.point_step

            try:
                x = struct.unpack_from(endian + 'f', msg.data, base + offsets['x'])[0]
                y = struct.unpack_from(endian + 'f', msg.data, base + offsets['y'])[0]
                z = struct.unpack_from(endian + 'f', msg.data, base + offsets['z'])[0]
            except struct.error:
                break

            if all(math.isfinite(v) for v in (x, y, z)):
                print(f'[{index:3d}] x={x: .4f}, y={y: .4f}, z={z: .4f}')
                valid += 1

            if valid >= 5:
                break

        if valid == 0:
            print('前 100 个点中没有找到有效 XYZ 点')


def main():
    rclpy.init()
    node = CloudChecker()
    deadline = time.monotonic() + 15.0

    while rclpy.ok() and not node.received and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=1.0)

    if not node.received:
        print('15 秒内没有收到 /livox/lidar')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
PY
```

运行：

```bash
python3 /tmp/check_livox_cloud.py
```

实测单帧元数据：

```text
frame_id       : livox_frame
height         : 1
width          : 20064
point_count    : 20064
point_step     : 26
row_step       : 521664
data_bytes     : 521664
is_bigendian   : False
is_dense       : True
```

字段：

```text
x            offset=0   datatype=7
 y           offset=4   datatype=7
z            offset=8   datatype=7
intensity    offset=12  datatype=7
tag          offset=16  datatype=2
line         offset=17  datatype=2
timestamp    offset=18  datatype=8
```

有效坐标示例：

```text
x=-0.8220, y=3.3540, z=0.3280
x=-0.8510, y=3.0090, z=0.3570
x=-0.9350, y=2.9150, z=0.4140
```

点云中也可能存在 `(0, 0, 0)` 的无效回波点，后续预处理时过滤即可。

---

## 14. 步骤十：检查网络丢包

```bash
ip -s link show eth0
```

实测：

```text
RX errors: 0
RX dropped: 0
TX errors: 0
TX dropped: 0
```

因此当时的点云频率波动不是明显的以太网物理层丢包导致的。

---

## 15. 步骤十一：经 Wi-Fi 向外部电脑发布，用 RViz 验证

这一步只用于观察原始点云，不作为最终实时 FAST-LIO 网络架构。

### 15.1 G1 侧绑定 wlan0

先在原驱动终端按 `Ctrl+C` 停止节点，然后创建临时 DDS 配置：

```bash
cat > /tmp/cyclonedds_livox_wifi.xml <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS>
  <Domain id="any">
    <General>
      <Interfaces>
        <NetworkInterface
          name="wlan0"
          priority="default"
          multicast="default"/>
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
    </General>

    <Discovery>
      <Peers>
        <Peer address="172.16.10.136"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
EOF
```

设置环境：

```bash
source /opt/ros/foxy/setup.bash
source "$HOME/test_livox/install/setup.bash"

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
export ROS_DOMAIN_ID=0
export ROS2CLI_DISABLE_DAEMON=1
export CYCLONEDDS_URI="file:///tmp/cyclonedds_livox_wifi.xml"
```

重新运行同一个 Livox 节点：

```bash
ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \
  -r __node:=livox_g1_test_qiaomu \
  -p xfer_format:=0 \
  -p multi_topic:=0 \
  -p data_src:=0 \
  -p publish_freq:=10.0 \
  -p output_data_type:=0 \
  -p frame_id:=livox_frame \
  -p lvx_file_path:=/tmp/unused.lvx \
  -p user_config_path:=/tmp/MID360_g1_qiaomu.json \
  -p cmdline_input_bd_code:=livox0000000001
```

雷达 UDP 数据仍从 G1 的 `eth0` 接收；这里只是让该 ROS 2 节点通过 `wlan0` 被外部电脑发现。

### 15.2 外部电脑设置 CycloneDDS

```bash
source /opt/ros/humble/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
export ROS_DOMAIN_ID=0
export ROS2CLI_DISABLE_DAEMON=1
```

自动查找通向 G1 的 Wi-Fi 接口：

```bash
WIFI_IF=$(ip route get 172.16.10.18 \
  | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')

echo "Wi-Fi interface: $WIFI_IF"
```

实测接口：

```text
wlp7s0
```

创建配置：

```bash
cat > /tmp/cyclonedds_g1_wifi.xml <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS>
  <Domain id="any">
    <General>
      <Interfaces>
        <NetworkInterface
          name="$WIFI_IF"
          priority="default"
          multicast="default"/>
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
    </General>

    <Discovery>
      <Peers>
        <Peer address="172.16.10.18"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
EOF

export CYCLONEDDS_URI="file:///tmp/cyclonedds_g1_wifi.xml"
```

确认跨机器发现成功：

```bash
ros2 topic list -t | grep -E '^/livox'
```

实测：

```text
/livox/imu [sensor_msgs/msg/Imu]
/livox/lidar [sensor_msgs/msg/PointCloud2]
```

电脑端测得完整点云仅约：

```text
0.5–0.6 Hz
```

一帧约 `0.5 MB`，10 Hz 时仅原始载荷就约 `5 MB/s`，还未包括 DDS、UDP/IP 和重传开销。因此 Wi-Fi 只用于本次低频 RViz 验证。

---

## 16. 步骤十二：安装并配置 RViz2

外部电脑安装：

```bash
conda deactivate 2>/dev/null || true
sudo apt update
sudo apt install -y ros-humble-rviz2
```

启动前重新加载环境：

```bash
source /opt/ros/humble/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
export ROS_DOMAIN_ID=0
export ROS2CLI_DISABLE_DAEMON=1
export CYCLONEDDS_URI="file:///tmp/cyclonedds_g1_wifi.xml"

rviz2
```

RViz 设置：

```text
Global Options
  Fixed Frame: livox_frame

PointCloud2
  Topic: /livox/lidar
  Reliability Policy: Reliable
  Durability Policy: Volatile
  Style: Points
  Size (Pixels): 2
  Decay Time: 10 或更大
  Position Transformer: XYZ
  Color Transformer: Intensity
```

为了只看雷达自身坐标系，不需要额外 TF，直接设置：

```text
Fixed Frame = livox_frame
```

若点云出现但视角不合适：

- 选中 `PointCloud2` 后按 `F` 聚焦；
- 使用鼠标滚轮缩放；
- 将 `Style` 改为 `Points`，不要使用尺寸过大的 `Flat Squares`；
- Wi-Fi 帧率很低时，将 `Decay Time` 设为 `10–30 s`，避免每帧很快消失。

最终 RViz 已成功看到原始点云。

---

## 17. 为什么点云上下颠倒

G1 上的 Mid-360 是倒置安装的。本阶段使用的是官方：

```text
Livox-SDK/livox_ros_driver2 1.2.6
```

本阶段临时配置尚未正式使用 `roll=180°` 作为最终修正，同时官方驱动不能满足后续方案要求的“点云和雷达内部 IMU 同时按 G1 倒装外参修正”。

因此 RViz 中看到上下颠倒的原始点云是符合预期的，不是通信失败。

不要通过 RViz 的相机设置或 `Invert Z Axis` 掩盖数据问题。最终应在驱动层正确修正点云和 IMU 坐标。

---

## 18. 如何停止和恢复

停止本次 Livox 节点：

```text
在启动 livox_g1_test_qiaomu 的终端按 Ctrl+C
```

检查没有残留：

```bash
pgrep -af 'livox_ros_driver2_node|livox_g1_test_qiaomu' || \
echo "临时 Livox 驱动已停止"
```

退出 SSH 或关闭终端后，本次 `export` 的环境变量自动失效。

由于没有改 `~/.bashrc`、源码或系统服务，不需要执行恢复脚本。

---

## 19. 下一阶段计划

下一阶段不再继续使用官方驱动做最终修正，而是在 G1 上建立独立工作空间，安装：

```text
deepglint/FAST_LIO_LOCALIZATION_HUMANOID
中的修改版 livox_ros_driver2
```

目标链路：

```text
G1 本机 Mid-360
  ↓ 修改版 livox_ros_driver2
/livox/custom_msg
/livox/imu（倒装外参已修正）
  ↓ G1 本地 rosbag2 录制
Wi-Fi 复制 rosbag 到外部电脑
  ↓
Ubuntu 22.04 + ROS 2 Humble 离线运行 FAST-LIO
```

下一阶段的具体目标：

1. 新建独立的 G1 修改版驱动工作空间，不覆盖 `~/test_livox`。
2. 设置：

   ```text
   host IP: 192.168.123.164
   lidar IP: 192.168.123.120
   roll: 180°
   ```

3. 确认修改版驱动同时旋转点云和 Mid-360 内部 IMU。
4. 输出：

   ```text
   /livox/custom_msg ≈ 10 Hz
   /livox/imu ≈ 200 Hz
   ```

5. 在 G1 本地录制 `/livox/custom_msg` 与 `/livox/imu`。
6. 复制 rosbag 到外部电脑。
7. 在电脑上离线运行 FAST-LIO，验证里程计、轨迹和累积点云地图。
8. 离线链路稳定后，再决定最终实时部署使用：
   - G1 机载计算机；
   - 机器人随身计算机；
   - 或开发阶段有线连接外部电脑。

---

## 20. 当前阶段状态汇总

| 阶段 | 状态 |
|---|---|
| SSH 与基础网络 | 已完成 |
| Unitree ROS 2/DDS 话题发现 | 已完成 |
| Mid-360 网络连通 | 已完成 |
| 官方 Livox 驱动启动 | 已完成 |
| `/livox/lidar` PointCloud2 | 已完成 |
| `/livox/imu` | 已完成 |
| 点云内容检查 | 已完成 |
| 外部电脑 Wi-Fi 订阅 | 已完成 |
| RViz 原始点云显示 | 已完成 |
| 发现倒装点云 | 已完成 |
| deepglint 修改版驱动 | 未开始，下一阶段 |
| `/livox/custom_msg` | 未开始 |
| rosbag2 录制 | 未开始 |
| FAST-LIO 里程计/建图 | 未开始 |
| 高程图 | 未开始 |
