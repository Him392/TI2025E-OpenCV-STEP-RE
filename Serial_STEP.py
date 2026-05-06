import serial
import struct
import time
# 创建串口连接对象  
serial_pan_port = 'COM9' 
serial_tilt_port = 'COM10'
ser_pan = serial.Serial('COM9', 115200)  # zdt默认115200，8N1,0x6B校验位，别动
ser_tilt = serial.Serial('COM10', 115200) 

def send_step_pan(ser_pan, direction, speed, accel, pulses):
    """
    构建并发送电机控制指令（固定校验位 0x6B）
    :param ser: 已初始化的 serial.Serial 对象
    :param direction: 方向 (0x00/0x01)
    :param speed: 速度 (最大 0x0BB8),定义的是最大速度，若距离短则受加速度限制，实际速度可能达不到设定值
    :param accel: 加速度 (0x00-0xFF)，目前推荐252
    :param pulses: 脉冲数 (4字节)，最大4294967295 ，一圈64000脉冲，每个脉冲对应0.005625度
    """
    addr = 0x01
    func_code = 0xFD
    mode = 0x02
    sync_flag = 0x00
    checksum = 0x6B  # 参数 默认写死
    
    # 构建字节包 (使用大端序 >)
    # B:1字节, H:2字节, I:4字节
    packet = struct.pack('>BB B H B I B B B', 
                         addr,        # 字节1
                         func_code,   # 字节2
                         direction,   # 字节3
                         speed,       # 字节4-5
                         accel,       # 字节6
                         pulses,      # 字节7-10
                         mode,        # 字节11
                         sync_flag,   # 字节12
                         checksum)    # 字节13 (固定为 0x6B)

    try:
        ser_pan.write(packet)
        # 打印调试信息，方便核对
        print(f"OK-speed: {speed}rpm, accel: {accel}, pulses: {pulses} ,{packet.hex(' ').upper()}")
    except Exception as e:
        print(f"ERR!!! {e}")

while True:
    send_step_pan(ser_pan, 0x01, 600, 252, 64000)# 
    time.sleep(10)  # 等待1秒
