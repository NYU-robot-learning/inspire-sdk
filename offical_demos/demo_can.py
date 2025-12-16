import serial
import time

# 寄存器字典
regdict = {
    'ID': 1000,
    'baudrate': 1001,
    'clearErr': 1004,
    'forceClb': 1009,
    'angleSet': 1486,
    'forceSet': 1498,
    'speedSet': 1522,  
    'angleAct': 1546,
    'forceAct': 1582,
    'errCode': 1606,
    'statusCode': 1612,
    'temp': 1618,
    'actionSeq': 2320,
    'actionRun': 2322
}

def init_serial(port='/dev/ttyUSB0', baudrate=115200):
    """初始化串口连接"""
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"串口 {port} 已打开")
        return ser
    except Exception as e:
        print(f"打开串口失败: {e}")
        return None

def convert_address_to_binary_write(address_dec, id_value):
    """将十进制地址转换为二进制字符串，用于写操作，前缀第六位为1。"""
    address_bin = bin(address_dec)[2:]  
    id_bin = bin(int(id_value, 2))[2:].zfill(14) 
    formatted_output = f"0000010{address_bin}{id_bin}"  # 第六位为1
    return formatted_output

def convert_address_to_binary_read(address_dec, id_value):
    """将十进制地址转换为二进制字符串，用于读操作，前缀第六位为0。"""
    address_bin = bin(address_dec)[2:]  
    id_bin = bin(int(id_value, 2))[2:].zfill(14) 
    formatted_output = f"0000000{address_bin}{id_bin}"  # 第六位为0
    return formatted_output

def binary_to_hex(binary_string):
    """将二进制字符串转换为十六进制"""
    decimal_value = int(binary_string, 2)
    return hex(decimal_value)[2:].upper()

def convert_number_to_bytes(number):
    """将一个整数转换为字节数组"""
    hex_string = hex(number)[2:].upper()  
    if len(hex_string) % 2 != 0:
        hex_string = '0' + hex_string  
    byte_array = [int(hex_string[i:i+2], 16) for i in range(0, len(hex_string), 2)][::-1]
    return byte_array

def send_command(ser, reg_name, values, id_value):
    """发送控制命令"""
    if reg_name in ['angleSet', 'forceSet', 'speedSet']:
        val_reg = [val & 0xFFFF for val in values]  
        write_register(ser, regdict[reg_name], val_reg, id_value)
    else:
        print('函数调用错误，正确方式：str的值为\'angleSet\'/\'forceSet\'/\'speedSet\'，val为长度为6的list，值为0~1000，允许使用-1作为占位符')

def write_register(ser, address, values, id_value):
    """写入寄存器并发送数据"""
    while True:
        chunk = values[:4]  # 取前4个数据
        values = values[4:]

        # 使用写操作的地址转换函数
        ext_id_bin = convert_address_to_binary_write(address, id_value)
        ext_id_hex = binary_to_hex(ext_id_bin)
        print(f"计算的扩展标识符: {ext_id_hex}")

        ext_id_number = int(ext_id_hex, 16)
        ext_id_bytes = convert_number_to_bytes(ext_id_number)

        print("提取的扩展标识符字节:")
        for byte in ext_id_bytes:
            print(f"字节: {byte:02X}")

        send_buffer = bytearray()
        send_buffer += bytes([0xAA, 0xAA])
        send_buffer.extend(ext_id_bytes)

        # 添加注册值
        for value in chunk:
            send_buffer.append(value & 0xFF)
            send_buffer.append(value >> 8)

        # 计算当前数据长度并添加额外的填充
        data_length = len(chunk) * 2
        if data_length < 8:
            padding_length = 8 - data_length
            send_buffer.extend([0xFF] * padding_length)

        # 添加数据长度和额外的固定字节
        send_buffer.append(len(chunk) * 2 + (padding_length if data_length < 8 else 0))  # 数据长度
        send_buffer.append(0x00)
        send_buffer.append(0x01)
        send_buffer.append(0x00)

        # 计算校验和
        check_sum = sum(send_buffer[2:]) & 0xFF
        send_buffer.append(check_sum)
        send_buffer += bytes([0x55, 0x55])

        ser.write(send_buffer)
        print("发送指令:", send_buffer.hex())
        ser.reset_input_buffer()  # 清除输入缓冲区
        break  # 仅执行一次

    # 发送后半段
    new_address = address + 8
    print(f"新地址 (后半段): {new_address}")

    ext_id_bin = convert_address_to_binary_write(new_address, id_value)
    ext_id_hex = binary_to_hex(ext_id_bin)
    print(f"计算的扩展标识符 (后半段): {ext_id_hex}")

    ext_id_number = int(ext_id_hex, 16)
    ext_id_bytes = convert_number_to_bytes(ext_id_number)

    print("提取的扩展标识符字节 (后半段):")
    for byte in ext_id_bytes:
        print(f"字节: {byte:02X}")

    # 继续构建发送缓冲区，使用后半段的扩展标识符字节
    send_buffer = bytearray()
    send_buffer += bytes([0xAA, 0xAA])
    send_buffer.extend(ext_id_bytes)

    # 发送剩余数据
    for value in values:
        send_buffer.append(value & 0xFF)
        send_buffer.append(value >> 8)

    # 计算当前数据长度并添加额外的填充
    data_length = len(values) * 2
    if data_length < 8:
        padding_length = 8 - data_length
        send_buffer.extend([0xFF] * padding_length)

    # 添加数据长度和额外的固定字节
    send_buffer.append(0x04)  # 数据长度
    send_buffer.append(0x00)
    send_buffer.append(0x01)
    send_buffer.append(0x00)

    # 计算校验和
    check_sum = sum(send_buffer[2:]) & 0xFF
    send_buffer.append(check_sum)
    send_buffer += bytes([0x55, 0x55])

    ser.write(send_buffer)
    print("发送指令 (后半段):", send_buffer.hex())

def read_register(ser, address, id_value):
    """读取寄存器并发送数据"""
    ext_id_bin = convert_address_to_binary_read(address, id_value)
    ext_id_hex = binary_to_hex(ext_id_bin)
    ext_id_number = int(ext_id_hex, 16)
    ext_id_bytes = convert_number_to_bytes(ext_id_number)

    send_buffer = bytearray()
    send_buffer += bytes([0xAA, 0xAA])
    send_buffer.extend(ext_id_bytes)

    # 读取命令
    send_buffer.append(0x08)  # 固定字节
    send_buffer.append(0x00)  # 固定字节
    send_buffer.append(0x00)  # 固定字节
    send_buffer.append(0x00)  # 固定字节
    send_buffer.append(0x00)  # 固定字节
    send_buffer.append(0x00)  # 固定字节
    send_buffer.append(0x00)  # 固定字节
    send_buffer.append(0x00)  # 固定字节
    send_buffer.append(0x01)  # 固定字节
    send_buffer.append(0x00)  # 固定字节
    send_buffer.append(0x01)  # 固定字节    
    send_buffer.append(0x00)  # 固定字节
    # 计算校验和
    check_sum = sum(send_buffer[2:]) & 0xFF
    send_buffer.append(check_sum)
    send_buffer += bytes([0x55, 0x55])

    ser.write(send_buffer)
    print("发送读取指令:", send_buffer.hex())
    ser.reset_input_buffer()  # 清除输入缓冲区
    
    # 等待设备响应
    time.sleep(0.1)  # 等待设备响应
    response1 = ser.read(23)  # 根据协议读取响应字节数
    print("读取的原始内容:", response1.hex())  # 打印原始内容的十六进制形式

    # 根据 reg_name 处理响应数据
    if address == regdict['temp']:
        # 处理 response1 数据
        if len(response1) >= 1:
            # 提取第 7 位到第 12 位的数据（从第 7 到第 12）
            frame1_data = response1[6:12]  # 从第 7 位到第 12 位

            # 处理第一帧数据
            values1 = []
            for i in range(0, len(frame1_data), 1):
                value = frame1_data[i]
                if value > 60000:
                    value = 0
                values1.append(value)
            print("读取数据（温度）:", tuple(values1))  # 打印温度数据
    else:
        # 处理 response1 数据
        if len(response1) >= 1:
            # 提取第一帧的第 7 位到第 14 位的数据
            frame1_data = response1[6:14]  # 从第 7 位到第 14 位

            # 处理第一帧数据
            values1 = []
            for i in range(0, len(frame1_data), 2):
                low_byte = frame1_data[i]      # 低八位
                high_byte = frame1_data[i + 1] # 高八位
                value = (high_byte << 8) | low_byte  # 组合成十进制
                if value > 60000:
                    value = 0                
                values1.append(value)

        # 发送后半段
        new_address = address + 8
        ext_id_bin = convert_address_to_binary_read(new_address, id_value)
        ext_id_hex = binary_to_hex(ext_id_bin)

        ext_id_number = int(ext_id_hex, 16)
        ext_id_bytes = convert_number_to_bytes(ext_id_number)

        # 继续构建发送缓冲区，使用后半段的扩展标识符字节
        send_buffer = bytearray()
        send_buffer += bytes([0xAA, 0xAA])
        send_buffer.extend(ext_id_bytes)

        # 读取命令
        send_buffer.append(0x04)  # 固定字节
        send_buffer.append(0x00)  # 固定字节
        send_buffer.append(0x00)  # 固定字节
        send_buffer.append(0x00)  # 固定字节
        send_buffer.append(0x00)  # 固定字节
        send_buffer.append(0x00)  # 固定字节
        send_buffer.append(0x00)  # 固定字节
        send_buffer.append(0x00)  # 固定字节
        send_buffer.append(0x01)  # 固定字节
        send_buffer.append(0x00)  # 固定字节
        send_buffer.append(0x01)  # 固定字节    
        send_buffer.append(0x00)  # 固定字节
        # 计算校验和
        check_sum = sum(send_buffer[2:]) & 0xFF
        send_buffer.append(check_sum)
        send_buffer += bytes([0x55, 0x55])

        ser.write(send_buffer)
        print("发送读取指令:", send_buffer.hex())
        ser.reset_input_buffer()  # 清除输入缓冲区
        
        # 等待设备响应
        time.sleep(0.1)  # 等待设备响应
        response2 = ser.read(23)  # 根据协议读取响应字节数
        print("读取的原始内容:", response2.hex())  # 打印原始内容的十六进制形式

        # 处理 response2 数据
        if len(response2) >= 1:
            # 提取第二帧的第 7 位到第 10 位的数据
            frame2_data = response2[6:10]  # 从第 7 位到第 10 位

            # 处理第二帧数据
            values2 = []
            for i in range(0, len(frame2_data), 2):
                low_byte = frame2_data[i]      # 低八位
                high_byte = frame2_data[i + 1] # 高八位
                value = (high_byte << 8) | low_byte  # 组合成十进制
                if value > 60000:
                    value = 0
                values2.append(value)

        # 组合最终输出
        combined_values = values1[:4] + values2[:2]  # 前四个第一帧的值 + 后两个第二帧的值
        print("读取数据:", tuple(combined_values))

def write6(ser, reg_name, val, id_value):
    """写入6个参数的函数"""
    send_command(ser, reg_name, val, id_value)

if __name__ == "__main__":
    port = '/dev/ttyUSB0'  
    ser = init_serial(port)
    if not ser:
        exit(1)

    print('设置灵巧手运动角度参数0，-1为不设置该运动角度！')  
    # 写入
    write6(ser, 'speedSet', [1000, 1000, 1000, 1000, 1000, 1000], '01')
    time.sleep(1)
    write6(ser, 'angleSet', [0, 0, 0, 0, 800, 0], '01')  # ID 设置为 '01'
    time.sleep(3)
    write6(ser, 'speedSet', [200, 200, 200, 200, 200, 200], '01')
    time.sleep(1)
    write6(ser, 'angleSet', [1000, 1000, 1000, 1000, 1000, 1000], '01') 
    time.sleep(3)

    # 读取
    read_register(ser, regdict['angleSet'], '01')
    read_register(ser, regdict['angleAct'], '01')
    read_register(ser, regdict['speedSet'], '01')
    read_register(ser, regdict['forceSet'], '01')
    read_register(ser, regdict['forceAct'], '01')    
    read_register(ser, regdict['temp'], '01')
    ser.close()

