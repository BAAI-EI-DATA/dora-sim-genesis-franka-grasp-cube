import numpy as np
from pynput import keyboard
import math


class KeyboardController:
    def __init__(self):
        self.keys_pressed = set()
        self.target_pos = np.array([0.4, 0.0, 0.4])  # 初始目标位置
        self.target_euler = np.array([0.0, 0.0, 0.0])  # 欧拉角 (roll, pitch, yaw)，单位：弧度
        self.position_step = 0.002  # 位置移动步长
        self.orientation_step = 0.005  # 姿态旋转步长，单位：弧度
        self.gripper_force = np.array([3.0, 3.0])  # 夹爪力控制

        # 启动键盘监听
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.listener.start()

    def on_press(self, key):
        try:
            self.keys_pressed.add(key.char)
        except AttributeError:
            self.keys_pressed.add(key)

    def on_release(self, key):
        try:
            self.keys_pressed.discard(key.char)
        except AttributeError:
            self.keys_pressed.discard(key)

    def euler_to_quat(self, roll, pitch, yaw):
        """
        将欧拉角转换为四元数
        使用ZYX顺序（先绕Z轴旋转，然后Y轴，最后X轴）
        参数：roll(x), pitch(y), yaw(z) 单位：弧度
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])  # 四元数格式：[w, x, y, z]
        # return np.array([0, 1, 0, 0])

    def update_target_from_keyboard(self):
        """根据按键更新目标位置和姿态"""
        # 位置控制 (使用方向键和功能键)
        if keyboard.Key.up in self.keys_pressed:  # 向前 (X轴+)
            self.target_pos[0] += self.position_step
        if keyboard.Key.down in self.keys_pressed:  # 向后 (X轴-)
            self.target_pos[0] -= self.position_step
        if keyboard.Key.left in self.keys_pressed:  # 向左 (Y轴+)
            self.target_pos[1] += self.position_step
        if keyboard.Key.right in self.keys_pressed:  # 向右 (Y轴-)
            self.target_pos[1] -= self.position_step
        if "=" in self.keys_pressed or "+" in self.keys_pressed:  # 向上 (Z轴+)
            self.target_pos[2] += self.position_step
        if "-" in self.keys_pressed:  # 向下 (Z轴-)
            self.target_pos[2] -= self.position_step

        # 姿态控制 (欧拉角) - 使用数字键
        if "6" in self.keys_pressed:  # 增加绕X轴旋转 (roll+)
            self.target_euler[0] += self.orientation_step
        if "4" in self.keys_pressed:  # 减少绕X轴旋转 (roll-)
            self.target_euler[0] -= self.orientation_step
        if "8" in self.keys_pressed:  # 增加绕Y轴旋转 (pitch+)
            self.target_euler[1] += self.orientation_step
        if "2" in self.keys_pressed:  # 减少绕Y轴旋转 (pitch-)
            self.target_euler[1] -= self.orientation_step
        if "7" in self.keys_pressed:  # 增加绕Z轴旋转 (yaw+)
            self.target_euler[2] += self.orientation_step
        if "9" in self.keys_pressed:  # 减少绕Z轴旋转 (yaw-)
            self.target_euler[2] -= self.orientation_step

        # 重置姿态 (使用空格键)
        if keyboard.Key.space in self.keys_pressed:
            self.target_euler = np.array([math.pi, 0.0, 0.0])

        # 夹爪控制 (使用b/n键)
        if "b" in self.keys_pressed:  # 夹紧
            self.gripper_force = np.array([-3.0, -3.0])
        elif "n" in self.keys_pressed:  # 松开
            self.gripper_force = np.array([3.0, 3.0])

        # 位置重置 (使用退格键)
        if keyboard.Key.backspace in self.keys_pressed:
            self.target_pos = np.array([0.4, 0.0, 0.4])
            self.target_euler = np.array([math.pi, 0.0, 0.0])

    def get_target_quat(self):
        """获取当前目标姿态的四元数表示"""
        return self.euler_to_quat(
            self.target_euler[0],  # roll
            self.target_euler[1],  # pitch
            self.target_euler[2],  # yaw
        )

    def print_controls(self):
        """打印控制说明"""
        print("\n" + "=" * 50)
        print("机械臂键盘控制说明")
        print("=" * 50)
        print("位置控制:")
        print("  方向键↑/↓ - 前/后移动 (X轴)")
        print("  方向键←/→ - 左/右移动 (Y轴)")
        print("  +/= 键   - 向上移动 (Z轴)")
        print("  - 键     - 向下移动 (Z轴)")
        print("\n姿态控制 (欧拉角):")
        print("  6/4 键 - 增加/减少绕X轴旋转 (Roll)")
        print("  8/2 键 - 增加/减少绕Y轴旋转 (Pitch)")
        print("  7/9 键 - 增加/减少绕Z轴旋转 (Yaw)")
        print("  空格键  - 重置姿态为初始值")
        print("\n夹爪控制:")
        print("  B 键 - 夹紧")
        print("  N 键 - 松开")
        print("\n其他:")
        print("  退格键 - 重置位置和姿态")
        print("  P 键  - 打印当前位置和姿态")
        print("  ESC键 - 退出程序")
        print("=" * 50)
        print(f"当前位置: {self.target_pos}")
        print(f"当前欧拉角 (roll,pitch,yaw): {self.target_euler}")
        print(f"四元数: {self.get_target_quat()}")
        print("=" * 50 + "\n")
