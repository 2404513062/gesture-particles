"""
╔══════════════════════════════════════════════════════════════╗
║          手势控制粒子交互系统                                  ║
║          Gesture-Controlled Particle System                  ║
╚══════════════════════════════════════════════════════════════╝

新手使用指南 (Beginner's Guide):
  第1步：安装 Python 3.8+ (python.org 下载)
  第2步：双击运行 install_and_run.bat (自动安装依赖并启动)
  或者手动: pip install -r requirements.txt
  第3步：在摄像头前做出手势，观察粒子效果！

支持的手势 (Supported Gestures):
  ✋ 张开手掌 → 🌸 花朵绽放
  ✊ 握拳     → 🔥 火焰爆发
  ✌️ 剪刀手   → 🌊 水波涟漪
  ☝️ 食指指向 → ⭐ 星光轨迹
  🤟 摇滚手势 → 💕 爱心粒子
  👍 大拇指   → 🎆 烟花效果
  👌 OK手势  → 🌌 银河漩涡

按键控制:
  Q / ESC → 退出
  H       → 显示/隐藏帮助
  1-7     → 手动切换粒子效果
  R       → 清除所有粒子
  Space   → 截图保存
"""

import sys
import io

# 修复 Windows 下 GBK 编码不支持 emoji 的问题
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        pass

import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
from collections import deque, Counter
from enum import Enum, auto
import os

# PyInstaller 打包后查找资源文件的路径
def _resource_path(relative_path):
    """获取资源文件的绝对路径（兼容 PyInstaller 打包）"""
    if getattr(sys, 'frozen', False):
        # 打包后运行：资源在临时解压目录
        base_path = sys._MEIPASS
    else:
        # 开发时运行：资源在脚本所在目录
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# ============================================================
# 1. 全局配置 - 可以在这里调整参数
# ============================================================
CAMERA_WIDTH = 1280          # 摄像头宽度
CAMERA_HEIGHT = 720          # 摄像头高度
MAX_PARTICLES = 400          # 最大粒子数量
GESTURE_CONFIRM_FRAMES = 3   # 手势确认所需帧数（防止误识别）
OVERLAY_DECAY = 0.90         # 粒子拖尾衰减速度（越小拖尾越短）
DETECTION_EVERY_N_FRAMES = 2 # 每N帧检测一次手势（提升性能）
WINDOW_NAME = "手势粒子交互系统 - Gesture Particle System"


class GestureType(Enum):
    """手势类型枚举"""
    OPEN_PALM = auto()   # 张开手掌 ✋
    FIST = auto()        # 握拳 ✊
    PEACE = auto()       # 剪刀手 ✌️
    POINT = auto()       # 食指指向 ☝️
    ROCK = auto()        # 摇滚手势 🤟
    THUMBS_UP = auto()   # 大拇指 👍
    OK_SIGN = auto()     # OK手势 👌
    NO_HAND = auto()     # 没有检测到手
    UNKNOWN = auto()     # 未知手势


class Particle:
    """单个粒子"""
    __slots__ = ('x', 'y', 'vx', 'vy', 'life', 'max_life',
                 'color', 'size', 'max_size', 'effect_data')

    def __init__(self, x, y, vx, vy, life, color, size):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.life = float(life)
        self.max_life = float(life)
        self.color = color  # BGR tuple
        self.size = float(size)
        self.max_size = float(size)
        self.effect_data = {}  # 效果专用数据

    def is_alive(self):
        return self.life > 0

    def alpha(self):
        """返回当前透明度比例 (0.0 ~ 1.0)"""
        return max(0.0, self.life / self.max_life)


# ============================================================
# 2. 手势检测器
# ============================================================
class HandDetector:
    """基于 MediaPipe Tasks API 的手势检测器"""

    def __init__(self):
        # 使用新版 MediaPipe Tasks API
        base_options = mp.tasks.BaseOptions(
            model_asset_path=_resource_path('hand_landmarker.task')
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,                        # 最多检测2只手
            min_hand_detection_confidence=0.7,  # 检测置信度阈值
            min_hand_presence_confidence=0.5,   # 手部存在置信度
            min_tracking_confidence=0.5,        # 追踪置信度阈值
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

        # 手势平滑：记录最近的手势以过滤误识别
        self._gesture_history = deque(maxlen=GESTURE_CONFIRM_FRAMES)
        self._current_gesture = GestureType.NO_HAND
        self._timestamp_ms = 0  # 视频时间戳（毫秒）

    def detect(self, frame_rgb):
        """检测手势，返回 (手势类型, 手部数据)"""
        # 转换为 MediaPipe Image 格式
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 累加时间戳（模拟 30fps）
        self._timestamp_ms += 33
        result = self.detector.detect_for_video(mp_image, self._timestamp_ms)

        if not result.hand_landmarks:
            self._update_gesture_history(GestureType.NO_HAND)
            return self._get_stable_gesture(), None

        # 取第一只手
        landmarks = result.hand_landmarks[0]  # List[NormalizedLandmark], 共21个
        handedness = result.handedness[0][0].category_name  # 'Left' 或 'Right'

        # 识别手势
        gesture = self._classify_gesture(landmarks, handedness)
        self._update_gesture_history(gesture)

        # 提取手部关键点数据
        hand_data = self._extract_hand_data(landmarks, handedness)
        hand_data['landmarks'] = landmarks  # 保存landmarks用于绘制
        hand_data['result'] = result        # 保存完整结果用于绘制

        return self._get_stable_gesture(), hand_data

    def _classify_gesture(self, lm, handedness):
        """
        根据21个手部关键点识别手势

        MediaPipe 手部关键点索引:
              8  (食指指尖)
        7  6  5  (食指 DIP, PIP, MCP)
        12  (中指指尖)
        11 10 9  (中指 DIP, PIP, MCP)
        16  (无名指指尖)
        15 14 13 (无名指 DIP, PIP, MCP)
        20  (小指指尖)
        19 18 17 (小指 DIP, PIP, MCP)
        4  (拇指指尖)
        3  2  1  (拇指 IP, MCP, CMC)
        0  (手腕)
        """
        # --- 判断拇指是否伸展 ---
        # 方法：比较拇指尖到食指根的距离 vs 拇指IP到食指根的距离
        thumb_tip = lm[4]
        thumb_ip = lm[3]
        index_mcp = lm[5]

        tip_dist = math.sqrt((thumb_tip.x - index_mcp.x)**2 +
                            (thumb_tip.y - index_mcp.y)**2)
        ip_dist = math.sqrt((thumb_ip.x - index_mcp.x)**2 +
                           (thumb_ip.y - index_mcp.y)**2)
        thumb_extended = tip_dist > ip_dist * 1.3

        # --- 判断其他四指是否伸展 ---
        # 指尖的 y 坐标小于 PIP 关节的 y 坐标 → 手指伸直
        # (图像坐标系中 y 轴向下，所以更小的 y = 更高 = 伸直)
        finger_tips = [8, 12, 16, 20]    # 食指、中指、无名指、小指指尖
        finger_pips = [6, 10, 14, 18]    # 对应 PIP 关节

        finger_states = [thumb_extended]
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            extended = lm[tip_idx].y < lm[pip_idx].y - 0.02  # 小阈值避免抖动
            finger_states.append(extended)

        # thumb, index, middle, ring, pinky
        t, i, m, r, p = finger_states

        # --- 手势分类 ---
        # OK手势：拇指和食指指尖靠近（形成圆圈）
        thumb_tip_pos = lm[4]
        index_tip_pos = lm[8]
        ok_distance = math.sqrt((thumb_tip_pos.x - index_tip_pos.x)**2 +
                                (thumb_tip_pos.y - index_tip_pos.y)**2)

        if ok_distance < 0.05 and m and r and p:
            return GestureType.OK_SIGN

        # 根据手指伸展模式分类
        num_extended = sum(finger_states)

        if num_extended == 5:
            return GestureType.OPEN_PALM       # 五指全伸 → 张开手掌
        elif num_extended == 0:
            return GestureType.FIST            # 五指全握 → 拳头
        elif i and m and not r and not p and not t:
            return GestureType.PEACE           # 食指+中指 → 剪刀手
        elif i and not m and not r and not p and not t:
            return GestureType.POINT           # 仅食指 → 指向
        elif t and i and not m and not r and p:
            return GestureType.ROCK            # 拇指+食指+小指 → 摇滚
        elif t and not i and not m and not r and not p:
            return GestureType.THUMBS_UP       # 仅拇指 → 点赞

        return GestureType.UNKNOWN

    def _extract_hand_data(self, lm, handedness):
        """提取手部关键位置数据"""
        h, w = CAMERA_HEIGHT, CAMERA_WIDTH

        # 手掌中心：手腕和中指根部的中点
        palm_x = int((lm[0].x + lm[9].x) / 2 * w)
        palm_y = int((lm[0].y + lm[9].y) / 2 * h)

        # 各指尖位置
        fingertips = {
            'thumb':  (int(lm[4].x * w),  int(lm[4].y * h)),
            'index':  (int(lm[8].x * w),  int(lm[8].y * h)),
            'middle': (int(lm[12].x * w), int(lm[12].y * h)),
            'ring':   (int(lm[16].x * w), int(lm[16].y * h)),
            'pinky':  (int(lm[20].x * w), int(lm[20].y * h)),
        }

        return {
            'palm_center': (palm_x, palm_y),
            'fingertips': fingertips,
            'handedness': handedness,
        }

    def _update_gesture_history(self, gesture):
        self._gesture_history.append(gesture)

    def _get_stable_gesture(self):
        """返回稳定的手势（需要连续多帧确认）"""
        if len(self._gesture_history) < GESTURE_CONFIRM_FRAMES:
            return GestureType.NO_HAND

        # 最近N帧中最常见的手势
        counter = Counter(self._gesture_history)
        most_common = counter.most_common(1)[0]

        # 如果最常见的出现次数超过一半，确认手势
        if most_common[1] >= GESTURE_CONFIRM_FRAMES // 2 + 1:
            self._current_gesture = most_common[0]
        else:
            self._current_gesture = GestureType.UNKNOWN

        return self._current_gesture

    def draw_hand(self, frame, hand_data):
        """在帧上绘制手部骨架"""
        if 'result' not in hand_data:
            return
        result = hand_data['result']
        if result.hand_landmarks:
            from mediapipe.tasks.python.vision import drawing_utils, HandLandmarksConnections
            for hand_landmarks in result.hand_landmarks:
                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,  # 21个 NormalizedLandmark 的列表
                    HandLandmarksConnections.HAND_CONNECTIONS,
                )

    def close(self):
        self.detector.close()


# ============================================================
# 3. 粒子效果系统
# ============================================================
class ParticleSystem:
    """粒子系统 - 管理和渲染所有粒子"""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.particles = []
        self.overlay = np.zeros((height, width, 3), dtype=np.uint8)

        # 效果计时器
        self._spawn_timers = {effect: 0.0 for effect in GestureType}

    @property
    def count(self):
        return len(self.particles)

    def spawn(self, x, y, vx, vy, life, color, size, effect_data=None):
        """创建一个新粒子"""
        if len(self.particles) >= MAX_PARTICLES:
            # 移除最老的粒子
            self.particles.pop(0)

        p = Particle(x, y, vx, vy, life, color, size)
        if effect_data:
            p.effect_data = effect_data
        self.particles.append(p)

    def spawn_burst(self, cx, cy, count, speed_range, life_range,
                    color_gen, size_range, angle_range=None):
        """
        以某点为中心爆发式生成粒子

        参数:
          cx, cy: 爆发中心
          count: 粒子数量
          speed_range: (最小速度, 最大速度)
          life_range: (最短生命, 最长生命)
          color_gen: 颜色生成函数，每次调用返回一个BGR颜色
          size_range: (最小尺寸, 最大尺寸)
          angle_range: (起始角度, 结束角度)，None则为全方向
        """
        min_speed, max_speed = speed_range
        min_life, max_life = life_range
        min_size, max_size = size_range

        if angle_range is None:
            angle_range = (0, 2 * math.pi)

        for _ in range(count):
            angle = random.uniform(*angle_range)
            speed = random.uniform(min_speed, max_speed)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = random.uniform(min_life, max_life)
            color = color_gen()
            size = random.uniform(min_size, max_size)

            self.spawn(cx, cy, vx, vy, life, color, size)

    def update(self, dt):
        """更新所有粒子状态"""
        alive = []
        for p in self.particles:
            if not p.is_alive():
                continue

            # 位置更新
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.life -= dt

            # 边界检查 - 超出屏幕的粒子加速消失
            if (p.x < -50 or p.x > self.width + 50 or
                p.y < -50 or p.y > self.height + 50):
                p.life = 0
                continue

            alive.append(p)

        self.particles = alive

    def render_overlay(self):
        """渲染粒子到叠加层，返回叠加层图像"""
        # 拖尾效果：让上一帧的粒子逐渐消失
        self.overlay = (self.overlay * OVERLAY_DECAY).astype(np.uint8)

        # 绘制所有粒子
        for p in self.particles:
            alpha = p.alpha()

            # 粒子随生命衰减而变小变暗
            current_size = max(1, int(p.size * (0.3 + 0.7 * alpha)))

            # 主粒子颜色（考虑透明度）
            color = tuple(int(c * (0.4 + 0.6 * alpha)) for c in p.color)

            x, y = int(p.x), int(p.y)

            # 绘制发光效果：外层光晕
            if current_size > 2:
                glow_size = current_size + 2
                glow_color = tuple(int(c * 0.3 * alpha) for c in p.color)
                cv2.circle(self.overlay, (x, y), glow_size, glow_color, -1)

            # 绘制粒子主体
            cv2.circle(self.overlay, (x, y), current_size, color, -1)

            # 高亮核心
            if current_size > 3:
                core_size = max(1, current_size // 2)
                core_color = tuple(min(255, int(c * 1.5)) for c in color)
                cv2.circle(self.overlay, (x, y), core_size, core_color, -1)

        return self.overlay

    def clear(self):
        """清除所有粒子"""
        self.particles.clear()
        self.overlay.fill(0)


# ============================================================
# 4. 各种粒子效果生成器
# ============================================================
class EffectEngine:
    """效果引擎 - 根据手势生成不同的粒子效果"""

    def __init__(self, particle_system, width, height):
        self.ps = particle_system
        self.width = width
        self.height = height
        self.time = 0.0

        # 水波效果状态
        self._ripple_centers = deque(maxlen=3)
        self._ripple_timer = 0.0

        # 花朵效果状态
        self._flower_angle = 0.0
        self._flower_timer = 0.0

        # 烟花效果状态
        self._firework_timer = 0.0

        # 银河效果状态
        self._galaxy_angle = 0.0

        # 爱心效果状态
        self._heart_timer = 0.0

    def update_time(self, dt):
        self.time += dt

    # ========== 颜色生成器 ==========

    @staticmethod
    def flower_color():
        """花朵颜色：粉色系"""
        b = random.randint(180, 255)  # 粉色偏蓝分量
        g = random.randint(50, 180)
        r = random.randint(200, 255)
        return (b, g, r)

    @staticmethod
    def fire_color():
        """火焰颜色：红橙黄"""
        t = random.random()
        if t < 0.3:
            return (0, random.randint(200, 255), 255)       # 黄
        elif t < 0.7:
            return (0, random.randint(100, 180), 255)       # 橙
        else:
            return (random.randint(0, 30), random.randint(0, 60), 255)  # 红

    @staticmethod
    def water_color():
        """水波颜色：蓝色系"""
        b = random.randint(180, 255)
        g = random.randint(150, 255)
        r = random.randint(50, 150)
        return (b, g, r)

    @staticmethod
    def star_color():
        """星星颜色：金白"""
        t = random.random()
        if t < 0.5:
            return (random.randint(200, 255), random.randint(210, 255), 255)  # 白
        elif t < 0.8:
            return (0, random.randint(200, 255), 255)                         # 金
        else:
            return (random.randint(150, 255), random.randint(230, 255), 255)  # 浅金

    @staticmethod
    def heart_color():
        """爱心颜色：红粉"""
        b = random.randint(50, 150)
        g = random.randint(20, 100)
        r = random.randint(220, 255)
        return (b, g, r)

    @staticmethod
    def galaxy_color():
        """银河颜色：紫蓝紫红"""
        t = random.random()
        if t < 0.33:
            return (random.randint(180, 255), random.randint(0, 60), random.randint(180, 255))  # 紫
        elif t < 0.66:
            return (random.randint(150, 255), random.randint(50, 150), random.randint(200, 255))  # 蓝紫
        else:
            return (random.randint(200, 255), random.randint(100, 200), random.randint(150, 200))  # 浅紫

    @staticmethod
    def firework_color():
        """烟花颜色：随机鲜艳色彩"""
        colors = [
            (0, 255, 255),     # 黄
            (0, 165, 255),     # 橙
            (0, 0, 255),       # 红
            (255, 0, 255),     # 品红
            (255, 255, 0),     # 青
            (0, 255, 0),       # 绿
            (255, 0, 0),       # 蓝
        ]
        base = random.choice(colors)
        # 稍微变色
        return tuple(min(255, max(0, c + random.randint(-30, 30))) for c in base)

    # ========== 效果生成函数 ==========

    def effect_flower_bloom(self, hand_data, dt):
        """🌸 花朵绽放 - 从掌心绽放花瓣状粒子"""
        cx, cy = hand_data['palm_center']
        self._flower_timer += dt
        self._flower_angle += dt * 0.5  # 花朵缓慢旋转

        # 持续生成花瓣粒子
        petals = 6  # 六瓣花
        particles_per_petal = 2

        for petal in range(petals):
            base_angle = self._flower_angle + (2 * math.pi / petals) * petal

            for _ in range(particles_per_petal):
                # 花瓣形状：粒子沿曲线分布
                angle = base_angle + random.uniform(-0.3, 0.3)
                dist = random.uniform(30, 120)
                # 添加波浪形偏移
                wave = math.sin(dist * 0.05 + self._flower_angle * 3) * 20

                px = cx + math.cos(angle) * dist + math.cos(angle + math.pi/2) * wave
                py = cy + math.sin(angle) * dist + math.sin(angle + math.pi/2) * wave

                # 速度方向：从中心向外 + 切向旋转
                speed = random.uniform(40, 100)
                vx = math.cos(angle) * speed + random.uniform(-20, 20)
                vy = math.sin(angle) * speed + random.uniform(-20, 20)

                life = random.uniform(1.5, 3.0)
                color = self.flower_color()
                size = random.uniform(4, 12)

                self.ps.spawn(px, py, vx, vy, life, color, size,
                             {'type': 'flower', 'angle': angle, 'dist': dist})

        # 花心粒子
        if self._flower_timer > 0.3:
            self._flower_timer = 0.0
            for _ in range(5):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(10, 30)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                self.ps.spawn(cx, cy, vx, vy, random.uniform(1.0, 2.0),
                             (0, random.randint(200, 255), 255),  # 金色花心
                             random.uniform(2, 5))

    def effect_fire_burst(self, hand_data, dt):
        """🔥 火焰爆发 - 从拳头爆发出火焰粒子"""
        cx, cy = hand_data['palm_center']

        # 主爆发（间歇性大爆发）
        self.ps.spawn_burst(
            cx, cy,
            count=8,
            speed_range=(80, 250),
            life_range=(0.8, 2.0),
            color_gen=self.fire_color,
            size_range=(5, 15),
        )

        # 上升的火星
        self.ps.spawn_burst(
            cx, cy - 20,
            count=5,
            speed_range=(20, 80),
            life_range=(1.0, 2.5),
            color_gen=self.fire_color,
            size_range=(2, 6),
            angle_range=(-math.pi * 0.7, -math.pi * 0.3),  # 向上方向
        )

        # 烟雾粒子（灰色，缓慢上升）
        for _ in range(2):
            angle = random.uniform(-math.pi * 0.6, -math.pi * 0.4)
            speed = random.uniform(10, 40)
            vx = math.cos(angle) * speed + random.uniform(-10, 10)
            vy = math.sin(angle) * speed
            smoke_gray = random.randint(60, 120)
            self.ps.spawn(cx + random.uniform(-20, 20), cy,
                         vx, vy, random.uniform(1.5, 3.0),
                         (smoke_gray, smoke_gray, smoke_gray),
                         random.uniform(8, 20))

    def effect_water_ripple(self, hand_data, dt):
        """🌊 水波涟漪 - 从食指和中指指尖扩散波纹"""
        fingertips = hand_data['fingertips']
        self._ripple_timer += dt

        # 从两个指尖生成波纹
        for finger_name in ['index', 'middle']:
            fx, fy = fingertips[finger_name]

            # 持续生成环形波纹粒子
            for _ in range(3):
                angle = random.uniform(0, 2 * math.pi)
                # 波纹以固定速度向外扩散
                ripple_speed = random.uniform(60, 120)
                vx = math.cos(angle) * ripple_speed
                vy = math.sin(angle) * ripple_speed

                life = random.uniform(1.5, 3.5)
                color = self.water_color()
                size = random.uniform(2, 5)

                self.ps.spawn(fx, fy, vx, vy, life, color, size,
                             {'type': 'ripple', 'origin': (fx, fy)})

        # 周期性生成强波纹
        if self._ripple_timer > 0.5:
            self._ripple_timer = 0.0
            for finger_name in ['index', 'middle']:
                fx, fy = fingertips[finger_name]
                self.ps.spawn_burst(
                    fx, fy, count=15,
                    speed_range=(30, 100),
                    life_range=(2.0, 4.0),
                    color_gen=self.water_color,
                    size_range=(2, 6),
                )

    def effect_star_trail(self, hand_data, dt):
        """⭐ 星光轨迹 - 指尖留下闪烁的星点轨迹"""
        index_tip = hand_data['fingertips']['index']
        fx, fy = index_tip

        # 在指尖周围生成闪烁星点
        for _ in range(4):
            # 随机散布在指尖周围
            offset_x = random.uniform(-15, 15)
            offset_y = random.uniform(-15, 15)

            # 微小随机速度（飘散效果）
            vx = random.uniform(-30, 30)
            vy = random.uniform(-30, 30)

            life = random.uniform(0.3, 1.2)
            color = self.star_color()
            size = random.uniform(1, 4)

            self.ps.spawn(fx + offset_x, fy + offset_y, vx, vy, life, color, size)

        # 偶尔生成较大闪光
        if random.random() < 0.3:
            vx = random.uniform(-15, 15)
            vy = random.uniform(-15, 15)
            self.ps.spawn(fx, fy, vx, vy, random.uniform(0.5, 1.5),
                         (255, 255, 255),  # 亮白闪光
                         random.uniform(3, 8))

    def effect_heart_particles(self, hand_data, dt):
        """💕 爱心粒子 - 生成心形扩散的粉色粒子"""
        cx, cy = hand_data['palm_center']
        self._heart_timer += dt

        # 心形公式: x = 16sin³(t), y = 13cos(t) - 5cos(2t) - 2cos(3t) - cos(4t)
        count = 3
        for _ in range(count):
            t = random.uniform(0, 2 * math.pi)
            # 心形坐标（缩放）
            scale = random.uniform(3, 8)
            hx = cx + scale * 16 * math.sin(t)**3
            hy = cy - scale * (13 * math.cos(t) - 5 * math.cos(2*t) -
                              2 * math.cos(3*t) - math.cos(4*t))

            # 从心形向外扩散
            dx = hx - cx
            dy = hy - cy
            dist = math.sqrt(dx**2 + dy**2) + 0.01

            speed = random.uniform(20, 60)
            vx = (dx / dist) * speed + random.uniform(-15, 15)
            vy = (dy / dist) * speed + random.uniform(-15, 15)

            life = random.uniform(1.5, 3.0)
            color = self.heart_color()
            size = random.uniform(3, 10)

            self.ps.spawn(hx, hy, vx, vy, life, color, size,
                         {'type': 'heart'})

        # 周期性心形爆发
        if self._heart_timer > 1.0:
            self._heart_timer = 0.0
            for _ in range(30):
                t = random.uniform(0, 2 * math.pi)
                scale = random.uniform(4, 10)
                hx = cx + scale * 16 * math.sin(t)**3
                hy = cy - scale * (13 * math.cos(t) - 5 * math.cos(2*t) -
                                  2 * math.cos(3*t) - math.cos(4*t))

                dx = hx - cx
                dy = hy - cy
                dist = math.sqrt(dx**2 + dy**2) + 0.01
                speed = random.uniform(30, 90)

                self.ps.spawn(
                    hx, hy,
                    (dx/dist) * speed, (dy/dist) * speed,
                    random.uniform(1.0, 2.5),
                    self.heart_color(),
                    random.uniform(2, 8)
                )

    def effect_firework(self, hand_data, dt):
        """🎆 烟花效果 - 粒子上升然后爆炸"""
        cx, cy = hand_data['palm_center']
        self._firework_timer += dt

        # 从拇指位置发射上升粒子
        thumb_tip = hand_data['fingertips']['thumb']
        tx, ty = thumb_tip

        # 上升的"火箭"粒子
        for _ in range(2):
            vx = random.uniform(-30, 30)
            vy = random.uniform(-200, -100)  # 向上
            self.ps.spawn(tx + random.uniform(-10, 10), ty,
                         vx, vy, random.uniform(0.8, 1.5),
                         (0, 255, 255),  # 金色火箭
                         random.uniform(3, 6),
                         {'type': 'firework_rise', 'explode_y': ty - random.uniform(100, 250)})

        # 周期性烟花爆炸
        if self._firework_timer > 2.0:
            self._firework_timer = 0.0
            # 在手掌上方爆炸
            boom_x = cx + random.uniform(-80, 80)
            boom_y = cy - random.uniform(100, 250)

            self.ps.spawn_burst(
                boom_x, boom_y,
                count=60,
                speed_range=(50, 300),
                life_range=(1.0, 3.0),
                color_gen=self.firework_color,
                size_range=(2, 8),
            )
            # 内层更亮的粒子
            self.ps.spawn_burst(
                boom_x, boom_y,
                count=20,
                speed_range=(20, 100),
                life_range=(0.5, 1.5),
                color_gen=lambda: (255, 255, 255),
                size_range=(1, 4),
            )

        # 检查是否有上升粒子到达爆炸高度（遍历副本避免修改冲突）
        for p in list(self.ps.particles):
            if p.effect_data.get('type') == 'firework_rise' and p.y <= p.effect_data.get('explode_y', 0):
                # 小爆炸
                self.ps.spawn_burst(
                    p.x, p.y,
                    count=15,
                    speed_range=(30, 150),
                    life_range=(0.5, 1.5),
                    color_gen=self.firework_color,
                    size_range=(2, 6),
                )
                p.life = 0  # 移除上升粒子

    def effect_galaxy_spiral(self, hand_data, dt):
        """🌌 银河漩涡 - 粒子围绕中心旋转形成漩涡"""
        cx, cy = hand_data['palm_center']
        self._galaxy_angle += dt * 2.0  # 旋转速度

        # 螺旋生成粒子
        arms = 3  # 三条旋臂
        particles_per_arm = 3

        for arm in range(arms):
            base_angle = self._galaxy_angle + (2 * math.pi / arms) * arm

            for _ in range(particles_per_arm):
                # 对数螺旋: r = a * e^(b*theta)
                dist = random.uniform(20, 150)
                angle = base_angle + dist * 0.03  # 螺旋弯曲

                px = cx + math.cos(angle) * dist
                py = cy + math.sin(angle) * dist

                # 切向速度（绕转） + 径向速度（向外扩散）
                tangential_speed = random.uniform(30, 80)
                radial_speed = random.uniform(10, 40)

                vx = (-math.sin(angle) * tangential_speed +
                      math.cos(angle) * radial_speed)
                vy = (math.cos(angle) * tangential_speed +
                      math.sin(angle) * radial_speed)

                life = random.uniform(2.0, 5.0)
                color = self.galaxy_color()
                size = random.uniform(1, 6)

                self.ps.spawn(px, py, vx, vy, life, color, size,
                             {'type': 'galaxy', 'orbit_angle': angle, 'orbit_dist': dist})

        # 中心吸引粒子
        for _ in range(2):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(30, 80)
            px = cx + math.cos(angle) * dist
            py = cy + math.sin(angle) * dist

            # 指向中心的速度
            dx = cx - px
            dy = cy - py
            speed = random.uniform(20, 50)

            self.ps.spawn(px, py, dx * 0.02, dy * 0.02,
                         random.uniform(0.5, 1.5),
                         (255, 255, 255),
                         random.uniform(1, 3))

    def effect_ambient(self, dt):
        """🌫️ 环境粒子 - 没有检测到手时的默认氛围粒子"""
        # 从屏幕底部缓慢升起
        if random.random() < 0.3:
            x = random.uniform(0, self.width)
            y = self.height + 10
            vx = random.uniform(-15, 15)
            vy = random.uniform(-40, -10)  # 缓慢上升

            # 柔和的暖色
            color = (
                random.randint(100, 180),
                random.randint(120, 200),
                random.randint(180, 255),
            )
            size = random.uniform(2, 8)
            life = random.uniform(3.0, 6.0)

            self.ps.spawn(x, y, vx, vy, life, color, size,
                         {'type': 'ambient'})

    def generate(self, gesture, hand_data, dt):
        """根据手势类型生成对应的粒子效果"""
        if gesture == GestureType.OPEN_PALM:
            self.effect_flower_bloom(hand_data, dt)
        elif gesture == GestureType.FIST:
            self.effect_fire_burst(hand_data, dt)
        elif gesture == GestureType.PEACE:
            self.effect_water_ripple(hand_data, dt)
        elif gesture == GestureType.POINT:
            self.effect_star_trail(hand_data, dt)
        elif gesture == GestureType.ROCK:
            self.effect_heart_particles(hand_data, dt)
        elif gesture == GestureType.THUMBS_UP:
            self.effect_firework(hand_data, dt)
        elif gesture == GestureType.OK_SIGN:
            self.effect_galaxy_spiral(hand_data, dt)
        elif gesture in (GestureType.NO_HAND, GestureType.UNKNOWN):
            self.effect_ambient(dt)


# ============================================================
# 5. UI 渲染
# ============================================================
class UIRenderer:
    """界面渲染器 - 在画面上显示信息和帮助"""

    # 手势到中文名称的映射
    GESTURE_NAMES = {
        GestureType.OPEN_PALM:  '张开手掌 ✋  花朵绽放',
        GestureType.FIST:       '握拳 ✊  火焰爆发',
        GestureType.PEACE:      '剪刀手 ✌️  水波涟漪',
        GestureType.POINT:      '食指指向 ☝️  星光轨迹',
        GestureType.ROCK:       '摇滚手势 🤟  爱心粒子',
        GestureType.THUMBS_UP:  '大拇指 👍  烟花效果',
        GestureType.OK_SIGN:    'OK手势 👌  银河漩涡',
        GestureType.NO_HAND:    '未检测到手 🖐️  环境粒子',
        GestureType.UNKNOWN:    '未知手势 ❓  环境粒子',
    }

    def __init__(self):
        self.show_help = True
        self._fps_history = deque(maxlen=30)
        self._last_time = time.time()
        self._fps = 0.0

    def update_fps(self):
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            self._fps_history.append(1.0 / dt)
        if self._fps_history:
            self._fps = sum(self._fps_history) / len(self._fps_history)

    def draw(self, frame, gesture, particle_count, manual_mode=False):
        """在帧上绘制UI信息"""
        h, w = frame.shape[:2]
        fps_text = f"FPS: {self._fps:.0f}"
        count_text = f"粒子数: {particle_count}"
        gesture_text = f"手势: {self.GESTURE_NAMES.get(gesture, '未知')}"
        mode_text = "手动模式" if manual_mode else "自动识别模式"

        # 半透明背景面板（左上角）
        panel = frame.copy()
        panel_h = 140 if self.show_help else 80
        cv2.rectangle(panel, (10, 10), (450, panel_h), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(frame, 0.7, panel, 0.3, 0)

        # 文字信息
        y = 40
        cv2.putText(frame, gesture_text, (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 30
        cv2.putText(frame, f"{fps_text}  |  {count_text}  |  {mode_text}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # 帮助面板
        if self.show_help:
            y += 35
            cv2.putText(frame, "按键: Q/Esc-退出 H-隐藏帮助 1-7-切换效果 R-清除 Space-截图",
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y += 22
            cv2.putText(frame, "手势: ✋花朵 ✊火焰 ✌️水波 ☝️星光 🤟爱心 👍烟花 👌银河",
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # 手未检测到时在中央显示提示
        if gesture == GestureType.NO_HAND:
            text = "请将手放入摄像头画面中"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 40
            # 底部提示
            cv2.rectangle(frame, (text_x - 15, text_y - 30),
                         (text_x + text_size[0] + 15, text_y + 10), (0, 0, 0), -1)
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)


# ============================================================
# 6. 主程序
# ============================================================
def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║       🌟 手势控制粒子交互系统 启动中... 🌟                ║
╚══════════════════════════════════════════════════════════╝
    """)

    # --- 初始化摄像头 ---
    print("[1/4] 打开摄像头...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("❌ 错误：无法打开摄像头！请检查摄像头是否连接。")
        print("   如果使用笔记本电脑，请确认摄像头未被其他应用占用。")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   摄像头已打开: {actual_w}x{actual_h}")

    # --- 初始化各模块 ---
    print("[2/4] 加载手势检测模型（首次加载可能需要下载）...")
    detector = HandDetector()
    print("   手势检测模型加载完成！")

    print("[3/4] 初始化粒子系统...")
    particle_system = ParticleSystem(actual_w, actual_h)
    effect_engine = EffectEngine(particle_system, actual_w, actual_h)
    print(f"   粒子系统就绪 (最大粒子数: {MAX_PARTICLES})")

    print("[4/4] 初始化界面...")
    ui = UIRenderer()

    # --- 运行状态 ---
    running = True
    frame_count = 0
    manual_gesture = None       # 手动模式下的效果
    cached_gesture = GestureType.NO_HAND  # 缓存上次检测结果
    cached_hand_data = None     # 缓存上次手部数据
    prev_time = time.time()

    print("""
✅ 系统启动完成！在摄像头前做出以下手势：
   ✋ 张开手掌 → 🌸 花朵绽放
   ✊ 握拳     → 🔥 火焰爆发
   ✌️ 剪刀手   → 🌊 水波涟漪
   ☝️ 食指指向 → ⭐ 星光轨迹
   🤟 摇滚手势 → 💕 爱心粒子
   👍 大拇指   → 🎆 烟花效果
   👌 OK手势  → 🌌 银河漩涡

   按 H 切换帮助显示，按 Q 退出
""")

    # --- 主循环 ---
    while running:
        # 计算帧间隔时间
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        dt = min(dt, 0.1)  # 限制最大dt，防止卡顿后粒子跳跃

        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头画面")
            break

        # 水平翻转（镜像效果，更自然）
        frame = cv2.flip(frame, 1)
        frame_count += 1

        # --- 手势检测（每N帧检测一次以提升性能）---
        if manual_gesture is not None:
            # 手动模式：手势固定，但仍跟踪手部位置
            current_gesture = manual_gesture
            if frame_count % DETECTION_EVERY_N_FRAMES == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, hand_data = detector.detect(frame_rgb)
                if hand_data is not None:
                    cached_hand_data = hand_data
                else:
                    hand_data = cached_hand_data
            else:
                hand_data = cached_hand_data
        elif frame_count % DETECTION_EVERY_N_FRAMES == 0:
            # 自动模式 + 检测帧：执行手势识别
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_gesture, hand_data = detector.detect(frame_rgb)
            cached_gesture = current_gesture
            if hand_data is not None:
                cached_hand_data = hand_data
        else:
            # 自动模式 + 非检测帧：使用缓存的结果
            current_gesture = cached_gesture
            hand_data = cached_hand_data

        # --- 生成粒子效果 ---
        effect_engine.update_time(dt)

        if hand_data or manual_gesture is not None:
            # 手动模式创建虚拟手部数据
            if hand_data is None and manual_gesture is not None:
                hand_data = {
                    'palm_center': (actual_w // 2, actual_h // 2),
                    'fingertips': {
                        'thumb': (actual_w // 2, actual_h // 2),
                        'index': (actual_w // 2, actual_h // 2 - 50),
                        'middle': (actual_w // 2, actual_h // 2 - 70),
                        'ring': (actual_w // 2, actual_h // 2 - 30),
                        'pinky': (actual_w // 2, actual_h // 2 - 20),
                    },
                    'handedness': 'Right',
                }
            effect_engine.generate(current_gesture, hand_data, dt)
        else:
            effect_engine.effect_ambient(dt)

        # --- 更新粒子 ---
        particle_system.update(dt)

        # --- 渲染 ---
        overlay = particle_system.render_overlay()
        result = cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)

        # 绘制手部骨架
        if hand_data and 'landmarks' in hand_data and manual_gesture is None:
            detector.draw_hand(result, hand_data)

        # 绘制UI
        ui.update_fps()
        ui.draw(result, current_gesture, particle_system.count,
                manual_gesture is not None)

        # --- 显示 ---
        cv2.imshow(WINDOW_NAME, result)

        # --- 按键处理 ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # Q 或 ESC 退出
            running = False
        elif key == ord('h') or key == ord('H'):
            ui.show_help = not ui.show_help
        elif key == ord('r') or key == ord('R'):
            particle_system.clear()
            print("🧹 已清除所有粒子")
        elif key == ord(' '):  # 空格截图
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, result)
            print(f"📸 截图已保存: {filename}")
        elif key == ord('1'):
            manual_gesture = GestureType.OPEN_PALM
            print("🎯 手动切换: 花朵绽放")
        elif key == ord('2'):
            manual_gesture = GestureType.FIST
            print("🎯 手动切换: 火焰爆发")
        elif key == ord('3'):
            manual_gesture = GestureType.PEACE
            print("🎯 手动切换: 水波涟漪")
        elif key == ord('4'):
            manual_gesture = GestureType.POINT
            print("🎯 手动切换: 星光轨迹")
        elif key == ord('5'):
            manual_gesture = GestureType.ROCK
            print("🎯 手动切换: 爱心粒子")
        elif key == ord('6'):
            manual_gesture = GestureType.THUMBS_UP
            print("🎯 手动切换: 烟花效果")
        elif key == ord('7'):
            manual_gesture = GestureType.OK_SIGN
            print("🎯 手动切换: 银河漩涡")
        elif key == ord('0') or key == ord('`'):
            manual_gesture = None
            print("🔄 切换回自动识别模式")

    # --- 清理（先关窗口，再释放资源，防止卡死）---
    print("\n正在关闭...")
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # 让窗口系统处理关闭事件
    cap.release()
    # MediaPipe detector.close() 可能会卡住，用 try 保护
    try:
        detector.close()
    except Exception:
        pass  # 忽略清理时的错误，反正进程马上就结束了
    print("已退出。")


if __name__ == '__main__':
    main()
