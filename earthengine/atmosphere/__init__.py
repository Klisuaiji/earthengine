"""Atmosphere / Climate 模块：太阳辐射、温度、环流 (20 / 21)。

输入：``Planet + Latitude + Ocean + Elevation + Terrain``。
第一阶段用物理约束 + 参数化模型（非完整 GCM）：纬度温度梯度、海陆温差、
山地雨影、迎风坡降水、背风坡干燥、大尺度湿润/干旱带。
"""

from . import solar, temperature, circulation

__all__ = ["solar", "temperature", "circulation"]
