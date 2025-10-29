#!/usr/bin/env python3
"""
3DOF 仿真结果 GIF 动画生成脚本

基于 CSV 数据生成车辆轨迹和状态的动画 GIF，包括：
- 车辆在 XY 平面的运动轨迹
- 实时的速度矢量和航向角
- 时间序列图表（yaw rate, lateral acceleration）

使用方法：
    python3 examples/three_dof_animate.py examples/out/three_dof_demo.csv
    python3 examples/three_dof_animate.py examples/out/three_dof_sine.csv --fps 10 --duration 5.0
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image
import io

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_simulation_data(csv_path):
    """加载仿真 CSV 数据"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    required_cols = ['t', 'x', 'y', 'psi', 'vx', 'vy', 'r']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")
    
    return df

def create_animation_frames(df, fps=15, duration=None):
    """创建动画帧数据"""
    if duration is None:
        duration = df['t'].iloc[-1]
    
    # 计算需要的帧数和时间步长
    total_frames = int(fps * duration)
    time_points = np.linspace(0, duration, total_frames)
    
    # 插值获取每帧的数据
    frames_data = []
    for t in time_points:
        # 找到最接近的时间点
        idx = np.argmin(np.abs(df['t'] - t))
        frame_data = df.iloc[idx].copy()
        frame_data['frame_time'] = t
        frames_data.append(frame_data)
    
    return frames_data

def setup_figure():
    """设置图形布局"""
    fig = plt.figure(figsize=(16, 10))
    
    # 主轨迹图 (左侧，较大)
    ax_traj = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
    ax_traj.set_aspect('equal')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.set_title('Vehicle Trajectory')
    
    # 横摆角速度图 (右上)
    ax_yaw = plt.subplot2grid((3, 3), (0, 2))
    ax_yaw.grid(True, alpha=0.3)
    ax_yaw.set_xlabel('Time (s)')
    ax_yaw.set_ylabel('Yaw Rate (rad/s)')
    ax_yaw.set_title('r(t)')
    
    # 侧向加速度图 (右中)
    ax_ay = plt.subplot2grid((3, 3), (1, 2))
    ax_ay.grid(True, alpha=0.3)
    ax_ay.set_xlabel('Time (s)')
    ax_ay.set_ylabel('Lateral Acc (m/s²)')
    ax_ay.set_title('ay(t)')
    
    # 速度信息 (右下)
    ax_info = plt.subplot2grid((3, 3), (2, 2))
    ax_info.axis('off')
    
    plt.tight_layout()
    return fig, ax_traj, ax_yaw, ax_ay, ax_info

def animate_frame(frame_idx, frames_data, df, ax_traj, ax_yaw, ax_ay, ax_info, 
                 traj_line, vehicle_patch, velocity_arrow, yaw_line, ay_line, info_text):
    """动画帧更新函数"""
    frame_data = frames_data[frame_idx]
    current_time = frame_data['frame_time']
    
    # 更新轨迹线（显示到当前时间）
    mask = df['t'] <= current_time
    traj_x = df.loc[mask, 'x']
    traj_y = df.loc[mask, 'y']
    traj_line.set_data(traj_x, traj_y)
    
    # 车辆位置和朝向
    x, y, psi = frame_data['x'], frame_data['y'], frame_data['psi']
    vx, vy = frame_data['vx'], frame_data['vy']
    
    # 更新车辆矩形 (简化为2m x 1m的矩形)
    car_length, car_width = 2.0, 1.0
    vehicle_patch.set_xy((x - car_length/2 * np.cos(psi) + car_width/2 * np.sin(psi),
                         y - car_length/2 * np.sin(psi) - car_width/2 * np.cos(psi)))
    vehicle_patch.angle = np.degrees(psi)
    
    # 更新速度矢量箭头
    v_scale = 0.5  # 速度矢量缩放
    velocity_arrow.set_positions((x, y), (x + vx * v_scale, y + vy * v_scale))
    
    # 更新时间序列图
    time_mask = df['t'] <= current_time
    time_data = df.loc[time_mask, 't']
    
    # 横摆角速度
    yaw_data = df.loc[time_mask, 'r']
    yaw_line.set_data(time_data, yaw_data)
    
    # 侧向加速度 (如果存在)
    if 'ay' in df.columns:
        ay_data = df.loc[time_mask, 'ay']
        ay_line.set_data(time_data, ay_data)
    
    # 更新信息文本
    speed = np.sqrt(vx**2 + vy**2) * 3.6  # km/h
    info_text.set_text(f'Time: {current_time:.2f} s\n'
                      f'Speed: {speed:.1f} km/h\n'
                      f'Yaw Rate: {frame_data["r"]:.3f} rad/s\n'
                      f'Position: ({x:.1f}, {y:.1f}) m')
    
    return [traj_line, vehicle_patch, velocity_arrow, yaw_line, ay_line, info_text]

def create_gif_animation(csv_path, output_path=None, fps=15, duration=None, dpi=100):
    """创建并保存 GIF 动画"""
    # 加载数据
    df = load_simulation_data(csv_path)
    frames_data = create_animation_frames(df, fps, duration)
    
    # 设置输出路径
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_dir = os.path.dirname(csv_path)
        output_path = os.path.join(output_dir, f"{base_name}_anim.gif")
    
    # 创建图形
    fig, ax_traj, ax_yaw, ax_ay, ax_info = setup_figure()
    
    # 设置轨迹图的范围
    x_range = df['x'].max() - df['x'].min()
    y_range = df['y'].max() - df['y'].min()
    margin = max(x_range, y_range) * 0.1
    
    ax_traj.set_xlim(df['x'].min() - margin, df['x'].max() + margin)
    ax_traj.set_ylim(df['y'].min() - margin, df['y'].max() + margin)
    
    # 设置时间序列图的范围
    ax_yaw.set_xlim(0, df['t'].iloc[-1])
    ax_yaw.set_ylim(df['r'].min() * 1.1, df['r'].max() * 1.1)
    
    if 'ay' in df.columns:
        ax_ay.set_xlim(0, df['t'].iloc[-1])
        ax_ay.set_ylim(df['ay'].min() * 1.1, df['ay'].max() * 1.1)
    
    # 初始化绘图元素
    traj_line, = ax_traj.plot([], [], 'b-', linewidth=2, label='Trajectory')
    vehicle_patch = Rectangle((0, 0), 2.0, 1.0, angle=0, facecolor='red', alpha=0.7)
    ax_traj.add_patch(vehicle_patch)
    
    velocity_arrow = FancyArrowPatch((0, 0), (0, 0), 
                                   arrowstyle='->', mutation_scale=20, 
                                   color='green', linewidth=2)
    ax_traj.add_patch(velocity_arrow)
    
    yaw_line, = ax_yaw.plot([], [], 'r-', linewidth=2)
    ay_line, = ax_ay.plot([], [], 'g-', linewidth=2)
    
    info_text = ax_info.text(0.1, 0.7, '', fontsize=10, verticalalignment='top',
                           transform=ax_info.transAxes)
    
    ax_traj.legend()
    
    # 创建动画
    print(f"Creating animation with {len(frames_data)} frames...")
    
    anim = animation.FuncAnimation(
        fig, animate_frame, frames=len(frames_data),
        fargs=(frames_data, df, ax_traj, ax_yaw, ax_ay, ax_info,
               traj_line, vehicle_patch, velocity_arrow, yaw_line, ay_line, info_text),
        interval=1000/fps, blit=False, repeat=True
    )
    
    # 保存为 GIF
    print(f"Saving GIF to {output_path}...")
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    
    plt.close(fig)
    print(f"GIF animation saved successfully!")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate GIF animation from 3DOF simulation CSV')
    parser.add_argument('csv_path', help='Path to the CSV file')
    parser.add_argument('--output', '-o', help='Output GIF path (default: auto-generated)')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second (default: 15)')
    parser.add_argument('--duration', type=float, help='Animation duration in seconds (default: full simulation)')
    parser.add_argument('--dpi', type=int, default=100, help='Output resolution DPI (default: 100)')
    
    args = parser.parse_args()
    
    try:
        output_path = create_gif_animation(
            args.csv_path, 
            args.output, 
            args.fps, 
            args.duration,
            args.dpi
        )
        print(f"Animation complete: {output_path}")
        
    except Exception as e:
        print(f"Error creating animation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()