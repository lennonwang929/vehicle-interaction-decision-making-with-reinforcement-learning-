#!/usr/bin/env python3
"""
可视化 DQN Agent 的每个 Episode 奖励
支持多种格式：CSV, JSON, 使用 Matplotlib 绘图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
from pathlib import Path


def plot_rewards(csv_file, output_fig=None, window_size=10):
    """
    绘制 Episode 奖励趋势图
    
    Args:
        csv_file: CSV 文件路径
        output_fig: 输出图片路径 (可选)
        window_size: 移动平均窗口大小
    """
    try:
        df = pd.read_csv(csv_file)
        
        if df.empty:
            print(f"警告: {csv_file} 是空的")
            return
            
        print(f"成功读取 {len(df)} 个 Episode 的数据")
        print(f"奖励统计:")
        print(f"  最小值: {df['reward'].min():.4f}")
        print(f"  最大值: {df['reward'].max():.4f}")
        print(f"  平均值: {df['reward'].mean():.4f}")
        print(f"  标准差: {df['reward'].std():.4f}")
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 第一个子图：原始数据和移动平均
        ax1 = axes[0]
        ax1.plot(df['episode'], df['reward'], label='Episode Reward', 
                alpha=0.6, color='steelblue', marker='o', markersize=3)
        
        # 计算移动平均
        if len(df) >= window_size:
            moving_avg = df['reward'].rolling(window=window_size, center=True).mean()
            ax1.plot(df['episode'], moving_avg, label=f'Moving Average (window={window_size})',
                    linewidth=2, color='red')
        
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.set_title('DQN Agent Episode Rewards', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 第二个子图：奖励分布直方图
        ax2 = axes[1]
        ax2.hist(df['reward'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(df['reward'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {df["reward"].mean():.4f}')
        ax2.axvline(df['reward'].median(), color='green', linestyle='--',
                   linewidth=2, label=f'Median: {df["reward"].median():.4f}')
        ax2.set_xlabel('Reward', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Episode Reward Distribution', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_fig:
            output_path = Path(output_fig)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_fig, dpi=150, bbox_inches='tight')
            print(f"\n图表已保存到: {output_fig}")
        
        plt.show()
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
    except Exception as e:
        print(f"错误: {e}")


def export_to_json(csv_file, json_file):
    """
    将 CSV 转换为 JSON 格式
    
    Args:
        csv_file: 输入 CSV 文件
        json_file: 输出 JSON 文件
    """
    try:
        df = pd.read_csv(csv_file)
        data = {
            'episodes': df['episode'].tolist(),
            'rewards': df['reward'].tolist(),
            'statistics': {
                'min': float(df['reward'].min()),
                'max': float(df['reward'].max()),
                'mean': float(df['reward'].mean()),
                'std': float(df['reward'].std()),
                'total_episodes': len(df)
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"已导出到 JSON: {json_file}")
    except Exception as e:
        print(f"导出 JSON 失败: {e}")


def print_summary(csv_file):
    """
    打印摘要统计信息
    
    Args:
        csv_file: CSV 文件路径
    """
    try:
        df = pd.read_csv(csv_file)
        
        print("\n" + "="*50)
        print("Episode 奖励统计摘要")
        print("="*50)
        print(f"总 Episode 数: {len(df)}")
        print(f"最小奖励: {df['reward'].min():.4f}")
        print(f"最大奖励: {df['reward'].max():.4f}")
        print(f"平均奖励: {df['reward'].mean():.4f}")
        print(f"中位数奖励: {df['reward'].median():.4f}")
        print(f"标准差: {df['reward'].std():.4f}")
        
        # 计算不同百分位数
        percentiles = [10, 25, 50, 75, 90]
        print("\n百分位数:")
        for p in percentiles:
            val = df['reward'].quantile(p/100)
            print(f"  {p}%: {val:.4f}")
        
        # 计算正的奖励比例
        positive_ratio = (df['reward'] > 0).sum() / len(df)
        print(f"\n正奖励 Episode 比例: {positive_ratio*100:.2f}%")
        
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"统计失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='可视化 DQN Agent 的 Episode 奖励趋势'
    )
    parser.add_argument(
        '-f', '--file',
        type=str,
        default='episode_rewards.csv',
        help='输入 CSV 文件路径 (默认: episode_rewards.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='输出图表文件路径 (如: rewards.png)'
    )
    parser.add_argument(
        '-j', '--json',
        type=str,
        help='导出到 JSON 文件'
    )
    parser.add_argument(
        '-w', '--window',
        type=int,
        default=10,
        help='移动平均窗口大小 (默认: 10)'
    )
    parser.add_argument(
        '-s', '--summary',
        action='store_true',
        help='仅显示摘要统计信息'
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"错误: 找不到文件 {args.file}")
        print(f"请确保 C++ 程序已运行，并生成了 {args.file}")
        return
    
    # 打印摘要
    print_summary(args.file)
    
    if args.summary:
        return
    
    # 导出到 JSON (可选)
    if args.json:
        export_to_json(args.file, args.json)
    
    # 绘制图表
    plot_rewards(args.file, args.output, args.window)


if __name__ == '__main__':
    main()
