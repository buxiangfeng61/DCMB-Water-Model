#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_控制函数.py

功能：批量调用 "04图润色mamba-GCN（调用ssm+真实值与预测值）.py" 模型脚本，
依次读取 1~17 号站点文件夹中的数据，运行模型，并将每次运行结果保存到对应的结果文件夹中。
模型脚本会在当前工作目录下读取 `data（训练集）.csv` 和 `data（验证集）.csv`，
并生成 `real_vs_predicted.csv`、`评价指标_XX.csv`，
以及 `true_vs_predicted_line_plot_XX.png`、`true_vs_predicted_scatter_plot_XX.png` 等输出文件。
本脚本通过切换工作目录后执行模型，然后将所有生成的输出文件移动到统一的结果目录中。
"""

import os
import subprocess
import shutil


def main():
    # 模型脚本绝对路径
    model_script = os.path.abspath(
        "Architecture of Space Experiment Model.py"
    )
    # 站点编号范围
    site_numbers = range(1, 18)

    for num in site_numbers:
        input_dir = f"{num}号站点"
        output_dir = f"{num}号站点结果"

        # 检查输入目录是否存在
        if not os.path.isdir(input_dir):
            print(f"输入目录不存在：{input_dir}，跳过")
            continue

        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        print(f"正在处理站点 {num}：输入 {input_dir} -> 输出 {output_dir}")

        try:
            # 切换到输入目录作为工作目录，执行模型脚本
            subprocess.run(
                ["python", model_script],
                cwd=input_dir,
                check=True
            )

            # 将生成的输出文件从输入目录移动到输出目录
            for fname in os.listdir(input_dir):
                if fname.endswith(".csv") or fname.endswith(".png"):
                    src = os.path.join(input_dir, fname)
                    dst = os.path.join(output_dir, fname)
                    shutil.move(src, dst)
            print(f"站点 {num} 处理完成，结果已保存到 {output_dir}")

        except subprocess.CalledProcessError as e:
            print(f"站点 {num} 处理失败：{e}")

    print("所有站点处理完成。")


if __name__ == "__main__":
    main()
