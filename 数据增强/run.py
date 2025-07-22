#!/usr/bin/env python3
"""
YOLO数据增强工具启动脚本
检查依赖并启动应用程序
"""

import sys
import os
import subprocess
import importlib.util

def check_package(package_name):
    """检查包是否已安装"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_requirements():
    """安装依赖包"""
    try:
        print("正在安装依赖包...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("依赖包安装完成！")
        return True
    except subprocess.CalledProcessError:
        print("依赖包安装失败！")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("YOLO数据增强工具启动器")
    print("=" * 50)
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("错误: 需要Python 3.7或更高版本")
        print(f"当前版本: {sys.version}")
        sys.exit(1)
    
    print(f"Python版本: {sys.version}")
    
    # 检查必要的包
    required_packages = [
        "PyQt5",
        "albumentations", 
        "cv2",
        "numpy",
        "PIL"
    ]
    
    missing_packages = []
    for package in required_packages:
        if not check_package(package):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少以下依赖包: {', '.join(missing_packages)}")
        
        if os.path.exists("requirements.txt"):
            response = input("是否自动安装依赖包? (y/n): ")
            if response.lower() == 'y':
                if install_requirements():
                    print("依赖包安装成功，正在启动应用...")
                else:
                    print("请手动安装依赖包: pip install -r requirements.txt")
                    sys.exit(1)
            else:
                print("请手动安装依赖包: pip install -r requirements.txt")
                sys.exit(1)
        else:
            print("requirements.txt文件不存在！")
            sys.exit(1)
    else:
        print("所有依赖包已安装 ✓")
    
    # 检查主程序文件
    if not os.path.exists("yolo_augmenter.py"):
        print("错误: 找不到主程序文件 yolo_augmenter.py")
        sys.exit(1)
    
    print("正在启动YOLO数据增强工具...")
    print("-" * 50)
    
    # 启动主程序
    try:
        from yolo_augmenter import main as run_augmenter
        run_augmenter()
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请检查是否所有依赖包都已正确安装")
        sys.exit(1)
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 