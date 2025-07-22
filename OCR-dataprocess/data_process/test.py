import cv2
import numpy as np
import time
import requests
from urllib.parse import urlparse

def pull_stream_opencv(stream_url, timeout=30, max_retries=3):
    """
    使用OpenCV拉取视频流（改进版）
    
    Args:
        stream_url (str): 视频流地址
        timeout (int): 连接超时时间（秒）
        max_retries (int): 最大重试次数
    """
    retry_count = 0
    
    while retry_count < max_retries:
        print(f"尝试连接视频流 (第 {retry_count + 1} 次)")
        
        # 创建VideoCapture对象并设置参数
        cap = cv2.VideoCapture()
        
        # 设置网络超时（毫秒）
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区大小
        cap.set(cv2.CAP_PROP_FPS, 25)  # 设置期望帧率
        
        # 尝试打开流
        success = cap.open(stream_url)
        
        if not success:
            print(f"错误：无法打开视频流 (尝试 {retry_count + 1}/{max_retries})")
            cap.release()
            retry_count += 1
            time.sleep(2)  # 等待2秒后重试
            continue
        
        print("成功连接到视频流")
        print(f"视频宽度: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"视频高度: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"帧率: {cap.get(cv2.CAP_PROP_FPS)}")
        
        frame_count = 0
        last_frame_time = time.time()
        
        try:
            while True:
                # 读取帧
                ret, frame = cap.read()
                
                if not ret:
                    print("无法读取帧，检查网络连接...")
                    current_time = time.time()
                    if current_time - last_frame_time > timeout:
                        print("读取超时，尝试重连...")
                        break
                    continue
                
                last_frame_time = time.time()
                frame_count += 1
                
                # 显示帧
                cv2.imshow('Video Stream', frame)
                
                # 每100帧显示一次统计信息
                if frame_count % 100 == 0:
                    print(f"已处理 {frame_count} 帧")
                
                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("用户退出")
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
                    
        except KeyboardInterrupt:
            print("用户中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        retry_count += 1
        if retry_count < max_retries:
            print(f"将在2秒后重试...")
            time.sleep(2)
    
    print(f"经过 {max_retries} 次尝试后仍无法稳定连接")
    return False

def pull_stream_with_ffmpeg(stream_url, output_file=None):
    """
    使用ffmpeg命令行工具拉取视频流
    
    Args:
        stream_url (str): 视频流地址
        output_file (str): 可选，保存视频文件路径
    """
    import subprocess
    import sys
    
    try:
        if output_file:
            # 保存到文件
            cmd = [
                'ffmpeg',
                '-i', stream_url,
                '-c', 'copy',
                '-t', '60',  # 录制60秒
                output_file,
                '-y'  # 覆盖输出文件
            ]
            print(f"开始录制视频流到: {output_file}")
        else:
            # 直接播放
            cmd = [
                'ffplay',
                '-i', stream_url,
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-framedrop'
            ]
            print("开始播放视频流...")
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            print("FFmpeg处理完成")
            return True
        else:
            print(f"FFmpeg错误: {process.stderr}")
            return False
            
    except FileNotFoundError:
        print("错误：未找到ffmpeg，请确保已安装ffmpeg并添加到PATH")
        return False
    except Exception as e:
        print(f"FFmpeg处理出错: {e}")
        return False

def check_stream_info(stream_url):
    """
    检查视频流信息
    
    Args:
        stream_url (str): 视频流地址
    """
    print("检查视频流信息...")
    
    # 检查URL格式
    parsed_url = urlparse(stream_url)
    print(f"协议: {parsed_url.scheme}")
    print(f"主机: {parsed_url.netloc}")
    print(f"路径: {parsed_url.path}")
    
    # 尝试HTTP HEAD请求
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.head(stream_url, headers=headers, timeout=10)
        print(f"HTTP状态码: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
        print(f"Content-Length: {response.headers.get('Content-Length', 'Unknown')}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"HTTP请求失败: {e}")
        return False

def pull_stream_with_requests(stream_url, save_path="stream_output.flv"):
    """
    使用requests直接下载流数据
    
    Args:
        stream_url (str): 视频流地址
        save_path (str): 保存路径
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Connection': 'keep-alive'
        }
        
        print("开始下载视频流...")
        response = requests.get(stream_url, headers=headers, stream=True, timeout=30)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (1024*1024) == 0:  # 每MB显示一次
                            print(f"已下载: {downloaded // (1024*1024)} MB")
                            
                        # 可以设置最大下载大小限制
                        if downloaded > 100 * 1024 * 1024:  # 100MB限制
                            print("达到下载限制，停止下载")
                            break
            
            print(f"视频流已保存到: {save_path}")
            return True
        else:
            print(f"HTTP错误: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"下载失败: {e}")
        return False

# 使用示例
if __name__ == "__main__":
    # 你的流地址
    stream_url = "https://rtmp01open.ys7.com:9188/v3/openlive/BF2192438_1_1.flv?expire=1782628855&id=859080931457290240&t=1dff6d731b59094e4008425f058c113aca28d4f94e9af463ffab38d310c34e83&ev=100"
    
    print("=== 视频流拉取工具 ===")
    print("选择拉取方式:")
    print("1. OpenCV (改进版)")
    print("2. FFmpeg 播放")
    print("3. FFmpeg 录制")
    print("4. 检查流信息")
    print("5. 直接下载流文件")
    
    choice = input("请输入选择 (1-5): ")
    
    if choice == "1":
        pull_stream_opencv(stream_url, timeout=30, max_retries=3)
    elif choice == "2":
        pull_stream_with_ffmpeg(stream_url)
    elif choice == "3":
        output_file = input("输入保存文件名 (默认: output.flv): ") or "output.flv"
        pull_stream_with_ffmpeg(stream_url, output_file)
    elif choice == "4":
        check_stream_info(stream_url)
    elif choice == "5":
        save_path = input("输入保存路径 (默认: stream_output.flv): ") or "stream_output.flv"
        pull_stream_with_requests(stream_url, save_path)
    else:
        print("默认使用OpenCV方式")
        pull_stream_opencv(stream_url)