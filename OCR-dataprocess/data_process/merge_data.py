import os
import shutil
from pathlib import Path
from typing import List, Dict
import datetime

def count_images_in_directory(directory: str) -> int:
    """统计目录中的图片文件数量"""
    if not os.path.exists(directory):
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    count = 0
    
    for file in os.listdir(directory):
        if Path(file).suffix.lower() in image_extensions:
            count += 1
    
    return count


def merge_datasets(dataset_paths: List[str], output_path: str, 
                  images_subfolder: str = "images") -> Dict[str, int]:
    """
    合并多个数据集到输出目录
    
    Args:
        dataset_paths: 包含多个数据集路径的列表
        output_path: 输出目录路径
        images_subfolder: 图片子文件夹名称，默认为"images"
    
    Returns:
        包含各数据集统计信息的字典
    """
    # 创建输出目录
    output_images_dir = os.path.join(output_path, images_subfolder)
    os.makedirs(output_images_dir, exist_ok=True)
    
    stats = {}
    total_copied = 0
    total_skipped = 0
    
    print("=" * 60)
    print("开始合并数据集...")
    print("=" * 60)
    
    for i, dataset_path in enumerate(dataset_paths, 1):
        print(f"\n处理第 {i} 个数据集: {dataset_path}")
        print("-" * 40)
        
        # 检查数据集路径是否存在
        if not os.path.exists(dataset_path):
            print(f"⚠️  数据集路径不存在: {dataset_path}")
            stats[dataset_path] = {"原始数量": 0, "拷贝数量": 0, "跳过数量": 0}
            continue
        
        # 查找图片目录
        images_dir = os.path.join(dataset_path, images_subfolder)
        if not os.path.exists(images_dir):
            # 如果没有images子目录，则直接在数据集根目录查找图片
            images_dir = dataset_path
        
        # 统计原始数量
        original_count = count_images_in_directory(images_dir)
        copied_count = 0
        skipped_count = 0
        
        print(f"📁 图片目录: {images_dir}")
        print(f"📊 原始图片数量: {original_count}")
        
        if original_count == 0:
            print("⚠️  该目录中没有找到图片文件")
            stats[dataset_path] = {"原始数量": 0, "拷贝数量": 0, "跳过数量": 0}
            continue
        
        # 遍历并拷贝图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for filename in os.listdir(images_dir):
            if Path(filename).suffix.lower() in image_extensions:
                source_file = os.path.join(images_dir, filename)
                target_file = os.path.join(output_images_dir, filename)
                
                try:
                    # 如果目标文件已存在，直接覆盖
                    if os.path.exists(target_file):
                        print(f"🔄 覆盖文件: {filename}")
                        skipped_count += 1
                    else:
                        print(f"📋 拷贝文件: {filename}")
                    
                    shutil.copy2(source_file, target_file)
                    copied_count += 1
                    
                except Exception as e:
                    print(f"❌ 拷贝文件失败 {filename}: {e}")
                    continue
        
        stats[dataset_path] = {
            "原始数量": original_count,
            "拷贝数量": copied_count,
            "跳过数量": skipped_count
        }
        
        total_copied += copied_count
        total_skipped += skipped_count
        
        print(f"✅ 完成拷贝: {copied_count} 个文件")
        if skipped_count > 0:
            print(f"🔄 覆盖文件: {skipped_count} 个")
    
    # 统计最终结果
    final_count = count_images_in_directory(output_images_dir)
    
    print("\n" + "=" * 60)
    print("合并完成 - 统计报告")
    print("=" * 60)
    
    for i, (dataset_path, stat) in enumerate(stats.items(), 1):
        print(f"\n数据集 {i}: {os.path.basename(dataset_path)}")
        print(f"  📂 路径: {dataset_path}")
        print(f"  📊 原始数量: {stat['原始数量']:,} 个文件")
        print(f"  📋 拷贝数量: {stat['拷贝数量']:,} 个文件")
        if stat['跳过数量'] > 0:
            print(f"  🔄 覆盖数量: {stat['跳过数量']:,} 个文件")
    
    print(f"\n" + "=" * 60)
    print("📈 汇总统计:")
    print(f"  🗂️  处理数据集数量: {len(dataset_paths)} 个")
    print(f"  📊 总计原始文件: {sum(stat['原始数量'] for stat in stats.values()):,} 个")
    print(f"  📋 总计拷贝文件: {total_copied:,} 个")
    print(f"  🔄 总计覆盖文件: {total_skipped:,} 个")
    print(f"  📁 最终数据集大小: {final_count:,} 个文件")
    print(f"  📂 输出目录: {output_path}")
    print("=" * 60)
    
    return stats


def main():
    """主函数 - 示例用法"""
    # 示例配置
    dataset_list = [

        "/expdata/givap/data/plate_recong/stand/orgin_images",
        "/expdata/givap/data/plate_recong/stand/live/b1",
        "/expdata/givap/data/plate_recong/stand/live/b1_aug",
        # "/expdata/givap/data/plate_recong/unstand/b1b2",
        # "/expdata/givap/data/plate_recong/unstand/b1b2_aug", 
        # "/expdata/givap/data/plate_recong/mix/mix_b1",
        # "/expdata/givap/data/plate_recong/mix/mix_b1_aug",


    ]
    
    output_directory = f"/expdata/givap/data/plate_recong/merge/exp5_dataset12_{datetime.datetime.now().strftime('%Y_%m_%d')}"
    
    print("数据集合并工具")
    print(f"将要合并 {len(dataset_list)} 个数据集:")
    for i, path in enumerate(dataset_list, 1):
        print(f"  {i}. {path}")
    print(f"\n输出目录: {output_directory}")
    
    # 询问用户确认
    confirm = input("\n是否继续执行? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes', '是']:
        print("操作已取消")
        return
    
    # 执行合并
    try:
        stats = merge_datasets(dataset_list, output_directory)
        print("\n✅ 数据集合并完成!")
        
    except Exception as e:
        print(f"\n❌ 合并过程中出现错误: {e}")
        return


if __name__ == "__main__":
    main()
