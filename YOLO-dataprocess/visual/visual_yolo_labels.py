import os
import cv2
import random
import argparse
from pathlib import Path

def visualize_yolo_dataset(image_dir, label_dir, class_names=None, output_dir=None):
    """
    Visualize YOLO format dataset with bounding boxes and class labels
    
    Args:
        image_dir (str): Path to directory containing images
        label_dir (str): Path to directory containing YOLO format label files
        class_names (list, optional): List of class names. Defaults to None (uses numeric class IDs)
        output_dir (str, optional): Directory to save visualized images. Defaults to None (only display)
    """
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"All visualization results will be saved to: {output_dir}")
    
    # Visualization parameters
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
              for _ in range(len(class_names) if class_names else 100)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    for img_file in image_files:
        # Read image
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Read corresponding label file
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            continue
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Draw bounding boxes
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            box_width = float(parts[3]) * img_width
            box_height = float(parts[4]) * img_height
            
            # Convert from center coordinates to corner coordinates
            x_min = int(x_center - box_width / 2)
            y_min = int(y_center - box_height / 2)
            x_max = int(x_center + box_width / 2)
            y_max = int(y_center + box_height / 2)
            
            # Clip coordinates to image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width - 1, x_max)
            y_max = min(img_height - 1, y_max)
            
            # Get color and label
            color = colors[class_id % len(colors)]
            label = class_names[class_id] if class_names else str(class_id)
            
            # Draw rectangle and label
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
            
            # Display label above bounding box
            label_size = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(img, 
                         (x_min, y_min - label_size[0][1] - 5),
                         (x_min + label_size[0][0], y_min),
                         color, -1)
            cv2.putText(img, label, 
                       (x_min, y_min - 5), 
                       font, font_scale, (255, 255, 255), thickness)
        
        # Save the image
        output_path = os.path.join(output_dir, f"vis_{img_file}")
        cv2.imwrite(output_path, img)
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO format dataset")
    parser.add_argument("--image_dir", type=str, default=r"D:\Data\YOLO可训练数据集\人脸检测\人脸检测\images\train", 
                        help="Directory containing images")
    parser.add_argument("--label_dir", type=str, default=r"D:\Data\YOLO可训练数据集\人脸检测\人脸检测\labels\train", 
                            help="Directory containing YOLO format label files")
    # classes = "pedestrian,people,bicycle,car,van,truck,tricycle,awning-tricycle,bus,motor"
    classes = "face"
    parser.add_argument("--class_names", type=str, default=classes, 
                        help="Comma-separated list of class names (e.g., 'person,car,dog')")
    
    args = parser.parse_args()
    
    # Automatically create output directory in the same parent directory as image_dir
    output_dir = os.path.join(os.path.dirname(args.image_dir), "visual_yolo_labels")
    
    # Parse class names if provided
    class_names = args.class_names.split(',') if args.class_names else None
    
    visualize_yolo_dataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        class_names=class_names,
        output_dir=output_dir
    )