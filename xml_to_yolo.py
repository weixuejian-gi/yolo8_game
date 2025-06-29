import os
import xml.etree.ElementTree as ET

def convert_xml_to_yolo(xml_dir, output_dir, class_list):
    os.makedirs(output_dir, exist_ok=True)
    
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue
            
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()
        
        img_width = int(root.find('size/width').text)
        img_height = int(root.find('size/height').text)
        
        txt_filename = os.path.splitext(xml_file)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                if cls_name not in class_list:
                    continue
                
                cls_id = class_list.index(cls_name)
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Convert to YOLO format (normalized center-x, center-y, width, height)
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# 使用方法
convert_xml_to_yolo(
    xml_dir='data/labels/train',  # 输入XML文件夹
    output_dir='data/labels/train',    # 输出TXT文件夹
    class_list=['person']  # 类别列表
)