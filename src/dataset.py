import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET

class CustomObjectDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.xml_files = [file for file in os.listdir(os.path.join(root_dir, 'annotations/xmls')) if file.endswith('.xml')]

    def __len__(self):
        return len(self.xml_files)

    def __getitem__(self, idx):
        xml_file = os.path.join(self.root_dir, 'annotations/xmls', self.xml_files[idx])
        img_name = os.path.splitext(self.xml_files[idx])[0] + '.jpg'
        img_path = os.path.join(self.root_dir, 'images', img_name)

        # Load image
        img = Image.open(img_path).convert("RGB").resize((224,224))

        # Load and parse XML annotation
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract image size
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)

        # Initialize lists for target data
        boxes = []
        labels = []

        # Extract bounding box information
        for obj in root.findall('object'):
            label = obj.find('name').text
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)

            # Append bounding box coordinates and label
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor([labels.index(label) for label in labels], dtype=torch.int64)

        # Calculate area (optional)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Define iscrowd (optional)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transform:
            img, target = self.transform(img, target)

        return img, target

if __name__ == "__main__":
    pass
