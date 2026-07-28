import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class RoadImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file: train.csv 或 val.csv 的路徑
        root_dir: data/raw/ 的路徑
        transform: PyTorch 的資料前處理 (torchvision.transforms)
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # 作業二元標籤規則已固定：Bad -> 0, Good -> 1
        self.label_map = {"Bad": 0, "Good": 1}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 組合圖片的完整路徑 (對應 data/raw/<檔名>)
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image_path'].split('/')[-1])
        
        # 讀取圖片並確保轉換為 RGB 色彩空間
        image = Image.open(img_name).convert('RGB')
        
        # 讀取字串標籤並轉換為數字 0 或 1
        str_label = self.data_frame.iloc[idx]['human_label']
        label = self.label_map[str_label]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
