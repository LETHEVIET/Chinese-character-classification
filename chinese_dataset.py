from typing import Tuple
from torch.utils.data import Dataset
import pathlib
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd 
import numpy

CANGJIE_DATASET_MEAN = 0.20013867
CANGJIE_DATASET_STD = 0.38466406

class Chinese_Class():
    def __init__(self, file_path):
        self.class_df = []
        class_dict = {
            "id": [],
            "char": [],
            "hex": [],
            "uni": [],
            "label": [],
        }
        with open(file_path, "r") as f:
            f.readline()
            for line in f:
                id, char, hex, uni, label = line.split()
                class_dict["id"].append(int(id))
                class_dict["char"].append(char)
                class_dict["hex"].append(hex)
                class_dict["uni"].append(uni)
                class_dict["label"].append(label)

        self.class_df = pd.DataFrame(class_dict)

    def get_class_name_from_path(self, image_path):
        return self.class_df.iloc[int(image_path.parent.stem)]["char"]
    
    def get_classes(self):
        classes = list(self.class_df["char"])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        idx_to_class = {i:cls_name for i, cls_name in enumerate(classes)}
        return classes, class_to_idx, idx_to_class
    
class Chinese_Dataset(Dataset):
    
    def __init__(self, targ_dir: str, set:str, transform=None) -> None:
        self.chinese = Chinese_Class(pathlib.Path(targ_dir) / "952_labels.txt")
        self.paths = list((pathlib.Path(targ_dir)/ f"952_{set}").glob("*/*.png"))
        self.transform = transform
        self.classes, self.class_to_idx, self.idx_to_class = self.chinese.get_classes()

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)
    
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.chinese.get_class_name_from_path(self.paths[index])
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

def get_chinese_training_dataloader(batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of chinese training dataset
        std: std of chinese training dataset
        path: path to chinese training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    chinese_train = Chinese_Dataset("etl_952_singlechar_size_64", "train", transform_train)
    chinese_training_loader = DataLoader(
        chinese_train, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return chinese_training_loader

def get_chinese_val_dataloader(batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of chinese val dataset
        std: std of chinese val dataset
        path: path to chinese val python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: chinese_val_loader:torch dataloader object
    """
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])

    chinese_val = Chinese_Dataset("etl_952_singlechar_size_64", "val", transform_val)
    
    chinese_val_loader = DataLoader(
        chinese_val, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return chinese_val_loader

def get_chinese_test_dataloader(batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of chinese test dataset
        std: std of chinese test dataset
        path: path to chinese test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: chinese_test_loader:torch dataloader object
    """
    
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    chinese_test = Chinese_Dataset("etl_952_singlechar_size_64", "test", transform_test)
    
    chinese_test_loader = DataLoader(
        chinese_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return chinese_test_loader

