from typing import Tuple
from torch.utils.data import Dataset
import pathlib
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd 
import numpy
import string

class Cangjie_Class():
    CHARS = 'abcdefghijklmnopqrstuvwxyz_'
    CHAR2LABEL = {char: i for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, file_path):

        self.class_df = []

        self.char_list = string.ascii_letters[:26]
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

        self.class_df["cls"] = self.class_df.apply(lambda row: 0 if row["label"] == "zc" else 1, axis=1)

    def get_class_name_from_path(self, image_path):
        label = self.class_df.iloc[int(image_path.parent.stem)]["label"]
        label += '_' * (5 - len(label))
        return label
    
    def get_classes(self):
        classes = list(self.class_df["label"])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        idx_to_class = {i:cls_name for i, cls_name in enumerate(classes)}
        return classes, class_to_idx, idx_to_class
    
class Cangjie_Dataset(Dataset):
    
    def __init__(self, targ_dir: str, set:str, transform=None) -> None:
        self.cangjie = Cangjie_Class(pathlib.Path(targ_dir) / "952_labels.txt")
        self.paths = list((pathlib.Path(targ_dir)/ f"952_{set}").glob("*/*.png"))
        self.transform = transform
        # self.classes, self.class_to_idx, self.idx_to_class = self.cangjie.get_classes()

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)
    
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        image = self.load_image(index)

        if self.transform:
            image = self.transform(image)

        text = self.cangjie.get_class_name_from_path(self.paths[index])
        target = [self.cangjie.CHAR2LABEL[c] for c in text]
        cls = torch.Tensor([0]) if text == 'zc___' else torch.Tensor([1])
        target = torch.LongTensor(target)

        return image, cls, target
        
def cangjie_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths

def get_cangjie_training_dataloader(batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cangjie training dataset
        std: std of cangjie training dataset
        path: path to cangjie training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    cangjie_train = Cangjie_Dataset("etl_952_singlechar_size_64", "train", transform_train)
    cangjie_training_loader = DataLoader(cangjie_train,
                                         shuffle=shuffle, 
                                         num_workers=num_workers, 
                                         batch_size=batch_size)

    return cangjie_training_loader

def get_cangjie_val_dataloader(batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cangjie val dataset
        std: std of cangjie val dataset
        path: path to cangjie val python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cangjie_val_loader:torch dataloader object
    """
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])

    cangjie_val = Cangjie_Dataset("etl_952_singlechar_size_64", "val", transform_val)
    
    cangjie_val_loader = DataLoader(cangjie_val, 
                                    shuffle=shuffle, 
                                    num_workers=num_workers, 
                                    batch_size=batch_size)

    return cangjie_val_loader

def get_cangjie_test_dataloader(batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cangjie test dataset
        std: std of cangjie test dataset
        path: path to cangjie test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cangjie_test_loader:torch dataloader object
    """
    
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    cangjie_test = Cangjie_Dataset("etl_952_singlechar_size_64", "test", transform_test)
    
    cangjie_test_loader = DataLoader(cangjie_test,
                                     shuffle=shuffle, 
                                     num_workers=num_workers, 
                                     batch_size=batch_size)

    return cangjie_test_loader

