import os
import json
from sys import exc_info
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
import time
from os import listdir
import matplotlib.pyplot as plt
import pylab as pl


def load_json(file_path):
    with open(file_path, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict


class Dirt_data(Dataset):
    def __init__(self, path, transforms=None):
        '''
        63 = type_1
        -87 = type_2
        -118 = type_3
        '''
        self.palette = [[63], [-87], [-118]]
        self.num_classes = 3
        self.path = path
        self.transform = transforms
        self.images = []
        self.masks = []
        for img in listdir(self.path):
            if img[-5:] == "2.tif":
                self.masks.append(os.path.join(self.path, img))
            elif img[-5:] == "0.tif":
                self.images.append(os.path.join(self.path, img))
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        mask = cv2.imread(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY).astype(np.int8)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        mask = np.expand_dims(mask, axis=-1)
        mask = mask_to_onehot(mask, self.palette)
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return (image, mask)

    def __len__(self):
        return len(self.images)

class Dirt_data_seg(Dataset):
    def __init__(self, path, transforms=None):
        '''
        63 = type_1
        -87 = type_2
        -118 = type_3
        '''
        self.palette = [[63], [-87], [-118]]
        self.num_classes = 3
        self.path = path
        self.transform = transforms
        self.images = []
        for img in listdir(self.path):
            self.images.append(os.path.join(self.path, img))

    def __getitem__(self, idx):
        image = self.images[idx]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=image)
            image = aug['image']
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)
        basename = os.path.basename(self.images[idx])
        file_name = os.path.splitext(basename)[0]
        return image, file_name

    def __len__(self):
        return len(self.images)

def save_log(name, config, train_loss, test_loss, dataset, path="Logs/", evaluations=None):
    date = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    source = {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'model': name,
        'batch_size': config["train"]["batch_size"],
        'epochs': config["train"]["epochs"],
        'if_augmentation': config["train"]["augmentation"],
        'dataset': dataset,
        'date': date
    }
    if evaluations:
        source.update(evaluations)

    np.save(path+name+"_"+date+".log.npy", source)
    print("log saved at: "+path+date+".log.npy")


def show_config(config):
    if config["general"]["modes"][config["general"]["chosen_mode"]] == "Train":
        print("Going to train these models:")
        for i in config["train"]["chosen_models"]:
            model_name = config["train"]["models"][i]
            print("\t"+model_name)
        print("On datasets:")
        for i in config["train"]["chosen_datasets"]:
            dataset_name = config["train"]["datasets"][i]
            print("\t"+dataset_name)
    elif config["general"]["modes"][config["general"]["chosen_mode"]] == "Eval":
        print("Going to evaluate these models:")
        for i in config["eval"]["chosen_models"]:
            model_name = config["eval"]["models"][i]
            print("\t"+model_name)
        print("On datasets:")
        for i in config["eval"]["chosen_datasets"]:
            dataset_name = config["eval"]["datasets"][i]
            print("\t"+dataset_name)
        print("By methods:")
        for i in config["eval"]["chosen_methods"]:
            method = config["eval"]["methods"][i]
            print("\t"+method)
    else:
        print("Going to evaluate these models:")


def save_outputs(images, masks, pred_masks, model_name,dataset_name, flag, palette, path="tmp/"):
    for i in range(len(images)):
        image = images[i]
        mask = masks[i]
        pred_mask = pred_masks[i]
        cv2.imwrite(path+model_name+"_"+dataset_name+"_"+str(flag)+"_"+str(i) +
                    "_image.jpg", image.transpose((1, 2, 0)))
        cv2.imwrite(path+model_name+"_"+dataset_name+"_"+str(flag)+"_"+str(i)+"_mask.jpg",
                    onehot_to_mask(mask.transpose((1, 2, 0)), palette, colors=True))
        cv2.imwrite(path+model_name+"_"+dataset_name+"_"+str(flag)+"_"+str(i)+"_pred_mask.jpg",
                    onehot_to_mask(pred_mask.transpose((1, 2, 0)), palette, colors=True))


def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map



def onehot_to_mask(mask, palette, colors=False):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    color_list = [[74, 24, 231], [24, 227, 255], [206, 130, 0]]
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.int32(colour_codes[x.astype(np.int32)])

    p = np.zeros((x.shape[0],x.shape[1],3),np.float32)
    if colors and len(color_list) >= len(palette):
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                p[i][j] = np.array(color_list[palette.index(x[i][j])])
        x = p
    return x

def visualize_loss(train_loss, test_loss, model_name="",dataset_name="",path="tmp/"):
    plt.figure(figsize = (7,5))
    pl.plot(train_loss,'-',label=u'Train')
    pl.plot(test_loss,'-', label = u'Test')
    pl.legend(["Train","Test"])
    pl.xlabel(u'epoches')
    pl.ylabel(u'loss')
    plt.title(model_name + '\'loss values in training')
    plt.savefig(path+model_name+"_"+dataset_name+"_loss.jpg")
    plt.close()


def load_log(path):
    log = np.load(path,allow_pickle=True)
    return log.item()

def load_recent_log(root_path = "Logs/"):
    logs = os.listdir(root_path)
    if not logs:
        print("no log")
        exit(0)
    else:
        logs = sorted(logs,key=lambda x: os.path.getmtime(os.path.join(root_path, x)))
    log_path = logs[-1]
    log = load_log(root_path+log_path)
    return log

def load_recent_model(root_path = "Saved_models/"):
    models = os.listdir(root_path)
    if not models:
        print("no model")
        exit(0)
    else:
        models = sorted(models,key=lambda x: os.path.getmtime(os.path.join(root_path, x)))
    model_path = models[-1]
    model = model_load(root_path+model_path)
    return model

def model_load(path):
    model = torch.load(path)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    return model


