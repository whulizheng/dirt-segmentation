import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
import torch
import numpy as np
import albumentations as A
import warnings
import time
import Utils
import os
from tqdm import tqdm


def device_prepare():
    if torch.cuda.is_available():
        print("Using GPU")
        return torch.device('cuda:0')

    else:
        print("Using CPU")
        return torch.device('cpu')


def define_augmentation(if_aug, width, height):
    if if_aug:
        return A.Compose([
            A.Resize(width=width, height=height, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04,
                               rotate_limit=0, p=0.25),
        ])
    else:
        return A.Compose([
            A.Resize(width=width, height=height, p=1.0),
        ])


def dataset_prepare(transform, data_type, data_path, batch_size, ratio=0.99):
    if data_type == 1:
        dataset = Utils.Dirt_data(
            data_path, transform)
        if ratio != 0:
            trainset, testset = random_split(dataset, [int(
                dataset.__len__()*ratio), int(dataset.__len__()-int(dataset.__len__()*ratio))])
            print("split "+str(len(trainset))+" imgs for training set and " +
                  str(len(testset)) + " for test")
            train_loader = torch.utils.data.DataLoader(
                dataset=trainset, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                dataset=testset, batch_size=batch_size)
            return train_loader, test_loader, dataset.num_classes, dataset.palette
        else:
            train_loader = None
            test_loader = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=batch_size)
            return train_loader, test_loader, dataset.num_classes, dataset
    if data_type == 2:
        dataset = Utils.Dirt_data_seg(
            data_path, transform)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=1)
        return data_loader,  dataset.num_classes, dataset.palette
    else:
        print("Wrong datatype")
        exit(-1)


def init_evaluation(name, num_classes):
    if name == "Dice":
        from Evaluations import Dice
        method = Dice.Dice(num_classes)
        return method
    else:
        print("Wrong Evaluation Name")
        exit(-1)


def init_models(name, num_classes, config):
    if name == "Unet":
        from Models import Unet
        model = Unet.Model((config["general"]["input_channels"],
                           config["general"]["width"], config["general"]["height"]), num_classes)
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = Unet.Loss(num_classes)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        return model, criterion, optimizer
    elif name == "FCN":
        from Models import FCN
        import torchvision
        pretrained_net = FCN.FeatureResNet()
        pretrained_net.load_state_dict(
            torchvision.models.resnet34(pretrained=True).state_dict())

        model = FCN.Model((config["general"]["input_channels"], config["general"]
                          ["width"], config["general"]["height"]), num_classes, pretrained_net)
        criterion = FCN.Loss(num_classes)
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        return model, criterion, optimizer
    else:
        print("Wrong Model Name")
        exit(-1)


def train(model, criterion, optimizer, device, train_loader, test_loader, epochs):
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start_time = time.time()
        running_train_loss = []
        for image, mask in train_loader:
            image = image.to(device, dtype=torch.float)
            mask = mask.to(device, dtype=torch.float)
            pred_mask = model.forward(image)  # forward propogation
            loss = criterion(pred_mask, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss.append(loss.item())
        else:
            running_test_loss = []
            with torch.no_grad():
                for image, mask in test_loader:
                    image = image.to(device, dtype=torch.float)
                    mask = mask.to(device, dtype=torch.float)
                    pred_mask = model.forward(image)
                    loss = criterion(pred_mask, mask)
                    running_test_loss.append(loss.item())

        epoch_train_loss = np.mean(running_train_loss)
        print('Train loss: {}'.format(epoch_train_loss))
        train_loss.append(epoch_train_loss)

        epoch_test_loss = np.mean(running_test_loss)
        print('Test loss: {}'.format(epoch_test_loss))
        test_loss.append(epoch_test_loss)

        time_elapsed = time.time() - start_time
        print('{:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    return model, train_loss, test_loss


def train_mode(config, device):
    Utils.show_config(config)
    batch_size = config["train"]["batch_size"]
    # augmentation
    transform = define_augmentation(
        config["train"]["augmentation"], config["general"]["width"], config["general"]["height"])
    if config["train"]["augmentation"]:
        print("Data Augmenting...")
    # prepare dataset
    for k in config["train"]["chosen_datasets"]:
        dataset_name = config["train"]["datasets"][k]
        data_type = config["train"]["data"][dataset_name]["data_type"]
        data_path = config["train"]["data"][dataset_name]["data_path"]
        train_loader, test_loader, num_classes, palette = dataset_prepare(
            transform, data_type, data_path, batch_size)

        for i in config["train"]["chosen_models"]:
            # init model
            model_name = config["train"]["models"][i]
            print("Training model: "+model_name + " On " + dataset_name)
            model, criterion, optimizer = init_models(
                model_name, num_classes, config)
            model.to(device)

            # train
            epochs = config["train"]["epochs"]
            train_loss, test_loss = [[], []]
            model, train_loss, test_loss = train(
                model, criterion, optimizer, device, train_loader, test_loader, epochs)

            # save_log
            if config["train"]["if_save_log"]:
                Utils.save_log(model_name, config, train_loss,
                               test_loss, dataset=dataset_name)
            # visualize_loss
            if config["train"]["if_visualize_loss"]:
                Utils.visualize_loss(train_loss=train_loss, test_loss=test_loss,
                                     model_name=model_name, dataset_name=dataset_name)
            # save_model
            if config["train"]["if_save_models"]:
                date = time.strftime('%Y_%m_%d_%H_%M_%S',
                                     time.localtime(time.time()))
                path = "Saved_models/" + model_name + "_"+dataset_name+"_" + date + ".pt"
                model.name = model_name + "_"+dataset_name+"_" + date
                torch.save(model, path)
                print("model "+model_name+" saved at: "+path)

            del model, criterion, optimizer


def eval_mode(config, device):
    # evaluation
    evaluations = {}
    transform = define_augmentation(
        False, config["general"]["width"], config["general"]["height"])
    for k in config["eval"]["chosen_datasets"]:
        dataset_name = config["eval"]["datasets"][k]
        data_type = config["eval"]["data"][dataset_name]["data_type"]
        data_path = config["eval"]["data"][dataset_name]["data_path"]
        _, test_loader, num_classes, palette = dataset_prepare(
            transform, data_type, data_path, 1, ratio=0)
        for m in config["eval"]["chosen_models"]:
            model_path = config["eval"]["models"][m]
            model = None
            if model_path == "latest_trained_model":
                model = Utils.load_recent_model()
            else:
                model = Utils.model_load(model_path)
            model_name = model.name
            for e in config["eval"]["chosen_methods"]:
                method_name = config["eval"]["methods"][e]
                print("Evaluating Model: "+model_name+" By " +
                      method_name + " On " + dataset_name)
                method = init_evaluation(method_name, num_classes)
                scores = []
                with torch.no_grad():
                    flag = 0
                    for image, mask in test_loader:
                        image = image.to(device, dtype=torch.float)
                        mask = mask.to(device, dtype=torch.float)
                        pred_mask = model.forward(image)
                        score = method(pred_mask, mask)
                        scores.append(score.item())
                        pred_mask = pred_mask.cpu().detach().numpy()
                        mask = mask.cpu().detach().numpy()
                        image = image.cpu().detach().numpy()
                        Utils.save_outputs(
                            image, mask, pred_mask, model_name, dataset_name, flag, palette)
                        flag += 1
                evaluations[method_name] = np.mean(scores)
                print("Model: "+model_name+" Got "+method_name +
                      ": "+str(float(np.mean(scores))))


def log_mode(config):
    print("Log mode")
    print("Loading recent log")
    log = Utils.load_recent_log()
    model_name = log["model"]
    try:
        dataset_name = log["dataset"]
    except:
        dataset_name = "null"
    train_loss = log["train_loss"]
    test_loss = log["test_loss"]
    if config["log"]["if_visualize_loss"]:
        print("visualizing loss values in tmp folder")
        Utils.visualize_loss(train_loss=train_loss, test_loss=test_loss,
                             model_name=model_name, dataset_name=dataset_name)


def seg_mode(config, device):
    print("Seg mode")
    # building
    transform = define_augmentation(
        False, config["general"]["width"], config["general"]["height"])
    output_dir = config["seg"]["output_dir"]
    for k in config["seg"]["chosen_datasets"]:
        dataset_name = config["seg"]["datasets"][k]
        data_type = config["seg"]["data"][dataset_name]["data_type"]
        data_path = config["seg"]["data"][dataset_name]["data_path"]
        data_loader, num_classes, palette = dataset_prepare(
            transform, data_type, data_path, 1, ratio=0)
        for m in config["seg"]["chosen_models"]:
            model_path = config["seg"]["models"][m]
            model = None
            if model_path == "latest_trained_model":
                model = Utils.load_recent_model()
            else:
                model = Utils.model_load(model_path)
            model_name = model.name
            output_path = output_dir + "Segmente "+dataset_name+" using "+model_name + "/"
            if os.path.exists(output_path):
                pass
            else:
                os.makedirs(output_path)
            print("Segmente "+dataset_name+" using "+model_name)
            for image, file_name in tqdm(data_loader):
                pred_mask = model.forward(image)
                pred_mask = pred_mask.cpu().detach().numpy()[0]
                cv2.imwrite(output_path+file_name[0]+"_"+model_name+"_"+dataset_name+"_.png",
                            Utils.onehot_to_mask(pred_mask.transpose((1, 2, 0)), palette, colors=True))


def main():
    config = Utils.load_json("config.json")
    device = device_prepare()
    if config["general"]["modes"][config["general"]["chosen_mode"]] == "Train":
        train_mode(config, device)
    elif config["general"]["modes"][config["general"]["chosen_mode"]] == "Eval":
        eval_mode(config, device)
    elif config["general"]["modes"][config["general"]["chosen_mode"]] == "Log":
        log_mode(config)
    elif config["general"]["modes"][config["general"]["chosen_mode"]] == "Seg":
        seg_mode(config, device)
    else:
        print("Mode error, pls check config file.")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
