from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi
import argparse
from torch import nn, optim
from torch.autograd import Variable, grad
from scipy import linalg as la
from transformers import CLIPProcessor, CLIPModel
import math
import torchvision.transforms as tvt
import os
import matplotlib.pyplot as plt
import pandas as pd
import wget
import zipfile
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as tfms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.utils import make_grid
from torchvision import utils
import random
from tqdm import trange
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
import open_clip

torch.set_num_threads(5)
torch.set_num_interop_threads(5)


def seed_everything(seed):
    """
    Changes the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ConfounderDataset_train(Dataset):
    def __init__(self, root_dir,
                 target_name, confounder_names,
                 model_type=None, augment_data=None):
        raise NotImplementedError

    def __len__(self):
        return len(self.training_sample)

    def __getitem__(self, idx):
        y = self.training_sample_y_array[idx]
        a = self.training_sample_confounder_array[idx]
        img_filename = os.path.join(
            './datasets',
            'waterbirds',
            self.training_sample[idx])
        img = preprocess(Image.open(img_filename))
        img_for_res = self.train_transform(Image.open(img_filename))
        return img, y, a, img_for_res


class CUBDataset_train(ConfounderDataset_train):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """

    def __init__(self):
        self.data_dir = os.path.join(
            './datasets/',
            'waterbirds')

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values

        self.training_sample = self.filename_array[self.split_array == 0]
        self.training_sample_y_array = self.y_array[self.split_array == 0]
        self.training_sample_confounder_array = self.confounder_array[self.split_array == 0]
        self.train_transform = get_transform_cub(train=True)
        self.eval_transform = get_transform_cub(train=False)


class ConfounderDataset_test(Dataset):
    def __init__(self, root_dir,
                 target_name, confounder_names,
                 model_type=None, augment_data=None):
        raise NotImplementedError

    def __len__(self):
        return len(self.test_sample)

    def __getitem__(self, idx):
        y = self.test_sample_y_array[idx]
        a = self.test_sample_confounder_array[idx]
        img_filename = os.path.join(
            './datasets',
            'waterbirds',
            self.test_sample[idx])
        img = preprocess(Image.open(img_filename))
        img_for_res = self.eval_transform(Image.open(img_filename))

        return img, y, a, img_for_res


class CUBDataset_test(ConfounderDataset_test):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """

    def __init__(self):
        self.data_dir = os.path.join(
            './datasets',
            'waterbirds')

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values

        self.test_sample = self.filename_array[self.split_array == 2]
        self.test_sample_y_array = self.y_array[self.split_array == 2]
        self.test_sample_confounder_array = self.confounder_array[self.split_array == 2]
        self.eval_transform = get_transform_cub(train=False)


def get_transform_cub(train):
    transform = tfms.Compose([
        tfms.Resize((336, 336)),
        tfms.ToTensor()
    ])
    return transform


def inference_a_test(vlm, spu_v0, spu_v1, test_data_loader):
    correct_00, total_00 = 0, 0
    correct_01, total_01 = 0, 0
    correct_10, total_10 = 0, 0
    correct_11, total_11 = 0, 0

    for step, (test_input, test_target, sensitive, _) in enumerate(tqdm(test_data_loader, desc="Testing")):
        with torch.no_grad():
            test_target = test_target.to(device)
            sensitive = sensitive.to(device)
            test_input = test_input.to(device)
            z = vlm.encode_image(test_input)
            infered_a = inference_a(vlm, landbg, waterbg, z)

            mask_00 = ((test_target == 0) & (sensitive == 0))
            mask_01 = ((test_target == 0) & (sensitive == 1))
            mask_10 = ((test_target == 1) & (sensitive == 0))
            mask_11 = ((test_target == 1) & (sensitive == 1))

            correct_00 += (infered_a[mask_00] == sensitive[mask_00]).float().sum().item()
            total_00 += mask_00.float().sum().item()

            correct_01 += (infered_a[mask_01] == sensitive[mask_01]).float().sum().item()
            total_01 += mask_01.float().sum().item()

            correct_10 += (infered_a[mask_10] == sensitive[mask_10]).float().sum().item()
            total_10 += mask_10.float().sum().item()

            correct_11 += (infered_a[mask_11] == sensitive[mask_11]).float().sum().item()
            total_11 += mask_11.float().sum().item()
    acc_00 = correct_00 / total_00
    acc_01 = correct_01 / total_01
    acc_10 = correct_10 / total_10
    acc_11 = correct_11 / total_11

    print(f'Accuracy for y=0, s=0: {acc_00}')
    print(f'Accuracy for y=0, s=1: {acc_01}')
    print(f'Accuracy for y=1, s=0: {acc_10}')
    print(f'Accuracy for y=1, s=1: {acc_11}')


def inference_a(vlm, spu_v0, spu_v1, z):
    text_embeddings = torch.cat((spu_v0, spu_v1), dim=0)
    norm_img_embeddings = z
    norm_text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    cosine_similarity = torch.mm(norm_img_embeddings, norm_text_embeddings.t())
    logits_per_image = cosine_similarity
    probs = logits_per_image.softmax(dim=1)
    _, predic = torch.max(probs.data, 1)
    return predic


def supervised_inference_a(img, device):
    resnet18 = models.resnet18(pretrained=False)
    num_classes = 2
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
    res_model = resnet18
    res_model.load_state_dict(torch.load('res_net.pth'))
    res_model = res_model.to(device)
    res_model.eval()
    img = img.to(device)
    test_pred_ = res_model(img)
    _, predic = torch.max(test_pred_.data, 1)
    return predic


def compute_scale(vlm, spu_v0, spu_v1, training_data_loader, device, a, partial_a, landbg, waterbg):
    vlm = vlm.to(device)
    scale_0 = []
    scale_1 = []
    spu0 = spu_v0 / spu_v0.norm(dim=1, keepdim=True)
    spu1 = spu_v1 / spu_v1.norm(dim=1, keepdim=True)

    for step, (test_input, _, sensitive, img) in enumerate(tqdm(training_data_loader, desc="Computing Scale")):
        with torch.no_grad():
            test_input = test_input.to(device)
            z = vlm.encode_image(test_input)
            if a == True:
                sensitive = sensitive
            else:
                if partial_a == False:
                    sensitive = inference_a(vlm, landbg, waterbg, z)
                elif partial_a == True:
                    sensitive = supervised_inference_a(img, device)

            mask_0 = sensitive == 0
            mask_0 = mask_0.to(device)
            h = z[mask_0]
            inner_land = torch.mm(h / h.norm(dim=1, keepdim=True), spu0.t())
            scale_0.extend(inner_land.detach().cpu().numpy())

            mask_1 = sensitive == 1
            mask_1 = mask_1.to(device)
            g = z[mask_1]
            inner_water = torch.mm(g / g.norm(dim=1, keepdim=True), spu1.t())
            scale_1.extend(inner_water.detach().cpu().numpy())
    scale_0 = np.array(scale_0)
    scale_1 = np.array(scale_1)
    print(np.mean(scale_0))
    print(np.mean(scale_1))
    return torch.tensor(np.mean(scale_0)), torch.tensor(np.mean(scale_1))


def run_epoch(vlm, dataloader, device, a, partial_a, landbg, waterbg, tokenizer, training_data_loader):
    scale_0, scale_1 = compute_scale(model, landbg, waterbg, training_data_loader, device, a, partial_a, landbg,
                                     waterbg)

    texts_label = ["a photo of a landbird.", "a photo of a waterbird."]
    text_label_tokened = tokenizer(texts_label).to(device)

    vlm = vlm.to(device)
    vlm.eval()
    test_pred = []
    test_gt = []
    sense_gt = []
    female_predic = []
    female_gt = []
    male_predic = []
    male_gt = []
    correct_00, total_00 = 0, 0
    correct_01, total_01 = 0, 0
    correct_10, total_10 = 0, 0
    correct_11, total_11 = 0, 0
    cos = nn.CosineSimilarity(dim=0)
    feature_a0 = []
    feature_a1 = []

    for step, (test_input, test_target, sensitive_real, img) in enumerate(tqdm(dataloader, desc="Zero Shot Testing")):
        with torch.no_grad():
            gt = test_target.detach().cpu().numpy()
            sen = sensitive_real.detach().cpu().numpy()
            test_gt.extend(gt)
            sense_gt.extend(sen)
            test_input = test_input.to(device)

            text_label_tokened
            z = vlm.encode_image(test_input)
            z = z / z.norm(dim=1, keepdim=True)

            if a == True:
                sensitive = sensitive_real
            if a == False:
                if partial_a == False:
                    sensitive = inference_a(vlm, landbg, waterbg, z)
                    sensitive = torch.tensor(sensitive)
                elif partial_a == True:
                    sensitive = supervised_inference_a(img, device)

            mask_0 = sensitive == 0
            mask_0 = mask_0.to(device)
            z[mask_0] -= scale_0 * landbg / landbg.norm(dim=1, keepdim=True)

            mask_1 = sensitive == 1
            mask_1 = mask_1.to(device)
            z[mask_1] -= scale_1 * waterbg / waterbg.norm(dim=1, keepdim=True)

            feature_a0.extend(z[mask_0].detach().cpu().numpy())
            feature_a1.extend(z[mask_1].detach().cpu().numpy())

            text_embeddings = vlm.encode_text(text_label_tokened)
            img_embeddings = z
            norm_img_embeddings = img_embeddings / img_embeddings.norm(dim=1, keepdim=True)
            norm_text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
            cosine_similarity = torch.mm(norm_img_embeddings, norm_text_embeddings.t())

            logits_per_image = cosine_similarity
            probs = logits_per_image.softmax(dim=1)
            _, predic = torch.max(probs.data, 1)
            predic = predic.detach().cpu()
            test_pred.extend(predic.numpy())
            label = test_target.squeeze().detach().cpu()
            mask_00 = ((label == 0) & (sensitive_real == 0))
            mask_01 = ((label == 0) & (sensitive_real == 1))
            mask_10 = ((label == 1) & (sensitive_real == 0))
            mask_11 = ((label == 1) & (sensitive_real == 1))

            correct_00 += (predic[mask_00] == label[mask_00]).float().sum().item()
            total_00 += mask_00.float().sum().item()

            correct_01 += (predic[mask_01] == label[mask_01]).float().sum().item()
            total_01 += mask_01.float().sum().item()

            correct_10 += (predic[mask_10] == label[mask_10]).float().sum().item()
            total_10 += mask_10.float().sum().item()

            correct_11 += (predic[mask_11] == label[mask_11]).float().sum().item()
            total_11 += mask_11.float().sum().item()
    acc_00 = correct_00 / total_00
    acc_01 = correct_01 / total_01
    acc_10 = correct_10 / total_10
    acc_11 = correct_11 / total_11

    print(f'Accuracy for y=0, s=0: {acc_00}')
    print(f'Accuracy for y=0, s=1: {acc_01}')
    print(f'Accuracy for y=1, s=0: {acc_10}')
    print(f'Accuracy for y=1, s=1: {acc_11}')

    feature_a0 = np.array(feature_a0)
    feature_a1 = np.array(feature_a1)
    a0_tensor = torch.from_numpy(np.mean(feature_a0, 0))
    a1_tensor = torch.from_numpy(np.mean(feature_a1, 0))

    for i in range(len(sense_gt)):
        if sense_gt[i] == 0:
            female_predic.append(test_pred[i])
            female_gt.append(test_gt[i])
        else:
            male_predic.append(test_pred[i])
            male_gt.append(test_gt[i])
    female_CM = confusion_matrix(female_gt, female_predic)
    male_CM = confusion_matrix(male_gt, male_predic)
    female_dp = (female_CM[1][1] + female_CM[0][1]) / (
            female_CM[0][0] + female_CM[0][1] + female_CM[1][0] + female_CM[1][1])
    male_dp = (male_CM[1][1] + male_CM[0][1]) / (male_CM[0][0] + male_CM[0][1] + male_CM[1][0] + male_CM[1][1])
    female_TPR = female_CM[1][1] / (female_CM[1][1] + female_CM[1][0])
    male_TPR = male_CM[1][1] / (male_CM[1][1] + male_CM[1][0])
    female_FPR = female_CM[0][1] / (female_CM[0][1] + female_CM[0][0])
    male_FPR = male_CM[0][1] / (male_CM[0][1] + male_CM[0][0])
    acc = accuracy_score(test_gt, test_pred)
    # print('Female TPR', female_TPR)
    # print('male TPR', male_TPR)
    # print('DP', abs(female_dp - male_dp))
    # print('EOP', abs(female_TPR - male_TPR))
    # print('EoD', 0.5*(abs(female_FPR-male_FPR)+ abs(female_TPR-male_TPR)))
    # print('acc', accuracy_score(test_gt, test_pred))


if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 256

    # 初始化模型
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained='laion2b_s32b_b82k')
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    # 加载数据集
    training_dataset = CUBDataset_train()
    test_dataset = CUBDataset_test()

    training_data_loader = torch.utils.data.DataLoader(dataset=training_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=0,
                                                       drop_last=True)

    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   drop_last=False)

    # 设置文本特征
    texts = ["a photo with a water background", "a photo with a land background"]
    text = tokenizer(texts).to(device)
    text_features = model.encode_text(text)
    waterbg = text_features[0].unsqueeze(0)
    landbg = text_features[1].unsqueeze(0)

    # 配置参数
    a = True  # If True -> TIE, If False -> TIE*
    partial_a = False

    # 运行测试
    print("Starting evaluation...")
    run_epoch(model, test_data_loader, device, a, partial_a, landbg, waterbg, tokenizer, training_data_loader)
