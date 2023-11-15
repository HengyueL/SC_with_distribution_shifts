import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
import argparse, random
import seaborn as sns
sns.set()
from PIL import Image
from torchvision.datasets import ImageFolder
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from selectivenet.model import SelectiveNet
from selectivenet.resnet_variant import SelectiveNetRes
from selectivenet.vgg_variant import vgg16_variant
import torchvision
import torch.nn as nn
# from main_orig import validate

COLORS = list(mcolors.TABLEAU_COLORS)

MEAN, STD = (0.485, 0.456, 0.406), (0.485, 0.456, 0.406)
TRAINFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])
TESTFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


MEAN_CIFAR, MEAN_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
tform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN_CIFAR, MEAN_STD)
])

def get_loader_c(
    data_path, 
    corr_type,
    corr_level,
    batch_size=512
):
    # === Below are full ImageNet-C with partial sampling ===
    data_path = os.path.join(
        data_path, corr_type, str(corr_level)
    )
    val_transforms = TESTFORM
    dataset = torchvision.datasets.ImageNet(
        data_path,
        split="val",
        transform=val_transforms
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=False
    )
    return val_loader

def get_loader_o(data_path, batch_size=256):
    val_transforms = TESTFORM
    num_workers = 4
    dataset = ImageFolder(
        root=data_path, transform=val_transforms
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return val_loader


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """flist format: impath label\nimpath label\n."""
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            data = line.strip().rsplit(maxsplit=1)
            if len(data) == 2:
                impath, imlabel = data, 
            else:
                impath, imlabel = data[0], -10
            imlist.append((impath, int(imlabel)))

    return imlist


class OpenImageDataset(Dataset):
    def __init__(
        self,
        root,
        flist,
        transform=None,
        target_transform=None,
        flist_reader=default_flist_reader,
        loader=default_loader
    ):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


def get_loader_openimage_o(data_path, text_path):
    val_transforms = TESTFORM
    batch_size = 256
    num_workers = 4
    dataset_path = data_path
    annot_txt_path = text_path
    dataset = OpenImageDataset(
        dataset_path,
        annot_txt_path,
        transform=val_transforms
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return val_loader


def get_loader_clean(
    data_path, 
    batch_size=512,
):
    val_transforms = TESTFORM
    dataset_val = torchvision.datasets.ImageNet(
        data_path,
        split="val",
        transform=val_transforms
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, 
        shuffle=False
    )
    return val_loader
        

def set_random_seeds(seed=21):
    """
        This function sets all random seed used in this experiment.
        For reproduce purpose.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class CifarCDataset(Dataset):
    def __init__(self, root_dir, corr_type,
                 corr_level,
                 transform=None):
        super(CifarCDataset).__init__()
        corr_data_file = os.path.join(root_dir, corr_type.lower() + ".npy")
        label_file = os.path.join(root_dir, "labels.npy")

        start_idx = (corr_level-1)*10000
        end_idx = (corr_level)*10000

        self.data = np.load(corr_data_file)[start_idx:end_idx, :, :, :]
        self.data = self.data / 255.
        self.labels = np.load(label_file)[start_idx:end_idx]
        self.transfrom = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx, :, :, :]
        label = self.labels[idx].astype(int)

        if self.transfrom is not None:
            data = self.transfrom(data)

        return data, label


def main(args):
    # === Construct Cifar 10 Dataset ===
    if args.dataset == 'imagenet':
        data_path = args.data_root_imagenet
        test_loader = get_loader_clean(
            data_path
        )
    elif args.dataset == 'imagenet-c':
        data_path = args.data_root_imagenet_c
        test_loader = get_loader_c(data_path, args.corr_type, args.corr_level)
    elif args.dataset == "imagenet-o":
        data_path = args.data_root_imagenet_o
        test_loader = get_loader_o(
            data_path
        )
    elif args.dataset == "openimage-o":
        data_path = args.data_root_openimage_o
        txt_path = "./openimage_o.txt"
        test_loader = get_loader_openimage_o(
            data_path, txt_path
        )
    elif args.dataset == 'cifar10':
        data_path = args.data_root_cifar10
        test_set = CIFAR10(
            root=data_path, train=False, transform=tform_cifar, download=False
        )
        test_loader = DataLoader(
            test_set, batch_size=256, shuffle=False, num_workers=8
        )
    elif args.dataset == 'cifar10-c':
        print("Cifar-10-c")
        data_path = args.data_root_cifar10_c
        test_set = CifarCDataset(
            root_dir=data_path, corr_type=args.corr_type,
            corr_level=args.corr_level, transform=tform_cifar
        )    
        test_loader = DataLoader(
            test_set, batch_size=256, shuffle=False, num_workers=8
        )
    elif args.dataset == "cifar100":
        print("CIFAR100")
        data_path = args.data_root_cifar100
        test_set = CIFAR100(
            root=data_path, train=False, transform=tform_cifar, download=True
        )
        test_loader = DataLoader(
            test_set, batch_size=256, shuffle=False, num_workers=8
        )
    else:
        raise RuntimeError("Undefined dataset.")
    
    dataset_str = args.dataset
    if dataset_str == "cifar10-c":
        corr_str = args.corr_type
        level_str = "%d" % args.corr_level
        name_str = "%s_%s_%s" % (dataset_str, corr_str, level_str) 
        save_data_root = os.path.join("..", "log_folder", "CIFAR", name_str)
    elif dataset_str == "cifar10":
        save_data_root = os.path.join("..", "log_folder", "CIFAR", dataset_str)
    elif dataset_str == "cifar100":
        save_data_root = os.path.join("..", "log_folder", "CIFAR", dataset_str)
    elif dataset_str == "imagenet":
        save_data_root = os.path.join("..", "log_folder", "ImageNet", dataset_str)
    elif dataset_str == "imagenet-c":
        corr_str = args.corr_type
        level_str = "%d" % args.corr_level
        name_str = "%s_%s_%s" % (dataset_str, corr_str, level_str) 
        save_data_root = os.path.join("..", "log_folder", "ScNet", "CombineOOD", name_str)
    elif dataset_str == "imagenet-o":
        save_data_root = os.path.join("..", "log_folder", "ScNet", "CombineOOD", dataset_str)
    elif dataset_str == "openimage-o":
        save_data_root = os.path.join("..", "log_folder", "ScNet", "CombineOOD", dataset_str)
    else:
        raise RuntimeError("UNsupported Dataset.")
    os.makedirs(save_data_root, exist_ok=True)

    # === Load Pretraiend Model ===
    if "cifar" in args.dataset:
        features = vgg16_variant(32, 0).cuda()
        model = SelectiveNet(features, 512, 10).cuda()
        # === Compute Laster Layer weight norm ===
        last_layer = model.classifier[-1]
        weights = last_layer.weight.data.clone().cpu().numpy()
        bias = last_layer.bias.data.clone().cpu().numpy()
        print(
            "Weight and bias: ", 
            weights.shape, 
            bias.shape
        )

        if torch.cuda.device_count() > 1: 
            model = torch.nn.DataParallel(model)
        model.cuda()

        ckp_dir = args.checkpoint_dir
        checkpoint = torch.load(ckp_dir)
        model.load_state_dict(checkpoint)
    else:
        model = SelectiveNetRes(512, 1000)
        # === Compute Laster Layer weight norm ===
        last_layer = model.classifier[-1]
        weights = last_layer.weight.data.clone().cpu().numpy()
        bias = last_layer.bias.data.clone().cpu().numpy()
        print(
            "Weight and bias: ", 
            weights.shape, 
            bias.shape
        )
        if torch.cuda.device_count() > 1: 
            model = torch.nn.DataParallel(model)
        model.cuda()

        ckp_dir = args.checkpoint_dir
        checkpoint = torch.load(ckp_dir)
        model.load_state_dict(checkpoint["model"])
    model.eval()


    save_weight_name = os.path.join(
        save_data_root, "last_layer_weights.npy"
    )
    save_bias_name = os.path.join(
        save_data_root, "last_layer_bias.npy"
    )

    # === Loop and get labels and pred_logits ===
    logits_log = []
    label_log = []
    features_list_log = []
    selection_score_list = []
    aux_logits_log = []
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda(non_blocking=True).float()
            target = target.cuda(non_blocking=True).float()
            # compute output
            logit_output, selector_output, aux_output = model(input)

            features = model.features(input).view(input.size(0), -1).cpu().numpy()
            
            # save
            logits = logit_output.cpu().numpy()
            aux_logits = aux_output.cpu().numpy()
            if args.dataset == "cifar100":
                labels = -10 * torch.ones_like(target).cpu().numpy()
            else:
                labels = target.cpu().numpy()
            selection_score = selector_output.cpu().numpy()

            logits_log.append(logits)
            aux_logits_log.append(aux_logits)
            label_log.append(labels)
            features_list_log.append(features)
            selection_score_list.append(selection_score)

    save_logits_name = os.path.join(save_data_root, "pred_logits.npy")
    np.save(save_logits_name, np.concatenate(logits_log, axis=0))
    save_aux_logits_name = os.path.join(save_data_root, "aux_logits.npy")
    np.save(save_aux_logits_name, np.concatenate(aux_logits_log, axis=0))
    save_labels_name = os.path.join(save_data_root, "labels.npy")
    np.save(save_labels_name, np.concatenate(label_log, axis=0))
    save_features_name = os.path.join(save_data_root, "features.npy")
    np.save(save_features_name, np.concatenate(features_list_log, axis=0))
    save_sc_score_name = os.path.join(save_data_root, "training_based_score.npy")
    np.save(save_sc_score_name, np.concatenate(selection_score_list, axis=0))

    np.save(save_weight_name, weights)
    np.save(save_bias_name, bias)

    print(
        "Final shape Check: ", 
        np.concatenate(logits_log, axis=0).shape,
        np.concatenate(aux_logits_log, axis=0).shape,
        np.concatenate(label_log, axis=0).shape,
        np.concatenate(features_list_log, axis=0).shape,
        np.concatenate(selection_score_list, axis=0).shape
    )

    if "-c" not in args.dataset and "100" not in args.dataset and "-o" not in args.dataset:
        logits_np = np.concatenate(logits_log, axis=0)
        aux_logits_np = np.concatenate(aux_logits_log, axis=0)
        labels_np = np.concatenate(label_log, axis=0)
        acc = np.mean(np.argmax(logits_np, axis=1) == labels_np) * 100
        aux_acc = np.mean(np.argmax(aux_logits_np, axis=1) == labels_np) * 100
        print("%s Acc: %.04f" % (args.dataset, acc))
        print("Aux Logit Acc: %.04f" % aux_acc)


if __name__ == "__main__":
    print("Collect self-adaptive-training CIFAR models prediction logits and weight norm of the last linear layer.")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", dest="dataset", type=str,
        default="cifar10",
        help="The dataset used to test the SC performance. [cifar10,  imagenet, cifar10-c, imagenet-c]"
    )

    # === ImageNet Experiment Related Dataset Path ===
    parser.add_argument(
        "--data_root_imagenet", dest="data_root_imagenet", type=str,
        default="",
        help="Path to ImageNet Dataset."
    )
    parser.add_argument(
        "--data_root_imagenet_c", dest="data_root_imagenet_c", type=str,
        default="",
        help="Path to ImageNet-C Dataset."
    )
    parser.add_argument(
        "--data_root_imagenet_o", dest="data_root_imagenet_o", type=str,
        default="",
        help="Path to ImageNet-O Dataset."
    )
    parser.add_argument(
        "--data_root_openimage_o", dest="data_root_openimage_o", type=str,
        default="",
        help="Path to OpenImage-O Dataset."
    )

    # === CIFAR Experiment Related Dataset Path ===
    parser.add_argument(
        "--data_root_cifar10", dest="data_root_cifar10", type=str,
        default="",
        help="Path to CIFAR-10 Dataset."
    )
    parser.add_argument(
        "--data_root_cifar10_c", dest="data_root_cifar10_c", type=str,
        default="",
        help="Path to CIFAR-10 Dataset."
    )
    parser.add_argument(
        "--data_root_cifar100", dest="data_root_cifar100", type=str,
        default="",
        help="Path to CIFAR-10 Dataset."
    )
    
    # === Location of the pretrained model weights ===
    parser.add_argument(
        "--checkpoint_dir", dest="checkpoint_dir",
        default="",
        help="Path to the pretrained weights."
    )
    
    args = parser.parse_args()

    if "imagenet" in args.dataset:
        if args.dataset == "imagenet-c":
            root_dir = args.data_root_imagenet_c  # root dir to imagenet_c dataset
            corr_type_list = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
            corr_level_list = [3]  # corr_level_list = list(range(1, 6, 1)) if you want to test for all images
            for corr_type in corr_type_list:
                for corr_level in corr_level_list:
                    args.corr_type = corr_type
                    args.corr_level = corr_level
                    main(args)
        args.dataset = "imagenet"
        main(args)
        args.dataset = "imagenet-o"
        main(args)
        args.dataset = "openimage-o"
        main(args)
    elif "cifar" in args.dataset:
        if args.dataset == "cifar10-c":
            root_dir = args.data_root_cifar10_c
            corr_type_list = [f.split(".npy")[0] for f in os.listdir(root_dir) if "labels" not in f]
            # corr_level_list = list(range(1, 6, 1))
            corr_level_list = [3]
            for corr_type in corr_type_list:
                for corr_level in corr_level_list:
                    args.corr_type = corr_type
                    args.corr_level = corr_level
                    main(args)
        args.dataset = "cifar100"
        main(args)
        args.dataset = "cifar10"
        main(args)

    print("Completed.")