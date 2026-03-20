import os
import sys
import re
import numpy as np
import pandas as pd
import PIL.Image
from dataclasses import dataclass
from multiprocessing import Value

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)
sys.path.append(os.path.dirname(SCRIPT_DIR))


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


class MakeupStyleDataset(Dataset):
    def __init__(self, data_root, anno_path, transform, tokenizer=None, **kwargs):
        self.data_root = data_root
        self.anno_path = anno_path
        self.transform = transform
        self.tokenizer = tokenizer
        self.dataset_size = kwargs.get("dataset_size", None)

        samples = self.make_dataset(self.anno_path)
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.classes = list(set(self.targets))

    def make_dataset(self, anno_path):
        samples = []

        df = pd.read_csv(anno_path, encoding="utf-8")
        for index, row in df.iterrows():
            img_file, target, cap = row["file"], row["class"], row["caption"]
            cap_label = row["class"]

            if "/" in img_file:
                img_folder = img_file.split("/")[0]
                img_path = os.path.join(self.data_root, img_file)
            else:
                img_folder = img_file.split("-")[0]
                img_path = os.path.join(self.data_root, img_folder, img_file)

            if os.path.exists(img_path):
                samples.append([img_path, int(target), cap, int(cap_label)])

        # start from 0
        target_list = [s[1] for s in samples]
        target_offset = np.array(target_list).min()
        cap_label_list = [s[3] for s in samples]
        cap_label_offset = np.array(cap_label_list).min()
        for s in samples:
            s[1] -= target_offset
            s[3] -= cap_label_offset

        return samples

    def __getitem__(self, index):
        img_path, target, text, text_label = self.samples[index]

        sample = PIL.Image.open(img_path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        # match = re.search(r"(The makeup is[^.]*\.)", text)
        match = re.search(r"(The makeup is.*?[.?!])", text, re.DOTALL)
        text = "photography of a person with makeup. {}".format(match.group(1))

        text_input = self.tokenizer(text)
        text_input = {k: v.squeeze(dim=0) for k, v in text_input.items()}

        return sample, target, text_input, text_label, index

    def __len__(self):
        if self.dataset_size is not None:
            return len(self.samples[:self.dataset_size])
        return len(self.samples)


def get_style_dataset(args, data_root, anno_path, preprocess_fn, is_train, tokenizer=None):
    dataset = MakeupStyleDataset(
        data_root,
        anno_path,
        preprocess_fn,
        tokenizer=tokenizer,
        dataset_size=args.train_num_samples
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    print("num samples/batches {}/{} at {}".format(num_samples, dataloader.num_batches, data_root))

    return DataInfo(dataloader, sampler)


def get_dataset_fn(dataset_type):
    if dataset_type == "style":
        return get_style_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args.dataset_type)(
            args, args.train_data, args.train_anno, preprocess_train, is_train=True, tokenizer=tokenizer)

    if args.val_data:
        data["query"] = get_dataset_fn(args.dataset_type)(
            args, args.val_data, args.val_anno, preprocess_val, is_train=False, tokenizer=tokenizer)
        data["base"] = get_dataset_fn(args.dataset_type)(
            args, args.memory_data, args.memory_anno, preprocess_val, is_train=False, tokenizer=tokenizer)

    return data


if __name__ == '__main__':
    import torchvision
    from style_clip.augment import get_clip_transform, ContrastiveTransformations, OPENAI_DATASET_MEAN, \
        OPENAI_DATASET_STD
    from style_clip.model import CustomTokenizer
    from utils.misc import denormalize_batch
    from utils.vis_utils import show_face_result

    data_root = "./output/makeup_pair/makeup"
    anno_path = "./output/makeup_pair/makeup_pair_ffhq_kontext-train.csv"

    # preprocess = get_clip_transform()
    preprocess = ContrastiveTransformations(size=224, lambda_c=0)
    tokenizer = CustomTokenizer("vit_large")

    # img = PIL.Image.open("./assets/images/00128-003-Smoky_Seductress-00.png")
    # sample = preprocess(img)
    # img1 = denormalize_batch(sample[0].unsqueeze(dim=0), OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
    # img1 = torchvision.transforms.functional.to_pil_image(img1[0])
    # img1.save("img1.png")

    dataset = MakeupStyleDataset(
        data_root,
        anno_path,
        preprocess,
        tokenizer=tokenizer,
        dataset_size=10
    )

    print("dataset size", len(dataset))
    print(dataset[0])

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
                                         drop_last=False)

    for i, (images, target, text_input, text_label, index) in enumerate(loader):
        img1 = denormalize_batch(images[0], OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
        img2 = denormalize_batch(images[1], OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)

        img1 = torchvision.transforms.functional.to_pil_image(img1[0])
        img2 = torchvision.transforms.functional.to_pil_image(img2[0])

        img1.save("img1.png")
        img2.save("img1.png")
