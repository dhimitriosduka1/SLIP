# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
import json
import os
import pickle
import zipfile

import numpy as np
from PIL import Image, ImageFile

import torch
from torchvision import datasets as t_datasets

import utils

import ast
import sys
import webdataset as wds
import torchvision.transforms as T
import math
import random
from multiprocessing import Value
import braceexpand
from dataclasses import dataclass
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    url_opener,
    tar_file_expander,
    valid_sample,
)
from torch.utils.data import (
    Dataset,
    DataLoader,
    SubsetRandomSampler,
    IterableDataset,
    get_worker_info,
)
from torch.utils.data.distributed import DistributedSampler
import logging

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def yfcc_loader(root, index):
    index = format(index, "0>8d")
    repo = index[:2]
    z = index[2:5]
    file_img = index[5:] + ".jpg"
    path_zip = os.path.join(root, "images", repo, z) + ".zip"
    with zipfile.ZipFile(path_zip, "r") as myzip:
        img = Image.open(myzip.open(file_img))
    return img.convert("RGB")


class ImageCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata):
        self.dataset = dataset
        self.root = root
        if self.dataset == "yfcc15m":
            with open(metadata, "rb") as f:
                self.samples = pickle.load(f)
        elif self.dataset == "coco":
            samples = defaultdict(list)
            with open(metadata) as f:
                annotations = json.load(f)["annotations"]
            for ann in annotations:
                samples[ann["image_id"]].append(ann["caption"])
            self.samples = [(k, v) for k, v in samples.items()]
        elif self.dataset == "cc12m" or self.dataset == "cc3m":
            self.samples = np.load(metadata, allow_pickle=True)
        elif self.dataset == "redcaps":
            with open(metadata) as f:
                annotations = json.load(f)
            self.samples = [
                (ann["image_id"], ann["subreddit"], ann["caption"])
                for ann in annotations
            ]

    def get_raw_item(self, i):
        if self.dataset == "yfcc15m":
            index, title, desc = self.samples[i]
            caption = np.random.choice([title, desc])
            img = yfcc_loader(self.root, index)
        elif self.dataset == "coco":
            index, captions = self.samples[i]
            path = os.path.join(self.root, "train2017", "{:012d}.jpg".format(index))
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == "cc3m":
            ann = self.samples[i]
            filename, captions = ann["image_id"], ann["captions"]
            path = os.path.join(self.root, str(filename))
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == "cc12m":
            ann = self.samples[i]
            filename, captions = ann["image_name"], ann["captions"]
            path = os.path.join(self.root, filename)
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == "redcaps":
            image_id, subreddit, caption = self.samples[i]
            path = os.path.join(self.root, subreddit, f"{image_id}.jpg")
            img = pil_loader(path)

        return img, caption

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class ImageCaptionDatasetCLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None, tokenizer=None):
        super().__init__(dataset, root, metadata)

        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            image = self.transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return image, caption


class ImageCaptionDatasetSLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform, augment, tokenizer=None):
        super().__init__(dataset, root, metadata)

        self.transform = transform
        self.augment = augment
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        image = self.transform(img)
        aug1 = self.augment(img)
        aug2 = self.augment(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return image, caption, aug1, aug2


class ImageCaptionDatasetSSL(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, augment):
        super().__init__(dataset, root, metadata)

        self.augment = augment

    def __getitem__(self, i):
        img, _ = self.get_raw_item(i)

        aug1 = self.augment(img)
        aug2 = self.augment(img)

        return aug1, aug2


class FileListDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(images)
        self.labels = np.load(labels)

    def __getitem__(self, index):
        img = pil_loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


def get_downstream_dataset(catalog, name, is_train, transform):
    entry = catalog[name]
    root = entry["path"]
    if entry["type"] == "imagefolder":
        dataset = t_datasets.ImageFolder(
            os.path.join(root, entry["train"] if is_train else entry["test"]),
            transform=transform,
        )
    elif entry["type"] == "special":
        if name == "cifar10":
            dataset = t_datasets.CIFAR10(
                root, train=is_train, transform=transform, download=True
            )
        elif name == "cifar100":
            dataset = t_datasets.CIFAR100(
                root, train=is_train, transform=transform, download=True
            )
        elif name == "stl10":
            dataset = t_datasets.STL10(
                root,
                split="train" if is_train else "test",
                transform=transform,
                download=True,
            )
        elif name == "mnist":
            dataset = t_datasets.MNIST(
                root, train=is_train, transform=transform, download=True
            )
    elif entry["type"] == "filelist":
        path = entry["train"] if is_train else entry["test"]
        val_images = os.path.join(root, path + "_images.npy")
        val_labels = os.path.join(root, path + "_labels.npy")
        if name == "clevr_counts":
            target_transform = lambda x: [
                "count_10",
                "count_3",
                "count_4",
                "count_5",
                "count_6",
                "count_7",
                "count_8",
                "count_9",
            ].index(x)
        else:
            target_transform = None
        dataset = FileListDataset(val_images, val_labels, transform, target_transform)
    else:
        raise Exception("Unknown dataset")

    return dataset


def get_dataset(train_transform, tokenizer, args):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augment = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.08, 1.0)),
            T.RandomApply(
                [T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([utils.GaussianBlur([0.1, 2.0])], p=0.5),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ]
    )

    if args.model.startswith("SIMCLR"):
        return ImageCaptionDatasetSSL(args.dataset, args.root, args.metadata, augment)
    elif args.model.startswith("CLIP"):
        return get_wds_dataset(
            args=args, preprocess_img=train_transform, augment=augment,is_train=True, tokenizer=tokenizer
        )
        # return ImageCaptionDatasetCLIP(
        #     args.dataset, args.root, args.metadata, train_transform, tokenizer
        # )
    elif args.model.startswith("SLIP"):
        return get_wds_dataset(
            args=args, preprocess_img=train_transform, is_train=True, tokenizer=tokenizer
        )
        # return ImageCaptionDatasetSLIP(
        #     args.dataset, args.root, args.metadata, train_transform, augment, tokenizer
        # )

# ================================= STUFF FOR WEBDATASET ===================================

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def with_additional_augmentations(
    preprocess_img, augment, tokenizer, rephrased_captions
):
    def map_fn(sample):
        out = {}

        key = f"{sample['__key__']}.jpg"

        image = sample["image"]
        out["image1"] = preprocess_img(image)
        out["image2"] = augment(image)
        out["image3"] = augment(image)

        text = sample["text"]
        random_text = random.sample(rephrased_captions[key]["paraphrases"], 1)[0]

        out["text1"] = tokenizer(text)[0]
        out["text2"] = tokenizer(random_text)[0]

        return out

    return map_fn


def get_wds_dataset(
    args, preprocess_img, augment, is_train, epoch=0, floor=False, tokenizer=None
):
    # =============== Needed for TeMo =======================
    _REPHRASED_CAPTIONS_PATH = (
        "/ptmp/dduka/databases/cc12m/paraphrases/cc12m_all_paraphrased.pkl"
    )

    # Load _REPHRASED_CAPTIONS_PATH
    with open(_REPHRASED_CAPTIONS_PATH, "rb") as f:
        REPHRASED_CAPTIONS = pickle.load(f)
    # =============== Needed for TeMo =======================

    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    "Currently, the number of dataset samples must be specified for the training dataset. "
                    "Please specify it via `--train-num-samples` if no dataset length info is present."
                )
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(
        epoch=epoch
    )  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert (
            resampled
        ), "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."

    if resampled:
        pipeline = [
            ResampledShards2(
                input_shards,
                weights=args.train_data_upsampling_factors,
                deterministic=True,
                epoch=shared_epoch,
            )
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend(
                [
                    detshuffle2(
                        bufsize=_SHARD_SHUFFLE_SIZE,
                        initial=_SHARD_SHUFFLE_INITIAL,
                        seed=args.seed,
                        epoch=shared_epoch,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                ]
            )
        pipeline.extend(
            [
                # at this point, we have an iterator over the shards assigned to each worker at each node
                tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                ),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(handler=log_and_continue),
            ]
        )
    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            # wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
            wds.map(
                with_additional_augmentations(
                    preprocess_img,
                    augment,
                    tokenizer,
                    REPHRASED_CAPTIONS,
                )
            ),
            # wds.to_tuple("image", "text"),
            wds.to_tuple("image1", "text1", "image2", "text2", "image3"),
            wds.batched(args.batch_size, partial=not is_train),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert (
                num_shards >= args.workers * args.world_size
            ), "number of shards must be >= total workers"
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(
            num_batches / num_workers
        )  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(
            num_worker_batches
        )  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, "r").read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split("::")
        assert len(weights) == len(
            urllist
        ), f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

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


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(
                self.weights
            ), f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(
                    url=self.rng.choices(self.urls, weights=self.weights, k=1)[0]
                )


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample
