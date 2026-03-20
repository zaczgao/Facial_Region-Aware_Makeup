#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist

# Root directory of the project
try:
    abspath = os.path.abspath(__file__)
except NameError:
    abspath = os.getcwd()
SCRIPT_DIR = os.path.dirname(abspath)

import style_clip.clip_utils as utils
from style_clip.distributed import world_info_from_env, is_dist_avail_and_initialized


# https://github.com/hirl-team/HCSC
@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, use_fp16=False, eval_embed='backbone'):
    local_rank, global_rank, world_size = world_info_from_env()

    model.eval()

    features = None
    for images, _, _, _, index in data_loader:
        images = images.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        with torch.amp.autocast(device_type='cuda', enabled=use_fp16):
            image_emb, content_emb, style_emb, text_features = model(images, None, output_dict=False)

            if eval_embed == 'backbone':
                feats = image_emb.clone()
            elif eval_embed == 'head':
                feats = style_emb.clone()

        # init storage feature matrix
        if global_rank == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1], dtype=feats.dtype)
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(world_size, index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        if is_dist_avail_and_initialized():
            y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
            y_all_reduce.wait()
        else:
            y_l = [index]
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            world_size,
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        if is_dist_avail_and_initialized():
            output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
            output_all_reduce.wait()
        else:
            output_l = [feats]

        # update storage feature matrix
        if global_rank == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


# @torch.no_grad()
# def extract_features_bkp(model, data_loader, use_cuda=True, use_fp16=False, eval_embed='backbone'):
#     local_rank, global_rank, world_size = world_info_from_env()
#
#     model.eval()
#
#     local_feats_list = []
#     local_idx_list = []
#     for images, _, _, _, index in data_loader:
#         images = images.cuda(non_blocking=True)
#         index = index.cuda(non_blocking=True)
#
#         with torch.amp.autocast(device_type='cuda', enabled=use_fp16):
#             image_emb, content_emb, style_emb, text_features = model(images, None, output_dict=False)
#
#             if eval_embed == 'backbone':
#                 feats = image_emb
#             elif eval_embed == 'head':
#                 feats = style_emb
#
#         local_feats_list.append(feats)
#         local_idx_list.append(index)
#
#     local_feats = torch.cat(local_feats_list)
#     local_idx = torch.cat(local_idx_list)
#
#     # init storage feature matrix
#     features = None
#     if global_rank == 0:
#         features = torch.zeros(len(data_loader.dataset), local_feats.shape[-1], dtype=local_feats.dtype)
#         if use_cuda:
#             features = features.cuda(non_blocking=True)
#         print(f"Storing features into tensor of shape {features.shape}")
#
#     # get indexes from all processes
#     if is_dist_avail_and_initialized():
#         y_l = [torch.empty_like(local_idx) for _ in range(world_size)]
#         torch.distributed.all_gather(y_l, local_idx)
#     else:
#         y_l = [local_idx]
#     index_all = torch.cat(y_l)
#
#     # share features between processes
#     if is_dist_avail_and_initialized():
#         output_l = [torch.empty_like(local_feats) for _ in range(world_size)]
#         torch.distributed.all_gather(output_l, local_feats)
#     else:
#         output_l = [local_feats]
#
#     # update storage feature matrix
#     if global_rank == 0:
#         if use_cuda:
#             features.index_copy_(0, index_all, torch.cat(output_l))
#         else:
#             features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
#     return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features,
                   test_labels, k, T, num_classes=1000, ):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
                   idx: min((idx + imgs_per_chunk), num_test_images), :
                   ]
        targets = test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


@torch.no_grad()
def evaluate(model, query_loader, base_loader, use_fp16=False, nb_knn=[1, 5, 100], eval_embed='backbone'):
    metric_logger = utils.MetricLogger(delimiter="  ")

    local_rank, global_rank, world_size = world_info_from_env()

    model.eval()

    # Valid loader is the query set
    # Train loader is the base set
    use_cuda = True
    query_features = extract_features(model, query_loader, use_cuda, use_fp16, eval_embed)
    base_features = extract_features(model, base_loader, use_cuda, use_fp16, eval_embed)

    if global_rank == 0:
        assert query_features.ndim == 2
        query_features = nn.functional.normalize(query_features, dim=1, p=2)
        base_features = nn.functional.normalize(base_features, dim=1, p=2)

        num_classes = len(base_loader.dataset.classes)
        query_labels = torch.tensor(query_loader.dataset.targets).long()
        base_labels = torch.tensor(base_loader.dataset.targets).long()
        # query_labels = torch.tensor([s[1] for s in query_loader.dataset.samples]).long()
        # base_labels = torch.tensor([s[1] for s in base_loader.dataset.samples]).long()

        if use_cuda:
            query_labels = query_labels.cuda()
            base_labels = base_labels.cuda()

        print("Features are ready!\nStart the k-NN classification.")
        for k in nb_knn:
            top1, top5 = knn_classifier(base_features, base_labels, query_features, query_labels, k, 0.07, num_classes)
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")

        print("Start stats.")
        # Find the nearest neighbor indices for each query
        similarities = query_features @ base_features.T
        # similarities_idx = torch.argsort(similarities, dim=1, descending=True).cpu()
        similarities_idx = torch.topk(similarities, k=max(nb_knn), dim=1).indices.cpu()

        # Map neighbor indices to labels
        gts = (query_labels.view(-1, 1) == base_labels.view(1, -1)).float().cpu().numpy()
        preds = np.array([gts[i][similarities_idx[i]] for i in range(len(gts))])

        for topk in nb_knn:
            mode_recall = utils.Metrics.get_recall_bin(np.copy(preds), topk)
            mode_mrr = utils.Metrics.get_mrr_bin(np.copy(preds), topk)
            mode_map = utils.Metrics.get_map_bin(np.copy(preds), topk)
            # print(f'Recall@{topk}: {mode_recall:.2f}, mAP@{topk}: {mode_map:.2f}')
            metric_logger.update(**{f'recall@{topk}': mode_recall, f'mAP@{topk}': mode_map, f'MRR@{topk}': mode_mrr})

        # gather the stats from all processes
        print("Averaged stats:", metric_logger)

    dist.barrier()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    pass
