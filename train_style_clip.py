import copy
import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
import numpy as np

import torch
from torch import optim

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

from style_clip.model import StyleCLIP, StyleLoss, CustomTokenizer
from style_clip.data import get_data
from style_clip.augment import ContrastiveTransformations, get_clip_transform
from style_clip.distributed import is_master, init_distributed_device, broadcast_object
from style_clip.logger import setup_logging
from style_clip.params import parse_args
from style_clip.scheduler import cosine_lr, const_lr, const_lr_cooldown
from style_clip.train import train_one_epoch
from style_clip.eval import evaluate
from style_clip.file_utils import pt_load
from style_clip import clip_utils


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


# def random_seed(seed=42):
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    args.lr = args.lr * (args.accum_freq * args.batch_size * args.world_size) / 1024.
    args.lr_head = args.lr_head * (args.accum_freq * args.batch_size * args.world_size) / 1024.

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"lrhead_{args.lr_head}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        # if os.path.exists(args.log_path) and not resume_latest:
        #     print(
        #         "Error. Experiment already exists. Use --name {} to specify a new experiment."
        #     )
        #     return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=False)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]

    random_seed(args.seed, 0)
    model = StyleCLIP(args.model, args.model_text)
    model = model.to(device)
    random_seed(args.seed, args.rank)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    else:
        model.prep_lora_model()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    model_without_ddp = model
    if args.distributed:
        if args.use_bn_sync and clip_utils.has_batchnorms(model):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
        model_without_ddp = model.module

    loss = StyleLoss(args.temp, args)

    # create optimizer and scaler
    optimizer = None
    optimizer_head = None
    scaler = None

    if args.train_data:
        num_params = 0
        if args.lock_image:
            backbone_params = clip_utils.collect_params(model_without_ddp.visual)
            num_params += sum([len(p["params"]) for p in backbone_params])
        else:
            backbone_params = [p for p in model_without_ddp.visual.parameters() if p.requires_grad]
            num_params += len(backbone_params)
        head_params = list(model_without_ddp.style_proj.parameters()) + list(model_without_ddp.content_proj.parameters())
        num_params += len(head_params)

        assert num_params == len([p for p in model_without_ddp.parameters() if p.requires_grad])
        print("num params:", num_params)

        opt = getattr(args, 'opt', 'adamw').lower()
        if opt == 'adamw':
            optimizer = optim.AdamW(
                backbone_params,
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
                weight_decay=args.wd
            )

            optimizer_head = optim.AdamW(
                head_params,
                lr=args.lr_head,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
                weight_decay=args.wd  # whether to apply weight decay CSD
            )
        else:
            assert False, f'Unknown optimizer {opt}'

        if is_master(args):
            defaults = copy.deepcopy(optimizer.defaults)
            defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
            logging.info(
                f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
            )

            defaults = copy.deepcopy(optimizer_head.defaults)
            defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
            logging.info(
                f'Created {type(optimizer_head).__name__} ({args.opt}) optimizer: {defaults}'
            )

        if args.precision == "amp":
            try:
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError) as e:
                scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if optimizer_head is not None:
                optimizer_head.load_state_dict(checkpoint["optimizer_head"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    preprocess_train = ContrastiveTransformations(size=model_without_ddp.processor.image_processor.size["shortest_edge"],
                                                  mean=model_without_ddp.processor.image_processor.image_mean,
                                                  std=model_without_ddp.processor.image_processor.image_std,
                                                  lambda_c=args.lambda_c)
    preprocess_val = get_clip_transform(size=model_without_ddp.processor.image_processor.size["shortest_edge"],
                                       mean=model_without_ddp.processor.image_processor.image_mean,
                                       std=model_without_ddp.processor.image_processor.image_std)

    # initialize datasets
    tokenizer = CustomTokenizer(args.model_text)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    scheduler_head = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
            scheduler_head = cosine_lr(optimizer_head, args.lr_head, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    optimizer = [optimizer, optimizer_head]
    scheduler = [scheduler, scheduler_head]

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    original_model = model

    # evaluate(model, data["query"].dataloader, data["base"].dataloader, use_fp16=scaler is not None, nb_knn=[1, 5, 100],
    #          eval_embed='backbone')
    if 'train' not in data:
        # Evaluate.
        evaluate(model, data["query"].dataloader, data["base"].dataloader, use_fp16=scaler is not None, nb_knn=[1, 5, 100], eval_embed='backbone')
        return

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, args, tb_writer=writer)
        completed_epoch = epoch + 1

        if (epoch == 0) or (completed_epoch % args.val_frequency == 0) or (completed_epoch == args.epochs):
            evaluate(model, data["query"].dataloader, data["base"].dataloader, use_fp16=scaler is not None, nb_knn=[1, 5, 100], eval_embed='backbone')

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer[0].state_dict(),
                "optimizer_head": optimizer[1].state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args):
        wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
