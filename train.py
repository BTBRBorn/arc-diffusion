from pathlib import Path
import argparse
import pickle
import torch

from get_tokenizer import Tokenizer
from get_dataloaders import create_dataloaders
import model as m
import engine
import utils
from configurations import Config

# Modules needed for multi-gpu training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # All intervals are expressed in terms of batches
    parser.add_argument("--checkpoint_intervals", type=int, default=100)
    parser.add_argument("--scheduler_intervals", type=int, default=30)
    parser.add_argument("--grad_accum_intervals", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--vocab_size", type=int, default=17)
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--n_layer", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--head_size", type=int, default=32)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--data_path", type=str, default="data/pretraining")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--compile_model", type=int, choices={0, 1}, default=0)
    parser.add_argument("--attention_mode", type=str, default="flash_attention")
    parser.add_argument("--use_mixed_precision", type=int, choices={0, 1}, default=1)
    parser.add_argument("--checkpoint_save_path", type=str, default="")
    parser.add_argument("--checkpoint_load_path", type=str, default="")
    parser.add_argument("--scheduler_iter", type=int, default=1200)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--tokenizer_path", type=str, default="")

    args = parser.parse_args()

    # torchrun will setup rank, local_rank and world_size for us
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{ddp_local_rank}")
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0

    torch.set_float32_matmul_precision("high")

    if args.checkpoint_load_path:
        checkpoint_dict = utils.load_checkpoint(
            Path(args.checkpoint_load_path),
            compile_model=args.compile_model,
            device=device,
        )

        config = checkpoint_dict["config"]

        base_model, model = checkpoint_dict["base_model"], checkpoint_dict["model"]
        models = (base_model, model)

        optimizer = checkpoint_dict["optimizer"]

        scheduler = checkpoint_dict["scheduler"]

        tokenizer = checkpoint_dict["tokenizer"]

        results = checkpoint_dict["results"]

    else:
        config = Config(args)

        base_model = m.GPT(config=config, device=device).to(device)

        if args.compile_model:
            model = DDP(base_model, device_ids=[device])
            model = torch.compile(model)
        else:
            model = DDP(base_model, device_ids=[device])

        models = (base_model, model)

        optimizer = utils.configure_optimizer(model, config)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler_iter,
            eta_min=1e-6,
        )

        if args.tokenizer_path:
            with open(Path(args.tokenizer_path), "rb") as fhandle:
                tokenizer = pickle.load(fhandle)
        else:
            tokenizer = Tokenizer(config.vocab_size)

        results = {
            "train_losses": [],
            "val_losses": [],
            "training_run": 0,
            "grad_accum_num": 1,
        }

    train_dataloader, val_dataloader, train_sampler = create_dataloaders(
        config, args.data_path
    )
    results["training_run"] += 1
    train_sampler.set_epoch(results["training_run"])

    if master_process:
        print("-" * 50)
        print(f"Device: {device}")
        print("-" * 50)
        print("Model:")
        print(model)
        print("-" * 50)
        print(optimizer)
        print("-" * 50)
        print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
        print("-" * 50)
        print(
            "Total number of tokens in every training step: "
            + f"{config.batch_size * config.block_size * results['grad_accum_num'] * world_size}",
        )
        print("-" * 50)


    results = engine.train(
        models=models,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        checkpoint_intervals=args.checkpoint_intervals,
        config=config,
        tokenizer=tokenizer,
        results=results,
        checkpoint_save_path=args.checkpoint_save_path,
        grad_accum_intervals=args.grad_accum_intervals,
        scheduler_intervals=args.scheduler_intervals,
        is_master=master_process,
        device=device,
        world_size=world_size,
    )

    dist.destroy_process_group()
