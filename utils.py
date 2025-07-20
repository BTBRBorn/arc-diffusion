import torch
import model as m
from pathlib import Path
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from matplotlib.colors import Normalize, ListedColormap

def configure_optimizer(model, config):

    optim_groups = [
        {
            "params": [p for p in model.parameters() if p.dim() >= 2],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for p in model.parameters() if p.dim() < 2],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        fused=True,
    )

    return optimizer

def save_checkpoint(
    checkpoint_path, model, optimizer, scheduler, tokenizer, config, results
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.parent.exists():
        checkpoint_path.parent.mkdir(parents=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "tokenizer": tokenizer,
            "config": config,
            "results": results,
        },
        Path(checkpoint_path),
    )


def load_checkpoint(checkpoint_path, device, compile_model=False, with_model=True, weight_only=False):

    checkpoint = torch.load(Path(checkpoint_path), weights_only=weight_only)

    config = checkpoint["config"]

    if with_model:
        base_model = m.GPT(config=config, device=device).to(device)

        base_model.load_state_dict(checkpoint["model_state_dict"])

        if compile_model:
            model = DDP(base_model, device_ids=[device])
            model = torch.compile(model)
        else:
            model = DDP(base_model, device_ids=[device])

        optimizer = configure_optimizer(model, config)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.scheduler_iter, eta_min=1e-6,
        )
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        base_model, model, optimizer, scheduler = None, None, None, None

    tokenizer = checkpoint["tokenizer"]

    results = checkpoint["results"]

    return_dict = {
        "base_model": base_model,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "tokenizer": tokenizer,
        "config": config,
        "results": results,
    }

    return return_dict


def plot_losses(results):
    train_loss, val_loss = results["train_losses"], results["val_losses"]
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="training", color="green")
    iters, losses = zip(*val_loss)
    plt.plot(iters, losses, label="validation", color="red")
    plt.title("Train and Validation Losses vs Num Epochs")
    plt.xlabel("Num Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")

#plot_task function below is copied from Micheal Hodel's https://github.com/michaelhodel/re-arc/utils.py
def plot_task(
    task: list[dict],
    title: str = None
) -> None:
    """
    displays a task
    """
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = Normalize(vmin=0, vmax=9)
    args = {'cmap': cmap, 'norm': norm}
    height = 2
    width = len(task)
    figure_size = (width * 3, height * 3)
    figure, axes = plt.subplots(height, width, figsize=figure_size)
    for column, example in enumerate(task):
        axes[0, column].imshow(example['input'], **args)
        axes[1, column].imshow(example['output'], **args)
        axes[0, column].axis('off')
        axes[1, column].axis('off')
    if title is not None:
        figure.suptitle(title, fontsize=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()