import torch
import torch.nn.functional as F
import time
import torch.distributed as dist
from pathlib import Path
import utils


# Calculate the loss for one batch
def calculate_loss(model, config, x, y, grad_accum_num=1):
    B, T = x.size()
    if config.use_mixed_precision:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x, config.attention_mode)
            loss = F.cross_entropy(logits.view(B * T, config.vocab_size), y.view(B * T))
            loss = loss / grad_accum_num
    else:
        logits = model(x, config.attention_mode)
        loss = F.cross_entropy(logits.view(B * T, config.vocab_size), y.view(B * T))
        loss = loss / grad_accum_num

    return loss


def train_step(
    model,
    x,
    y,
    optimizer,
    config,
    grad_accum_num,
    batch_num,
):
    if batch_num % grad_accum_num != 0:
        with model.no_sync():
            loss = calculate_loss(model, config, x, y, grad_accum_num)
            loss.backward()
            norm = None
    else:
        loss = calculate_loss(model, config, x, y, grad_accum_num)
        # Sync all the gradients
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return loss, norm


def val_step(model, dataloader, config, device):
    avg_loss = 0.0
    model.eval()
    iter_steps = 0
    with torch.inference_mode():
        for x, y in dataloader:
            iter_steps += 1
            x, y = x.to(device), y.to(device)
            loss = calculate_loss(model, config, x, y)
            avg_loss += loss.detach()

        avg_loss /= iter_steps
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)

    return avg_loss


def train(
    models,
    optimizer,
    scheduler,
    train_dataloader,
    val_dataloader,
    checkpoint_intervals,
    config,
    tokenizer,
    results,
    checkpoint_save_path,
    grad_accum_intervals,
    scheduler_intervals,
    is_master,
    device,
    world_size,
):
    base_model, model = models
    model.train()
    total_iter = len(train_dataloader) // scheduler_intervals
    current_iter = len(results["train_losses"]) + 1
    grad_accum_num = results["grad_accum_num"]
    val_loss = val_step(
        model=model,
        dataloader=val_dataloader,
        config=config,
        device=device,
    )
    if is_master:
        print(f"Continuing training from iteration: {current_iter}")
        print(f"Validation Loss after iteration {current_iter}: {val_loss:.4f}")
    batch_tokens = config.batch_size * config.block_size * world_size
    total_tokens = 0
    iter_num = 0
    start = time.perf_counter()
    avg_loss = 0
    norms = []
    for batch_num, (x, y) in enumerate(train_dataloader, start=1):
        total_tokens += batch_tokens
        x, y = x.to(device), y.to(device)
        train_loss, norm = train_step(
            model,
            x,
            y,
            optimizer,
            config,
            grad_accum_num,
            batch_num,
        )

        if is_master:
            avg_loss += train_loss.item() * grad_accum_num
            if norm is not None:
                norms.append(norm.item())

        if not (batch_num % scheduler_intervals):
            scheduler.step()
            if is_master:
                end = time.perf_counter()
                dt = end - start
                lr = scheduler.get_last_lr()
                token_per_sec = total_tokens / dt
                iter_num += 1

                # Calculate avg_loss
                avg_loss /= scheduler_intervals
                results["train_losses"].append(avg_loss)
                # Calculate avg_norm
                avg_norm = sum(norms) / len(norms)
                print(
                    f"Iter: {iter_num}/{total_iter}, Train Loss: {avg_loss:.4f}, dt: {dt:.4f} seconds, "
                    + f"tokens/sec: {token_per_sec:.2f}, norm: {avg_norm:.4f}, learning_rate: {lr[0]:.6e}"
                )
                # Re-initialize all variables
                avg_loss = 0
                norms = []
                total_tokens = 0
                start = time.perf_counter()

        # Every grad_accum_intervals iterations increase batch accumulation by one
        if not (batch_num % grad_accum_intervals):
            grad_accum_num += 1
            results["grad_accum_num"] = grad_accum_num
            if is_master:
                print(f"Gradient accumulation is increased to: {grad_accum_num}")

        # Checkpoint and validation
        if not (batch_num % checkpoint_intervals):
            val_loss = val_step(
                model=model, dataloader=val_dataloader, config=config, device=device
            )

            if is_master:
                current_iter = len(results["train_losses"])
                print(f"Validation Loss after iteration {current_iter}: {val_loss:.4f}")
                results["val_losses"].append((current_iter, val_loss.item()))

            if is_master and checkpoint_save_path:
                utils.save_checkpoint(
                    checkpoint_path=Path(checkpoint_save_path),
                    model=base_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    tokenizer=tokenizer,
                    config=config,
                    results=results,
                )

    return results
