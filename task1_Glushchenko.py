import os
import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    # pin_memory и non_blocking позволяют копировать батчи cpu в gpu асинхронно
    # persistent_workers не пересоздает worker процессы между эпохами
    num_workers = min(4, os.cpu_count() or 1) if use_cuda else 0
    dataloader_kwargs = {
        "batch_size": 256,
        "shuffle": True,
        "pin_memory": use_cuda,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        dataloader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    dataloader = DataLoader(prepare_data(), **dataloader_kwargs)

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)

    # не храним loss тензоры в списке иначе сохраняется весь graph и память на gpu течет))
    loss_sum = torch.zeros((), device=device)
    num_samples = 0

    forward_events = []
    backward_events = []
    forward_times = []
    backward_times = []

    for data, target in dataloader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # шум создаем сразу на gpu без промежуточного cpu тензора и лишней синхронной копии
        data = data + torch.randn_like(data)

        # set_to_none=True быстрее и экономнее чем занулять уже выделенные градиенты
        optimizer.zero_grad(set_to_none=True)

        if use_cuda:
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)

            # cuda операции асинхронны поэтому time.time здесь дает нечестные метрики
            fwd_start.record()
            output = model(data)
            loss = criterion(output, target)
            fwd_end.record()

            bwd_start.record()
            loss.backward()
            optimizer.step()
            bwd_end.record()

            forward_events.append((fwd_start, fwd_end))
            backward_events.append((bwd_start, bwd_end))
        else:
            time_start = time.perf_counter()
            output = model(data)
            loss = criterion(output, target)
            forward_times.append(time.perf_counter() - time_start)

            time_start_bwd = time.perf_counter()
            loss.backward()
            optimizer.step()
            backward_times.append(time.perf_counter() - time_start_bwd)

        # для метрик берем detach и считаем средний loss по объектам, а не по батчам
        with torch.no_grad():
            batch_size = target.size(0)
            loss_sum += loss.detach() * batch_size
            num_samples += batch_size

        # не вызываем loss.item и torch.cuda.empty_cache в каждом батче
        # они синхронизируют cpu/gpu и ломают асинхронный пайплайн

    if use_cuda:
        torch.cuda.synchronize()
        forward_times = [start.elapsed_time(end) / 1000 for start, end in forward_events]
        backward_times = [start.elapsed_time(end) / 1000 for start, end in backward_events]

    avg_loss = (loss_sum / num_samples).item()
    print(f"Epoch finished, avg loss is {avg_loss:.4f}, "
          f"avg forward time is {statistics.fmean(forward_times):.6f}, "
          f"avg backward+step time is {statistics.fmean(backward_times):.6f}")


if __name__ == '__main__':
    train()
