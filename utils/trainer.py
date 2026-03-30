import numpy as np
import torch
import torch.nn as nn
from config import CONFIG

def train(model, train_loader, optimizer, epoch, device):
    model.train()
    model.reset_weight()  # Start recording gate weights

    total_cls_loss = 0
    all_preds = []
    all_targets = []

    criterion_sum = nn.CrossEntropyLoss(reduction='sum')
    criterion_mean = nn.CrossEntropyLoss()

    for batch in train_loader:
        video = batch['video_form'].to(device)
        audio = batch['waveform'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()

        # Forward with regularization (only training model supports return_reg=True)
        output, probs = model(audio, video, return_reg=True)

        cls_loss_mean = criterion_mean(output, target)
        reg_loss = (probs * model.flop_weights.to(probs.device)).sum(dim=1).mean()
        loss = cls_loss_mean + (CONFIG['reg'] * reg_loss)

        loss.backward()
        optimizer.step()

        # Train: ging
        cls_loss_sum = criterion_sum(output, target).item()
        total_cls_loss += cls_loss_sum

        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(torch.argmax(target, dim=1).cpu().numpy())

    # Metrics
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    avg_loss = total_cls_loss / len(train_loader.dataset)

    # FLOPs & expert ratios (dynamic for 2 or 3 experts)
    flops = model.cal_flop()
    ratios = model.weight_stat()          # tuple of length 2 or 3

    metrics = {
        'Accuracy': acc,
        'Loss': avg_loss,
        'FLOP': flops,
        'Ratio_1': ratios[0],   # Audio
        'Ratio_2': ratios[1],   # Video
    }
    if len(ratios) == 3:        # 3-branch model
        metrics['Ratio_3'] = ratios[2]  # Fusion

    return metrics


def evaluate(model, loader, device, split='Evaluating'):
    model.eval()
    model.reset_weight()  # Start recording gate weights

    total_loss = 0
    all_preds = []
    all_targets = []

    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for batch in loader:
            video = batch['video_form'].to(device)
            audio = batch['waveform'].to(device)
            target = batch['target'].to(device)

            # Forward with return_reg=True (only training model supports it)
            output, _ = model(audio, video, return_reg=True)

            loss = criterion(output, target)
            total_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(torch.argmax(target, dim=1).cpu().numpy())

    # Metrics
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    avg_loss = total_loss / len(loader.dataset)

    # FLOPs & expert ratios (dynamic for 2 or 3 experts)
    flops = model.cal_flop()
    ratios = model.weight_stat()          # tuple of length 2 or 3

    metrics = {
        'Accuracy': acc,
        'Loss': avg_loss,
        'FLOP': flops,
        'Ratio_1': ratios[0],   # Audio
        'Ratio_2': ratios[1],   # Video
    }
    if len(ratios) == 3:        # 3-branch model
        metrics['Ratio_3'] = ratios[2]  # Fusion

    return metrics