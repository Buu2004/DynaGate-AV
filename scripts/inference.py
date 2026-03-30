import argparse
import time
import psutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from config import CONFIG
from models.inference_model import DynamicFFIAModelOneSwitch
from data.dataset import get_dataloader   

def run_inference(model, loader, device, fusion_type):
    model = model.to(device)
    model.eval()
    all_preds, all_targets, all_chosen = [], [], []
    start_time = time.time()

    process = psutil.Process()
    peak_mem_mb = 0
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    for batch in tqdm(loader, desc=f"Inference on {device}"):
        video = batch['video_form'].to(device)
        audio = batch['waveform'].to(device)
        target = batch['target'].to(device)

        output, chosen = model(audio, video)

        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(torch.argmax(target, dim=1).cpu().numpy())
        all_chosen.extend(chosen.cpu().numpy())

        if device == 'cuda':
            torch.cuda.synchronize()
            peak_mem_mb = max(peak_mem_mb, torch.cuda.max_memory_allocated() / 1e6)
        else:
            peak_mem_mb = max(peak_mem_mb, process.memory_info().rss / 1e6)

    total_time = time.time() - start_time
    num_samples = len(all_preds)
    fps = num_samples / total_time

    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds, normalize='true') * 100
    chosen_arr = np.array(all_chosen)
    expert_names = ['Audio', 'Video', 'Fusion'] if fusion_type is not None else ['Audio', 'Video']
    class_names = ['none', 'strong', 'medium', 'weak']
    overall_usage = np.bincount(chosen_arr, minlength=len(expert_names)) / len(chosen_arr) * 100

    # Per-class expert usage
    class_usage = np.zeros((4, len(expert_names)))
    for c in range(4):
        mask = np.array(all_targets) == c
        if mask.sum() > 0:
            class_usage[c] = np.bincount(chosen_arr[mask], minlength=len(expert_names)) / mask.sum() * 100

    # ==================== PLOTS ====================
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Acc = {acc:.3f})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrix_{fusion_type or "2branch"}.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=expert_names, y=overall_usage)
    plt.title(f'Overall Expert Usage (%)')
    plt.ylabel('Usage %')
    plt.savefig(f'expert_usage_overall_{fusion_type or "2branch"}.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(class_usage, annot=True, fmt='.1f', cmap='Oranges',
                xticklabels=expert_names, yticklabels=class_names)
    plt.title(f'Expert Usage per Class (%)')
    plt.xlabel('Chosen Expert')
    plt.ylabel('True Class')
    plt.savefig(f'expert_per_class_{fusion_type or "2branch"}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n One-Switch Inference on {device.upper()} | {fusion_type or '2-branch'} 🔥")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Time     : {total_time:.2f} s")
    print(f"   FPS      : {fps:.1f} samples/sec")
    print(f"   Peak Mem : {peak_mem_mb:.0f} MB")
    print(f"   Expert usage: {dict(zip(expert_names, overall_usage.round(1)))}")
    return acc, overall_usage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion_type', type=str, default='cross_attn',
                        choices=['none', 'cross_attn', 'simple_cross', 'self_attn', 'mbt'],
                        help='none = 2-branch, else 3-branch fusion')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained .pth (e.g. best_model_run_1_cross_attn.pth)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    fusion_type = None if args.fusion_type == 'none' else args.fusion_type

    model = DynamicFFIAModelOneSwitch(num_classes=CONFIG['num_classes'], fusion_type=fusion_type)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    print(f"Loaded model: {args.model_path}")

    test_loader = get_dataloader(split='test', batch_size=CONFIG['batch_size'],
                                 sample_rate=CONFIG['audio_cfg']['sample_rate'],
                                 seed=CONFIG['seed'], shuffle=False)

    run_inference(model, test_loader, args.device, fusion_type)