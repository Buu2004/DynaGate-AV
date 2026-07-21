import argparse
import numpy as np
import torch
import torch.optim as optim
from config import CONFIG
from models.dynamic_ffia import DynamicFFIAModel
from data.dataset import get_dataloader
from utils.trainer import train, evaluate
from utils.scheduler import IncreaseRegOnPlateau

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dynamic FFIA with regularization scheduler")
    parser.add_argument('--fusion_type', type=str, default='cross_attn',
                        choices=['none', 'cross_attn', 'simple_cross', 'self_attn', 'mbt'],
                        help='none = 2-branch only | others = 3-branch with selected fusion')
    parser.add_argument('--n_runs', type=int, default=CONFIG['n_runs'])
    parser.add_argument('--epochs', type=int, default=CONFIG['n_epochs'])
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--mode', type=str, default='max', choices=['max', 'min'])
    parser.add_argument('--factor', type=float, default=1.5)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=1e-2)
    parser.add_argument('--threshold_mode', type=str, default='rel', choices=['rel', 'abs'])
    parser.add_argument('--min_reg', type=float, default=0.00001)
    parser.add_argument('--max_reg', type=float, default=0.003)
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', default=True)
    args = parser.parse_args()

    CONFIG['n_runs'] = args.n_runs
    CONFIG['n_epochs'] = args.epochs
    CONFIG['batch_size'] = args.batch_size

    fusion_type = None if args.fusion_type == 'none' else args.fusion_type
    is_three_branch = fusion_type is not None
    num_log_cols = 6 if is_three_branch else 5

    train_loader = get_dataloader(
        split='train', batch_size=CONFIG['batch_size'],
        sample_rate=CONFIG['audio_cfg']['sample_rate'], seed=CONFIG['seed'],
        shuffle=True, drop_last=True, num_workers=4
    )
    val_loader = get_dataloader(
        split='val', batch_size=CONFIG['batch_size'],
        sample_rate=CONFIG['audio_cfg']['sample_rate'], seed=CONFIG['seed'],
        shuffle=False, num_workers=4
    )
    test_loader = get_dataloader(
        split='test', batch_size=CONFIG['batch_size'],
        sample_rate=CONFIG['audio_cfg']['sample_rate'], seed=CONFIG['seed'],
        shuffle=False, num_workers=4
    )

    log = np.zeros((CONFIG['n_runs'], num_log_cols))

    for n in range(CONFIG['n_runs']):
        print(f"\n{'='*80}")
        print(f"RUN {n+1}/{CONFIG['n_runs']} | Model: {'3-branch (' + fusion_type + ')' if is_three_branch else '2-branch'}")
        print(f"{'='*80}")

        model = DynamicFFIAModel(
            num_classes=CONFIG['num_classes'],
            fusion_type=fusion_type
        ).to(CONFIG['device'])
        
        try:
            model.load_state_dict(torch.load('/kaggle/working/best_model_955.pth', map_location=CONFIG['device']))
            print("Successfully loaded pre-trained weights.")
        except Exception as e:
            print(f"Could not load pre-trained weights: {e}. Starting from scratch.")

        optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

        reg_scheduler = IncreaseRegOnPlateau(
            reg=CONFIG['reg'],
            mode=args.mode,
            factor=args.factor,
            patience=args.patience,
            threshold=args.threshold,
            threshold_mode=args.threshold_mode,
            min_reg=args.min_reg,
            max_reg=args.max_reg,
            verbose=args.verbose
        )

        best_val_loss = float('inf')
        best_model_path = f"best_model_run_{n+1}_{fusion_type or '2branch'}_reg_scheduler.pth"

        for epoch in range(CONFIG['n_epochs']):
            t_metrics = train(model, train_loader, optimizer, epoch, CONFIG['device'])
            t_metrics['epoch'] = epoch + 1
            print(f"Epoch {epoch+1} Train: Acc={t_metrics['Accuracy']:.3f}, Loss={t_metrics['Loss']:.4f}, FLOP={t_metrics['FLOP']:.2e}, R1={t_metrics['Ratio_1']:.2f}, R2={t_metrics['Ratio_2']:.2f}" + (f", R3={t_metrics['Ratio_3']:.2f}" if is_three_branch else ""))

            v_metrics = evaluate(model, val_loader, CONFIG['device'], split='Validation')
            v_metrics['epoch'] = epoch + 1
            print(f"Epoch {epoch+1} Val: Acc={v_metrics['Accuracy']:.3f}, Loss={v_metrics['Loss']:.4f}, FLOP={v_metrics['FLOP']:.2e}, R1={v_metrics['Ratio_1']:.2f}, R2={v_metrics['Ratio_2']:.2f}" + (f", R3={v_metrics['Ratio_3']:.2f}" if is_three_branch else ""))

            reg_scheduler.step(v_metrics['Accuracy'])
            CONFIG['reg'] = reg_scheduler.reg

            if v_metrics['Loss'] < best_val_loss:
                best_val_loss = v_metrics['Loss']
                torch.save(model.state_dict(), best_model_path)
                print(f"   >>> Saved best model (Val Loss = {best_val_loss:.4f})")

        model.load_state_dict(torch.load(best_model_path, map_location=CONFIG['device']))
        print(f"\nLoaded best model from run {n+1} for final test")

        test_metrics = evaluate(model, test_loader, CONFIG['device'], split='Testing')
        test_metrics['epoch'] = CONFIG['n_epochs']

        log[n, 0] = test_metrics['Accuracy']
        log[n, 1] = test_metrics['Loss']
        log[n, 2] = test_metrics['FLOP']
        log[n, 3] = test_metrics['Ratio_1']
        log[n, 4] = test_metrics['Ratio_2']
        if is_three_branch:
            log[n, 5] = test_metrics['Ratio_3']

        print(f"Run {n+1} Result: Acc={test_metrics['Accuracy']:.3f}, FLOP={test_metrics['FLOP']:.2e}")

    print("\n" + "="*80)
    print("FINAL RESULTS ACROSS ALL RUNS")
    print("="*80)

    print(f"Test Accuracy : {np.mean(log[:, 0])*100:.2f} ± {np.std(log[:, 0])*100:.2f} %")
    print(f"Test Loss     : {np.mean(log[:, 1]):.4f} ± {np.std(log[:, 1]):.4f}")
    print(f"Avg FLOP      : {np.mean(log[:, 2]):.2e} ± {np.std(log[:, 2]):.2e}")
    print(f"Ratio_1 (Audio) : {np.mean(log[:, 3]):.3f} ± {np.std(log[:, 3]):.3f}")
    print(f"Ratio_2 (Video) : {np.mean(log[:, 4]):.3f} ± {np.std(log[:, 4]):.3f}")
    if is_three_branch:
        print(f"Ratio_3 (Fusion): {np.mean(log[:, 5]):.3f} ± {np.std(log[:, 5]):.3f}")

    best_idx = np.argmax(log[:, 0])
    print(f"\nBest run (max Accuracy): Run {best_idx+1}")
    print(f"   Acc = {log[best_idx, 0]:.4f} | FLOP = {log[best_idx, 2]:.2e}")
    print(f"   R1 = {log[best_idx, 3]:.3f} | R2 = {log[best_idx, 4]:.3f}" +
          (f" | R3 = {log[best_idx, 5]:.3f}" if is_three_branch else ""))