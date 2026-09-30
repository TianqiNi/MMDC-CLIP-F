"""Aggregate the completed private screen; emit no patient rows or identifiers."""
import argparse
import json
from pathlib import Path


METHODS = ('candidate', 'no_effect_supervision', 'magnitude_effects')
LABELS = {'candidate': 'Signed', 'no_effect_supervision': 'Error only', 'magnitude_effects': 'Unsigned'}


def fmt(value):
    return f'{value:.4f}' if value is not None else 'NA'


def gain(candidate, baseline):
    return 100 * (1 - candidate / baseline) if baseline else None


def effect_summary(private, method, seed, epoch):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    values = load_file(str(private / f'{method}-seed-{seed}/scores-epoch-{epoch:02d}.safetensors'))
    targets = [load_file(str(private / f'prepared/tune-{i:02d}.safetensors')) for i in range(17)]
    truth = torch.stack([t['target.omission_labels'] for t in targets]).numpy()
    valid = torch.stack([t['target.valid_removal_mask'] for t in targets]).numpy()
    result = {}
    for name in ('raw_effect', 'reported_effect'):
        predictions = values[name].numpy()
        matrix = np.zeros((3, 3), dtype=int)
        for t, p in zip(truth[valid], predictions[valid]):
            matrix[int(t), int(p)] += 1
        support = matrix.sum(1)
        recalls = [float(matrix[i, i] / n) if n else None for i, n in enumerate(support)]
        result[name] = {'class_order': ['damage', 'unchanged', 'repair'],
            'confusion_matrix': matrix.tolist(), 'support': support.tolist(), 'recall': recalls,
            'balanced_accuracy': float(np.mean([r for r in recalls if r is not None])),
            'predicted_class_counts': matrix.sum(0).tolist(),
            'majority_unchanged_accuracy': float(support[1] / support.sum())}
    return result


def aggregate(summary, private):
    import statistics
    output = {'evidence_kind': 'real_data_development_screening_only', 'seeds': [42, 43, 44],
              'msp': summary['msp'], 'provenance': summary['provenance'], 'modes': {}}
    for mode in ('selected', 'fixed_epoch_20'):
        rows, seed_gains = [], []
        for seed in output['seeds']:
            values = {m: summary['results'][f'{m}:{seed}'][mode] for m in METHODS}
            candidate = values['candidate']
            gains = {m: gain(candidate['balanced_aurc'], values[m]['balanced_aurc'])
                     for m in METHODS[1:]}
            gains['msp'] = gain(candidate['balanced_aurc'], summary['msp']['balanced_aurc'])
            gains.update(seed=seed, clean_delta_vs_msp=candidate['clean_aurc'] - summary['msp']['clean_aurc'])
            seed_gains.append(gains)
            for method, value in values.items():
                cells = value['cells']
                rows.append({'method': method, 'seed': seed, 'epoch': value['epoch'],
                    'clean_aurc': value['clean_aurc'], 'balanced_aurc': value['balanced_aurc'],
                    'clean_error_auroc': cells[0]['error_auroc'],
                    'balanced_error_auroc': statistics.mean(x['error_auroc'] for x in cells[1:]),
                    'clean_brier': cells[0]['brier'],
                    'balanced_brier': statistics.mean(x['brier'] for x in cells[1:])})
        mean_sd = {}
        for method in METHODS:
            subset = [x for x in rows if x['method'] == method]
            mean_sd[method] = {key: {'mean': statistics.mean(x[key] for x in subset),
                                     'sample_sd': statistics.stdev(x[key] for x in subset)}
                              for key in ('clean_aurc', 'balanced_aurc', 'clean_error_auroc',
                                          'balanced_error_auroc', 'clean_brier', 'balanced_brier')}
        output['modes'][mode] = {'rows': rows, 'seed_gains_percent': seed_gains, 'mean_sd': mean_sd}
    selected = output['modes']['selected']['seed_gains_percent']
    fixed = output['modes']['fixed_epoch_20']['seed_gains_percent']
    promising_seeds = [x['seed'] for x in selected if all(x[m] >= 5 for m in (*METHODS[1:], 'msp'))]
    clean_seeds = [x['seed'] for x in selected if x['clean_delta_vs_msp'] <= .005]
    fixed_positive = [x['seed'] for x in fixed if all(x[m] > 0 for m in (*METHODS[1:], 'msp'))]
    output['development_heuristic'] = {'promising_balanced_seeds': promising_seeds,
        'clean_within_0_005_seeds': clean_seeds, 'fixed_epoch_directionally_positive_seeds': fixed_positive,
        'promising': len(promising_seeds) >= 2 and len(clean_seeds) >= 2 and len(fixed_positive) >= 2,
        'interpretation': 'Prespecified development heuristic only; tune checkpoint selection is optimistic. Not a publication or full pilot gate.'}
    output['signed_effect_diagnostics'] = {str(seed): effect_summary(private, 'candidate', seed,
        summary['results'][f'candidate:{seed}']['selected']['epoch']) for seed in output['seeds']}
    return output


def markdown(report):
    msp = report['msp']
    lines = ['# Matched signed-supervision development screen', '',
        'This is real RSNA confidence-fit/tune evidence. It does not qualify the full P5B pilot or establish publication-level generalization.', '',
        'The epoch-38 classifier stayed frozen. Each head used 987 confidence-fit exams, 20 epochs, seeds 42/43/44, identical draws/shared initialization, Adam 1e-4, weight decay 0 and batch size 6. Screening uses 370 tune exams in clean plus 16 allowed noise/blur cells. No pilot or locked test outcomes were accessed.', '',
        f"MSP reference: clean AURC **{msp['clean_aurc']:.6f}**, balanced stressed AURC **{msp['balanced_aurc']:.6f}**. Clean classifier accuracy: **{100*msp['cells'][0]['accuracy']:.2f}%**. The MSP scores are the same deterministic reference for all seeds.", '',
        'Lower AURC and Brier are better; higher error AUROC is better. Balanced metrics equally weight the 16 cells. Seed variation is not a confidence interval.']
    for mode, title in [('selected', 'Tune-selected checkpoints'), ('fixed_epoch_20', 'Fixed epoch 20')]:
        lines += ['', f'## {title}', '', '| Method | Seed | Epoch | Clean AURC | Stressed AURC | Stressed error AUROC | Stressed Brier |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for row in report['modes'][mode]['rows']:
            lines.append(f"| {LABELS[row['method']]} | {row['seed']} | {row['epoch']} | {row['clean_aurc']:.6f} | {row['balanced_aurc']:.6f} | {row['balanced_error_auroc']:.4f} | {row['balanced_brier']:.4f} |")
        lines += ['', '| Signed relative AURC reduction (%) | vs error only | vs unsigned | vs MSP | Clean Δ vs MSP |',
                  '|---|---:|---:|---:|---:|']
        for row in report['modes'][mode]['seed_gains_percent']:
            lines.append(f"| Seed {row['seed']} | {row['no_effect_supervision']:+.2f} | {row['magnitude_effects']:+.2f} | {row['msp']:+.2f} | {row['clean_delta_vs_msp']:+.6f} |")
    gate = report['development_heuristic']
    lines += ['', '## Prespecified screening decision', '',
        f"Promising balanced-AURC seeds (≥5% versus both learned controls and MSP): **{gate['promising_balanced_seeds']}**.", '',
        f"Seeds within the diagnostic clean MSP +0.005 bound: **{gate['clean_within_0_005_seeds']}**. Fixed-epoch signed gains versus all three references: **{gate['fixed_epoch_directionally_positive_seeds']}**.", '',
        f"The prespecified promising-signal heuristic **{'is satisfied' if gate['promising'] else 'is not satisfied'}**. It is a development filter, not a statistical significance claim. The full protocol's same-seed clean MV-ACN reference was not fitted.", '',
        '## Signed-effect diagnostics', '',
        'Each row aggregates the 17 tune cells; repeated views/cells are not independent patients. Raw auxiliary predictions exclude the deterministic unchanged-prediction overwrite. Constrained output can appear better because unchanged classifier predictions imply zero effect by construction.', '',
        '| Seed | Raw damage recall | Raw unchanged recall | Raw repair recall | Raw balanced accuracy | Constrained balanced accuracy |',
        '|---|---:|---:|---:|---:|---:|']
    for seed, values in report['signed_effect_diagnostics'].items():
        raw, constrained = values['raw_effect'], values['reported_effect']
        lines.append(f"| {seed} | {fmt(raw['recall'][0])} | {fmt(raw['recall'][1])} | {fmt(raw['recall'][2])} | {fmt(raw['balanced_accuracy'])} | {fmt(constrained['balanced_accuracy'])} |")
    lines += ['', '## Limits and next decision', '',
        'The 20 checkpoints were screened on the same tune patients used to select the frozen classifier. These results are optimistic development evidence. Neither paired-patient significance nor transfer to held-out corruption families, missing-view panels, an independent pilot or test has been established.', '',
        'The strongest mandatory same-input MLP/density, original/matched MV-ACN, ViLU and evidential/calibrated controls remain pending. A positive signed-versus-control screen would justify completing them; an inconsistent signed advantage would require a separate research decision rather than outcome-driven retuning. This screen reports all seeds and fixed-budget outcomes.', '',
        '![All checkpoint trajectories](trajectories.png)', '']
    return '\n'.join(lines)


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'candidate': '#0072B2', 'no_effect_supervision': '#D55E00', 'magnitude_effects': '#009E73'}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True, sharey='row')
    for column, seed in enumerate((42, 43, 44)):
        for row, endpoint in enumerate(('balanced_aurc', 'clean_aurc')):
            ax = axes[row, column]
            for method in METHODS:
                history = summary['results'][f'{method}:{seed}']['history']
                ax.plot([x['epoch'] for x in history], [x[endpoint] for x in history],
                        color=colors[method], label=LABELS[method], linewidth=1.6)
                chosen = summary['results'][f'{method}:{seed}']['selected']
                ax.scatter([chosen['epoch']], [chosen[endpoint]], color=colors[method], s=25, zorder=3)
            ax.axhline(summary['msp'][endpoint], color='#333333', linestyle='--', label='MSP', linewidth=1.2)
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(f'Seed {seed}')
            if column == 0:
                ax.set_ylabel('Balanced stressed AURC' if row == 0 else 'Clean AURC')
            if row == 1:
                ax.set_xlabel('Epoch')
            ax.set_xticks([1, 5, 10, 15, 20])
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Tune-only signed-supervision screen — lower AURC is better\nDots: checkpoint selected by balanced stressed AURC; all 20 epochs shown', fontsize=11)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'trajectories.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True)
    parser.add_argument('--private', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    output, private = Path(args.output), Path(args.private)
    summary = json.loads(Path(args.summary).read_text())
    if args.plot_only:
        plot(summary, output)
        return
    if json.loads((private / 'execution.json').read_text())['status'] != 'PASS':
        raise ValueError('cannot report an incomplete or failed screening run')
    report = aggregate(summary, private)
    (output / 'aggregate.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'results.md').write_text(markdown(report))
    print(json.dumps(report['development_heuristic']))


if __name__ == '__main__':
    main()
