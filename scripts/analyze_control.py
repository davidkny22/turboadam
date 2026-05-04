import json

with open('experiments/results/control_experiment/baseline_log.jsonl') as f:
    baseline = [json.loads(l) for l in f]
with open('experiments/results/control_experiment/turboadam_log.jsonl') as f:
    turbo = [json.loads(l) for l in f]

print('Step | Baseline (AdamW) | TurboAdam (m+v) | Delta | Rel Gap')
print('-'*65)
for b, t in zip(baseline, turbo):
    step = b['step']
    delta = t['loss'] - b['loss']
    rel = delta / b['loss'] * 100 if b['loss'] > 0 else 0
    marker = '***' if step == 500 else ''
    print(f"{step:>4} | {b['loss']:>16.4f} | {t['loss']:>15.4f} | {delta:>+5.2f} | {rel:>5.1f}% {marker}")

print()
print(f"Baseline  time: {baseline[-1]['elapsed_s']:.1f}s ({baseline[-1]['elapsed_s']/500:.2f}s/step)")
print(f"TurboAdam time: {turbo[-1]['elapsed_s']:.1f}s ({turbo[-1]['elapsed_s']/500:.2f}s/step)")
print(f"Speed delta: {(turbo[-1]['elapsed_s']/baseline[-1]['elapsed_s']-1)*100:.1f}%")
