python -c "
import json
d = json.load(open('MPO/logs/Qwen2.5-VL-7B/Qwen2.5-VL-7B/diffusers-flux-schnell/mpo/cuckoo/cuckoo_0.json'))
from collections import Counter, defaultdict
correct = defaultdict(int); total = defaultdict(int); confusion = defaultdict(Counter)
for ex in d['train_best_correct_examples'] + d['train_best_wrong_examples']:
    ans = ex.get('label') or ex.get('answer')
    if isinstance(ans, list): ans = ans[0]
    pred = ex.get('model_answer', '')
    total[ans] += 1
    if ex.get('correct') == 1: correct[ans] += 1
    confusion[ans][pred] += 1
for cls in sorted(total):
    print(f'{cls}: {correct[cls]}/{total[cls]} = {correct[cls]/total[cls]:.3f}')
print()
for true_cls, preds in confusion.items():
    print(f'  {true_cls}: {dict(preds)}')
"