import argparse
import collections
import json
import re
import string

def parse_args():
    parser = argparse.ArgumentParser('Evaluation script for custom QA dataset.')
    parser.add_argument('data_file', metavar='data.json', help='Input ground truth JSON file.')
    parser.add_argument('pred_file', metavar='pred.json', help='Model predictions JSON file.')
    parser.add_argument('--out-file', '-o', metavar='eval.json', help='Write accuracy metrics to file (default is stdout).')
    return parser.parse_args()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)

def evaluate(data_file, pred_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    with open(pred_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    exact_scores = {}
    f1_scores = {}

    for qid, gold_answer in ground_truth.items():
        if qid not in predictions:
            print(f"Missing prediction for {qid}")
            continue

        pred_answer = predictions[qid]
        exact_scores[qid] = compute_exact(gold_answer, pred_answer)
        f1_scores[qid] = compute_f1(gold_answer, pred_answer)

    total = len(ground_truth)
    exact_match = 100.0 * sum(exact_scores.values()) / total
    f1 = 100.0 * sum(f1_scores.values()) / total

    return {
        'exact': exact_match,
        'f1': f1,
        'total': total
    }
    
def main():
    args = parse_args()
    metrics = evaluate(args.data_file, args.pred_file)

    if args.out_file:
        with open(args.out_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
