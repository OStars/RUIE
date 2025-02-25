import os
import json
import jsonlines
import argparse
from metrics import *
from collections import defaultdict

def calculate_f1(prediction_dir, task):
    EvaluatorDict = {
        'RE':EvaluatorRE,
        'NER':EvaluatorNER,
        'EET':EvaluatorEET,
        'EEA':EvaluatorEEA
    }
    task_dict = dict()
    for prediction_file in os.listdir(prediction_dir):
        task_path = os.path.join(prediction_dir, prediction_file)
        report_dir_root = os.path.join(prediction_dir, 'report')
        with open(task_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                dataset_name = prediction_file.split("_decoding")[0]
                if task not in task_dict:
                    task_dict[task] = dict()
                if dataset_name not in task_dict[task]:
                    task_dict[task][dataset_name] = EvaluatorDict[task]()
                task_dict[task][dataset_name].add(data, data['decoded_text'].split("\n\n")[-1])

    # export report
    if not os.path.exists(report_dir_root):
        os.mkdir(report_dir_root)

    # export tsv
    for task_name, eval_dict in task_dict.items():
        print('\n'+'-'*16+task_name+'-'*16+'\n')
        rows = []
        scores = []
        report_dir = os.path.join(report_dir_root, task_name)
        if not os.path.exists(report_dir):
            os.mkdir(report_dir)
        for dataset_name, evaluator in eval_dict.items():
            evaluator.dump_audit_report(os.path.join(report_dir, dataset_name+'.json'))
            rows.append((dataset_name, *evaluator.get_metric()))
            scores.append(evaluator.get_metric())
        rows = sorted(rows, key=lambda x: x[0].lower())
        if len(scores) == 0:
            continue
        rows.append(('Average', *[sum([score[i] for score in scores])/len(scores) for i in range(len(scores[0]))]))
        with open(os.path.join(report_dir_root, 'report_%s.tsv'%task_name), 'w', encoding='utf-8') as f:
            if len(rows[0]) == 4:
                f.write('%-48s\t%-8s\t%-8s\t%-8s\n'%("dataset_name", "f1", "recall", "precision"))
                print('%-48s\t%-8s\t%-8s\t%-8s'%("dataset_name", "f1", "recall", "precision"))
                for row in rows:
                    f.write('%-48s\t%-8g\t%-8g\t%-8g\n'%row)
                    print('%-48s\t%-8g\t%-8g\t%-8g'%row)
            else:
                f.write('%-48s\t%-8s\n'%("dataset_name", "f1"))
                for row in rows:
                    f.write('%-48s\t%-8g\n'%row)
                    print('%-48s\t%-8g'%row)


def merge_predictions(path: str):
    prefix2file = defaultdict(set)
    for fn in os.listdir(path):
        fp = os.path.join(path, fn)
        prefix = "_".join(fn.split("_")[:2])
        if os.path.isfile(fp) and fn != prefix + "_decoding_results.jsonl":
            prefix2file["_".join(fn.split("_")[:2])].add(fp)
    
    for prefix, files in prefix2file.items():
        if len(files) > 1:
            merged_preds = []
            for fp in files:
                with open(fp, "r") as f:
                    for line in f:
                        merged_preds.append(json.loads(line))
            with jsonlines.open(os.path.join(path, prefix + "_decoding_results.jsonl"), "w") as f:
                for pred in merged_preds:
                    f.write(pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["NER", "RE", "EE", "EET", "EEA"])
    args = parser.parse_args()

    os.environ['RANDOM_RECORD'] = '1'

    if args.task in ["EET", "EEA"]:
        merge_predictions(args.prediction_dir)
    calculate_f1(args.prediction_dir, args.task)