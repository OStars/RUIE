import re
import os
import json
import torch
import random
import numpy as np

from typing import List


@torch.no_grad()
def accuracy(output: torch.tensor, target: torch.tensor, topk=(1,)) -> List[float]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


@torch.no_grad()
def batch_mrr(output: torch.tensor, target: torch.tensor) -> float:
    assert len(output.shape) == 2
    assert len(target.shape) == 1
    sorted_score, sorted_indices = torch.sort(output, dim=-1, descending=True)
    _, rank = torch.nonzero(sorted_indices.eq(target.unsqueeze(-1)).long(), as_tuple=True)
    assert rank.shape[0] == output.shape[0]

    rank = rank + 1
    mrr = torch.sum(100 / rank.float()) / rank.shape[0]
    return mrr.item()


class MetricBase:
    def __init__(self):
        raise NotImplementedError()
    def update(self, y_truth, y_pred):
        raise NotImplementedError()
    def get_metric(self, detail=False):
        raise NotImplementedError()
    def get_last(self):
        raise NotImplementedError()


class MetricF1(MetricBase):
    def __init__(self):
        self.sum_TP = 0
        self.sum_FN = 0
        self.sum_FP = 0
        self.last_TP = None
        self.last_FN = None
        self.last_FP = None

    def update(self, y_truth: set, y_pred: set):
        self.last_TP = len(y_truth & y_pred)
        self.last_FN = len(y_truth - y_pred)
        self.last_FP = len(y_pred - y_truth)
        self.sum_TP += self.last_TP
        self.sum_FN += self.last_FN
        self.sum_FP += self.last_FP

    def get_metric(self, detail=False):
        TP = self.sum_TP
        FN = self.sum_FN
        FP = self.sum_FP
        if TP + FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)
        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
        if recall + precision == 0:
            f1 = 0
        else:
            f1 = 2 * recall * precision / (recall + precision)
        self.recall = recall
        self.precision = precision
        if detail:
            return f1, recall, precision
        return (f1, )

    def get_detail(self):
        if not hasattr(self, 'recall'):
            f1 = self.get_metric(False)
        return f1, self.recall, self.precision

    def get_last(self):
        return self.last_TP, self.last_FN, self.last_FP


class MetricF1NA(MetricF1):
    "对于RE中关系类型为NA的特殊处理"
    def update(self, y_truth: set, y_pred: set):
        self.last_TP = 0
        self.last_FN = 0
        self.last_FP = 0
        for truth in y_truth:
            if ',na,' in truth:
                pattern = re.escape(truth).replace(',na,', ',(.+),')
                pattern = re.compile(pattern)
                pred_fail = False
                for pred in y_pred:
                    match = pattern.match(pred)
                    if match is not None and match.group(1) != 'na':
                        pred_fail = True
                        break
                if not pred_fail:
                    self.last_TP += 1
            else:
                if truth in y_pred:
                    self.last_TP += 1
                else:
                    self.last_FN += 1
        for pred in y_pred:
            if ',na,' in pred:
                pattern = re.escape(pred).replace(',na,', ',(.+),')
                pattern = re.compile(pattern)
                pred_fail = False
                for truth in y_truth:
                    match = pattern.match(truth)
                    if match is not None and match.group(1) != 'na':    # pred: (A,NA,B); truth:(A,notNA,B)
                        pred_fail = True
                        break
                if pred_fail:
                    self.last_FP += 1
                else:
                    self.last_TP += 0
            else:
                if pred not in y_truth:
                    self.last_FP += 1
        self.sum_TP += self.last_TP
        self.sum_FN += self.last_FN
        self.sum_FP += self.last_FP


class AuditBase:
    def __init__(self, record_limit=16):
        # record_limit: maximum size of record, `-1` for infinite, `0` for no record
        self.record_limit = record_limit
        self.cnt = 0
        self.record = []

    def _check(self, last) -> bool:
        # must be overrided
        # return whether be recorded or not
        raise NotImplementedError()

    def _add_record(self, new_record):
        self.cnt += 1
        if self.record_limit < 0 or len(self.record) < self.record_limit:
            # record limit check
            self.record.append(new_record)
        elif os.environ.get('RANDOM_RECORD')=='1':
            if random.randint(1,self.cnt) <= self.record_limit:
                idx = random.randint(0,len(self.record)-1)
                self.record[idx] = new_record

    def update(self, last):
        if self._check(last):
            new_record = {
                'input_text': last['json_data']['input_text'],
                'query': last['json_data']['query'],
                'decoded_text': last['json_data']['decoded_text'],
                'predict': last['predict'],
                'y_truth': last['y_truth'],
                'y_pred': last['y_pred']
            }
            new_record = self._to_json_object(new_record)
            self._add_record(new_record)

    @staticmethod
    def _to_json_object(obj):
        if isinstance(obj, str) or isinstance(obj, int) or isinstance(obj, float):
            return obj
        if isinstance(obj, tuple) or isinstance(obj, list) or isinstance(obj, set):
            return [AuditBase._to_json_object(x) for x in obj]
        if isinstance(obj, dict):
            return {AuditBase._to_json_object(k): AuditBase._to_json_object(v) for k, v in obj.items()}
        else:
            raise NotImplementedError()

    def get_cnt(self):
        return self.cnt

    def get_record(self):
        return self.record

    def get_report(self):
        return {
            'count': self.cnt,
            'record': self.record
        }

    def get_name(self):
        return self.__class__.__name__

class AuditVoid(AuditBase):
    "检测空输出"
    def _check(self, last) -> bool:
        return last['predict'].strip() == ''

class AuditLong(AuditBase):
    "检测过长的输出"
    def _check(self, last) -> bool:
        return len(last['predict']) >= 512

class AuditInsane(AuditBase):
    "检测胡言乱语"
    def _check(self, last) -> bool:
        return last['predict'].strip().lower() not in {'na', 'no relation', 'none', '[]', ''} and len(last['y_pred']) == 0

class AuditBothEmpty(AuditBase):
    "检测Label和predict都为空的条目"
    def _check(self, last) -> bool:
        return len(last['y_truth']) == 0 and len(last['y_pred']) == 0

class AuditLabelEmptyOnly(AuditBase):
    "检测label为空，但predict不为空"
    def _check(self, last) -> bool:
        return len(last['y_truth']) == 0 and len(last['y_pred']) != 0

class AuditPredEmptyOnly(AuditBase):
    "检测predict为空，label不为空"
    def _check(self, last) -> bool:
        return len(last['y_truth']) != 0 and len(last['y_pred']) == 0
    
class AuditNA(AuditBase):
    "检测包含类型为NA的输出，目前只用于RE"
    def _check(self, last) -> bool:
        for i in last['y_pred']:    # assert isinstance(i, str)
            if ',na,' in i:
                return True
        return False

class AuditInvalid(AuditBase):
    "检测包含非法标签类型的输出，目前只用于RE和NER"
    def _check(self, last) -> bool:
        json_data = last['json_data']
        labels_str = json_data["query"][json_data["query"].find("[")+1: json_data["query"].find("]")]
        valid_labels = [EvaluatorBase._format(label.strip()) for label in labels_str.split(",")]
        if len(valid_labels) == 0:
            return False
        valid_labels = set(valid_labels)

        for pred in last['y_pred']:
            pred = pred.split(':')
            if len(pred) >= 2:
                label = pred[0]
                if label not in valid_labels:
                    return True
        return False

class AuditFidelity(AuditBase):
    "检测不来源于句子的实体，目前只用于RE和NER"
    def _check(self, last) -> bool:
        for item in last['y_pred']:
            item = item.split(':')
            if len(item) < 2:
                continue
            ents = item[-1].split(',')
            for ent in ents:
                if EvaluatorBase._format(ent) not in EvaluatorBase._format(last['json_data']['text']):
                    return True
            return False

class AuditGoldenlabelFault(AuditBase):
    "golden label中的三元组有空缺，目前只用于RE"
    def _check(self, last) -> bool:
        for item in last['y_truth']:
            cnt = 0
            if len(item.split(':')) < 2:
                continue
            for i in item.split(':')[-1].split(','):
                i = i.strip()
                if i != '':
                    cnt += 1
            if cnt <= 1:
                return True
        return False

class AuditRepeat(AuditBase):
    "检测复读机"
    def _check(self, last) -> bool:
        pattern = r'(\w{5,})\1{2,}'
        match = re.search(pattern, last['predict'])
        return match is not None

class AuditRetard(AuditBase):
    "检测二者都非空前提下的错误"
    def _check(self, last) -> bool:
        last_metric = last['metric']
        if hasattr(last_metric, 'last_TP'):
            if len(last['y_pred']) != 0 and len(last['y_truth']) != 0:
                return last_metric.last_TP == 0
        if hasattr(last_metric, 'scores'):
            return last_metric.scores[-1] == 0
        return False
        
class AuditWhatever(AuditBase):
    "无差别逮捕"
    def _check(self, last) -> bool:
        return True
    
    def update(self, last):
        self.cnt += 1
    
class EvaluatorBase:
    def __init__(self):
        self.last = dict()
        self._init_audit()
        self._init_metric()
    
    def _init_metric(self):
        # must be overrided to init self.metric
        self.metric = MetricBase()

    def _init_audit(self):
        # override if necessary
        self.audit = [
            AuditVoid(),
            AuditBothEmpty(),
            AuditLabelEmptyOnly(),
            AuditPredEmptyOnly(),
            AuditLong(),
            AuditInsane(),
            AuditRepeat(),
            AuditRetard(),
            AuditWhatever()
        ]
    
    def _update_audit(self):
        # override if necessary
        for audit in self.audit:
            audit.update(self.last)

    def _extract(self, json_data, predict: str):
        # must be overrided
        # return: y_truth, y_pred
        raise NotImplementedError()

    def add(self, json_data, predict):
        assert isinstance(json_data, list) == isinstance(predict, list)

        if isinstance(json_data, list) and isinstance(predict, list):
            for i, j in zip(json_data, predict):
                self.add(i, j)
                return

        y_truth, y_pred = self._extract(json_data, predict)
        self.metric.update(y_truth, y_pred)

        # audit
        self.last['json_data'] = json_data
        self.last['predict'] = predict
        self.last['y_truth'] = y_truth
        self.last['y_pred'] = y_pred
        self.last['metric'] = self.metric

        self._update_audit()
    
    def get_metric(self, detail=True) -> float:
        return self.metric.get_metric(detail)

    def get_audit_report(self):
        return {
            a.get_name() : a.get_report()
            for a in self.audit
        }

    def dump_audit_report(self, fpath):
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(self.get_audit_report(), f, indent=4, ensure_ascii=False)

    @staticmethod
    def _remove_redundant_space(s):
        # '   a  b  \t  c  \n' --> 'a b c'
        #'  kjc,  jns , ((  : ()  )  ( . )( ln  kc  a,,  ' --> 'kjc,jns,((:())(.)(ln kc a,,'
        s = ' '.join(s.split())
        s = re.sub(r"\s*(,|:|\(|\)|\.|_|;|'|-)\s*", r'\1', s)
        return s
    
    @staticmethod
    def _format(s):
        s = EvaluatorBase._remove_redundant_space(s)
        s = s.lower()
        s = s.replace('{','').replace('}','')
        s = re.sub(',+', ',', s)
        s = re.sub('\.+', '.', s)
        s = re.sub(';+', ';', s)
        s = s.replace('’', "'")
        s = s.replace('location', 'located')
        return s


class EvaluatorNER(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricF1()

    def _init_audit(self):
        super()._init_audit()    
        self.audit += [
            AuditInvalid(),
            AuditFidelity()
        ]

    def _extract(self, json_data, predict):
        # person: a; person: b; org: c
        entity_truth = set()
        for ent in self._format(json_data['answer_text']).split(';'):
            ent = self._format(ent)
            entity_truth.add(ent)
        
        if self._format(json_data["answer_text"]) == "none":
            entity_truth = set()
        
        schema = json_data['query'][json_data['query'].find("[")+1:json_data['query'].find("]")]
        schema = set(s.strip().lower() if "location" not in s.strip().lower() else s.strip().lower().replace("location", "located") for s in schema.split(","))
        schema = sorted(list(schema), key=lambda x: len(x), reverse=True)
        entity_pred = set()
        pattern = re.compile(r"(" + "|".join(r"{}:[^;\n]+".format(s) for s in schema) + ")")
        for ent in pattern.findall(predict.lower().replace("location", "located")):
            ent = self._format(ent)
            if ent and ent.split(":")[0] in schema and ent.split(":")[1] in json_data["query"].lower():
                entity_pred.add(ent)
        return entity_truth, entity_pred

class EvaluatorRE(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricF1NA()

    def _init_audit(self):
        super()._init_audit()    
        self.audit += [
            AuditInvalid(),
            AuditFidelity(),
            AuditGoldenlabelFault(),
            AuditNA()
        ]

    def _extract(self, json_data, predict):
        y_truth = set()
        for rel in self._format(json_data['answer_text']).split(';'):
            elem = self._format(rel)
            if ':' not in elem:
                continue
            y_truth.add(elem)

        schema = json_data['query'][json_data['query'].find("[")+1:json_data['query'].find("]")]
        schema = set(s.strip().lower() if "location" not in s.strip().lower() else s.strip().lower().replace("location", "located") for s in schema.split(","))
        schema = sorted(list(schema), key=lambda x: len(x), reverse=True)
        y_pred = set()
        pattern = re.compile(r"(" + "|".join(r"{}:[^;\n]+".format(s) for s in schema) + ")")
        for rel in pattern.findall(predict.lower().replace("location", "located")):
            elem = self._format(rel)
            if ':' not in elem:
                continue
            y_pred.add(elem)
        return y_truth, y_pred


class EvaluatorEET(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricF1()
    def _extract(self, json_data, predict: str):
        y_truth = set()
        for item in self._format(json_data['answer_text']).split(';'):
            if ':' not in item:
                continue
            y_truth.add(self._format(item))

        schema = json_data['query'][json_data['query'].find("[")+1:json_data['query'].find("]")]
        schema = set(s.strip().lower() if "location" not in s.strip().lower() else s.strip().lower().replace("location", "located") for s in schema.split(","))
        schema = sorted(list(schema), key=lambda x: len(x), reverse=True)

        y_pred = set()
        pattern = re.compile(r"(" + "|".join(r"{}:[^;\n]+".format(s) for s in schema) + ")")
        for trigger in pattern.findall(predict.lower().replace("location", "located")):
            if trigger:
                y_pred.add(self._format(trigger))
        return y_truth, y_pred

class EvaluatorEEA(EvaluatorBase):
    def _init_metric(self):
        self.metric = MetricF1()

    def _extract(self, json_data, predict: str):
        y_truth = set()
        for item in json_data['answer_text'].split(';'):
            if ':' not in item:
                continue
            y_truth.add(self._format(item))

        schema = json_data['query'][json_data['query'].find("[")+1:json_data['query'].find("]")]
        schema = set(s.strip().lower() if "location" not in s.strip().lower() else s.strip().lower().replace("location", "located") for s in schema.split(","))
        schema = sorted(list(schema), key=lambda x: len(x), reverse=True)
        
        y_pred = set()
        pattern = re.compile(r"(" + "|".join(r"{}:[^;\n]+".format(s) for s in schema) + ")")
        for item in pattern.findall(predict.lower().replace("location", "located")):
            if item and item.split(':')[0] in schema:
                y_pred.add(self._format(item))
        
        return y_truth, y_pred