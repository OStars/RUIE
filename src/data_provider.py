import os
import re
import json
import gzip
import random
from datasets import Dataset
from typing import Dict, List

TEMPLATE = "Task: {task}\nSchema: {schema}\nInput: \"{input}\"\nOutput: {output}"

TASK_MP = {
    "NER": "Named entity recognition",
    "RE": "Relation extraction",
    "EET": "Event detection",
    "EEA": "Event argument extraction"
}

with open("data/uie_configs.json", "r") as f:
    UIE_CONFIG = json.load(f)

with open("data/uie_template.json", "r") as f:
    TASK2INSTRUCTION = json.load(f)


def get_answer_by_task(instance: Dict, task_name: str, shuffle=True) -> str:
    if task_name == "NER":
        kv_pairs = []
        for entity in instance['entities']:
            if entity['type'] == 'NA' or entity['type'] == '':
                continue
            kv_pair = [entity['name'], entity['type']]
            kv_pairs.append(kv_pair)

        if len(kv_pairs) > 0:
            if shuffle:
                random.shuffle(kv_pairs)
            answer = "; ".join(["{}: {}".format(v, k) for (k, v) in kv_pairs])
        else:
            answer = "None"
    elif task_name == "RE":
        relation_pairs = []

        for relation in instance['relations']:
            if relation['type'] == 'NA' or relation['type'] == '':
                continue
            relation_pair = [relation['head']['name'],
                             relation['type'], relation['tail']['name']]
            relation_pairs.append(relation_pair)

        if len(relation_pairs) > 0:
            if shuffle:
                random.shuffle(relation_pairs)
            answer = "; ".join("{}: {}, {}".format(r, h, t)
                               for (h, r, t) in relation_pairs)
        else:
            answer = 'None'
    elif task_name == "EET":
        event_pairs = []

        for k, event in enumerate(instance['events']):
            instance['events'][k]['trigger'] = event['trigger']
            instance['events'][k]['type'] = event['type']

            if event['type'] == 'NA' or event['type'] == '':
                continue
            event_type = event['type']
            event_trigger = event['trigger']
            event_pair = [event_type, event_trigger]
            event_pairs.append(event_pair)

        if len(event_pairs) > 0:
            if shuffle:
                random.shuffle(event_pairs)
            answer = "; ".join(["{}: {}".format(type, trigger)
                               for (type, trigger) in event_pairs])
        else:
            answer = 'None'
    elif task_name == "EEA":
        event = instance['events'][0]
        event_arguments = ["{}: {}".format(argument['role'], argument['name']) for
                           argument in event['arguments']]

        if len(event_arguments) > 0:
            if shuffle:
                random.shuffle(event_arguments)
            answer = "; ".join(event_arguments)
        else:
            answer = "None"
    else:
        raise ValueError(f"Unsupported task_name: {task_name}.")

    return answer


def get_labels(label_path: str, task_name: str, instance: Dict = None, shuffle=True) -> str:
    output_labels = ""
    with open(label_path, "r") as f:
        labels = json.load(f)
        if task_name in ["NER", "RE"]:
            if shuffle:
                random.shuffle(labels)
            output_labels = "[{}]".format(", ".join(labels))
        elif task_name == "EET":
            labels = list(labels.keys())
            if shuffle:
                random.shuffle(labels)
            output_labels = "[{}]".format(", ".join(labels))
        elif task_name == "EEA":
            labels = labels[instance['events'][0]['type']]
            if shuffle:
                random.shuffle(labels)
            output_labels = "Given event trigger: \"{}: {}\"; Candidate arguments: [{}]".format(
                instance['events'][0]['type'], instance['events'][0]['trigger'], ", ".join(labels))
        else:
            raise ValueError(f"Unsupported task_name: {task_name}.")

    return output_labels


def get_golden_labels(instance: Dict, task_name: str) -> Dict:
    if task_name == "NER":
        return instance["entities"]
    elif task_name == "RE":
        return instance["relations"]
    elif task_name in ["EE", "EET", "EEA"]:
        return instance["events"]
    else:
        raise ValueError(f"Unsupported task_name: {task_name}.")


def get_corpus(data_base_dir: str, shuffle=False) -> Dataset:
    corpus = []
    global_id = 0

    for root_dir, dirs, files in os.walk(data_base_dir):
        task_name = os.path.basename(root_dir)
        if task_name in UIE_CONFIG:
            dataset_count = 0
            task_corpus = UIE_CONFIG[task_name]["corpus"]
            task_corpus = set([corpus_info["dataset name"]
                              for corpus_info in task_corpus])
            for dataset_name in dirs:
                if dataset_name in task_corpus:
                    label_path = os.path.join(
                        root_dir, dataset_name, "labels.json")
                    data_path = os.path.join(
                        root_dir, dataset_name, "train.json")
                    with open(data_path, "r") as f:
                        dataset = json.load(f)

                    for instance in dataset:
                        labels = get_labels(
                            label_path, task_name, instance, shuffle=shuffle)
                        answer = get_answer_by_task(
                            instance, task_name, shuffle=shuffle)
                        text = instance["sentence"].replace("\n\n", "\n")
                        content = TEMPLATE.format(
                            task=TASK_MP[task_name], schema=labels, input=text, output=answer)
                        corpus.append({
                            "id": global_id,
                            "task_name": task_name + "_" + dataset_name,
                            "text": text,
                            "contents": content,
                            "instruction": TASK2INSTRUCTION[task_name],
                            "golden_labels": str(get_golden_labels(instance, task_name))
                        })
                        global_id += 1
                    dataset_count += 1
            print(f"{task_name} task processed {dataset_count} datasets.")

    return Dataset.from_list(corpus)


def get_dataset_by_split(data_base_dir: str, split="train", is_sample=True, shuffle=True) -> Dataset:
    random.seed(1234)
    output_datset = []

    for root_dir, dirs, files in os.walk(data_base_dir):
        task_name = os.path.basename(root_dir)
        if task_name in UIE_CONFIG.keys():
            dataset_count = 0
            task_corpus = UIE_CONFIG[task_name]["train"] if split == "train" else UIE_CONFIG[task_name]["corpus"]
            task_corpus = set([corpus_info["dataset name"]
                              for corpus_info in task_corpus])
            for dataset_name in dirs:
                if dataset_name in task_corpus:
                    label_path = os.path.join(
                        root_dir, dataset_name, "labels.json")
                    data_path = os.path.join(
                        root_dir, dataset_name, split + ".json")
                    with open(data_path, "r") as f:
                        dataset = json.load(f)

                    sample_idxs = range(len(dataset))
                    if is_sample and split.lower() != "test" and len(dataset) > 10000:
                        sample_idxs = random.sample(sample_idxs, k=10000)

                    for idx in sample_idxs:
                        instance = dataset[idx]
                        labels = get_labels(
                            label_path, task_name, instance, shuffle)
                        answer = get_answer_by_task(
                            instance, task_name, shuffle)
                        text = instance["sentence"]
                        content = TEMPLATE.format(
                            task=TASK_MP[task_name], schema=labels, input=text, output="").strip()
                        output_datset.append({
                            "text": text,
                            "query": content,
                            "answers": [answer],
                            "options": [],
                            "instruction": TASK2INSTRUCTION[task_name],
                            "query_id": f"{task_name}_{dataset_name}_{split}_{idx}",
                            "task_name": task_name + "_" + dataset_name,
                            "golden_labels": str(get_golden_labels(instance, task_name))
                        })
                    dataset_count += 1
            print(f"{task_name} task processed {dataset_count} datasets.")

    return Dataset.from_list(output_datset)


overlap_or_nest_cnt = 0


def merge_section(sentence: str, keywords: list[dict]):
    if not keywords:
        return keywords

    pos2keywords = {str(keyword["pos"]): keyword for keyword in keywords}

    merged_pos = []
    l = keywords[0]["pos"][0]
    r = keywords[0]["pos"][1]
    for keyword in keywords:
        if keyword["pos"][0] < r:
            l = min(l, keyword["pos"][0])
            r = max(r, keyword["pos"][1])
        else:
            merged_pos.append([l, r])
            l = keyword["pos"][0]
            r = keyword["pos"][1]
    merged_pos.append([l, r])

    merged_keywords = []
    for pos in merged_pos:
        if str(pos) not in pos2keywords:
            global overlap_or_nest_cnt
            overlap_or_nest_cnt += 1
            # print(f"There are overlapping keywords in sentence: {sentence}.")
        merged_keywords.append({
            "name": sentence[pos[0]:pos[1]],
            "type": pos2keywords[str(pos)]["type"] if str(pos) in pos2keywords else "MERGED_KEYWORD",
            "pos": pos
        })

    return merged_keywords


def add_keywords_to_sentence_new_strategy(sentence, keywords):
    """
    Adds <Keyword> </Keyword> tags around the keywords in the given sentence using a new strategy.

    Args:
    sentence (str): The input sentence.
    keywords (list): A list of dictionaries with 'name', 'type', and 'pos' keys.

    Returns:
    str: The modified sentence with <Keyword> tags.
    """
    # Sort keywords by start position in ascending order
    sorted_keywords = sorted(
        keywords, key=lambda x: (x['pos'][0], x["pos"][1]))

    sorted_keywords = merge_section(sentence, sorted_keywords)

    # Create a list to hold parts of the sentence and keywords
    parts = []

    # Previous end position, initially set to 0
    prev_end = 0

    for keyword in sorted_keywords:
        start, end = keyword['pos']
        # Append the text before the current keyword
        parts.append(sentence[prev_end:start])
        # Append the keyword wrapped in <Keyword> tags
        parts.append(f'<Keyword> {sentence[start:end]} </Keyword>')
        # Update the previous end position
        prev_end = end

    # Append the remaining text after the last keyword
    parts.append(sentence[prev_end:])

    # Join all parts to form the modified sentence
    modified_sentence = ''.join(parts)

    return modified_sentence


def augment_with_keyword(examples: Dict[str, List]) -> Dict[str, List]:
    aug_texts = []
    aug_queries = []

    for text, query, task_name, golden_labels in zip(examples["text"], examples["query"], examples["task_name"], examples["golden_labels"]):
        text = text + " "

        vis = set()
        keywords = []
        if task_name.startswith("NER"):
            for keyword in eval(golden_labels):
                keyword_pos = keyword["pos"]
                if not keyword_pos or text[keyword_pos[0]:keyword_pos[1]] != keyword["name"] or text[keyword_pos[1]].isalpha():
                    match = re.compile(r"\b({})\b".format(
                        re.escape(keyword["name"]))).search(text)
                    if match is not None:
                        start, end = match.span()
                        if "{}:{}".format(start, end) not in vis:
                            vis.add("{}:{}".format(start, end))
                            keyword["pos"] = [start, end]
                            keywords.append(
                                {"name": keyword["name"], "type": keyword["type"] if "type" in keyword else "", "pos": keyword["pos"]})
                else:
                    if "{}:{}".format(keyword_pos[0], keyword_pos[1]) not in vis:
                        vis.add("{}:{}".format(keyword_pos[0], keyword_pos[1]))
                        keywords.append(
                            {"name": keyword["name"], "type": keyword["type"] if "type" in keyword else "", "pos": keyword["pos"]})
        elif task_name.startswith("RE"):
            for relation in eval(golden_labels):
                for keyword in [relation["head"], relation["tail"]]:
                    keyword_pos = keyword["pos"]
                    if not keyword_pos or text[keyword_pos[0]:keyword_pos[1]] != keyword["name"] or text[keyword_pos[1]].isalpha():
                        match = re.compile(r"\b({})\b".format(
                            re.escape(keyword["name"]))).search(text)
                        if match is not None:
                            start, end = match.span()
                            if "{}:{}".format(start, end) not in vis:
                                vis.add("{}:{}".format(start, end))
                                keyword["pos"] = [start, end]
                                keywords.append(
                                    {"name": keyword["name"], "type": keyword["type"] if "type" in keyword else "", "pos": keyword["pos"]})
                    else:
                        if "{}:{}".format(keyword_pos[0], keyword_pos[1]) not in vis:
                            vis.add("{}:{}".format(
                                keyword_pos[0], keyword_pos[1]))
                            keywords.append(
                                {"name": keyword["name"], "type": keyword["type"] if "type" in keyword else "", "pos": keyword["pos"]})
        elif task_name.startswith("EET") or task_name.startswith("EEA"):
            for event in eval(golden_labels):
                if event["trigger"]:
                    trigger_pos = event["pos"]
                    if not trigger_pos or text[trigger_pos[0]:trigger_pos[1]] != event["trigger"] or text[trigger_pos[1]].isalpha():
                        match = re.compile(r"\b({})\b".format(
                            re.escape(event["trigger"]))).search(text)
                        if match is not None:
                            trigger_start, trigger_end = match.span()
                            if "{}:{}".format(trigger_start, trigger_end) not in vis:
                                vis.add("{}:{}".format(
                                    trigger_start, trigger_end))
                                keywords.append({"name": event["trigger"], "type": event["type"], "pos": [
                                                trigger_start, trigger_end]})
                    else:
                        if "{}:{}".format(trigger_pos[0], trigger_pos[1]) not in vis:
                            vis.add("{}:{}".format(
                                trigger_pos[0], trigger_pos[1]))
                            keywords.append(
                                {"name": event["trigger"], "type": event["type"], "pos": trigger_pos})

                for argument in event["arguments"]:
                    argument_pos = argument["pos"]
                    if not argument_pos or text[argument_pos[0]:argument_pos[1]] != argument["name"] or text[argument_pos[1]].isalpha():
                        match = re.compile(r"\b({})\b".format(
                            re.escape(argument["name"]))).search(text)
                        if match is not None:
                            argument_start, argument_end = match.span()
                            if "{}:{}".format(argument_start, argument_end) not in vis:
                                vis.add("{}:{}".format(
                                    argument_start, argument_end))
                                keywords.append({"name": argument["name"], "type": argument["role"], "pos": [
                                                argument_start, argument_end]})
                    else:
                        if "{}:{}".format(argument_pos[0], argument_pos[1]) not in vis:
                            vis.add("{}:{}".format(
                                argument_pos[0], argument_pos[1]))
                            keywords.append(
                                {"name": argument["name"], "type": argument["role"], "pos": argument_pos})

        text = text[:-1]
        aug_text = add_keywords_to_sentence_new_strategy(text, keywords)
        aug_texts.append(aug_text)
        aug_queries.append("{}\nInput: {}\nOutput:".format(
            query.split("\nInput:")[0], aug_text))
    examples["original_text"] = examples["text"]
    examples["original_query"] = examples["query"]
    examples["text"] = aug_texts
    examples["query"] = aug_queries

    return examples


def augment_corpus_with_keyword(examples: Dict[str, List]) -> Dict[str, List]:
    aug_texts = []
    aug_contents = []

    for text, contents, task_name, golden_labels in zip(examples["text"], examples["contents"], examples["task_name"], examples["golden_labels"]):
        text = text + " "

        vis = set()
        keywords = []
        if task_name.startswith("NER"):
            for keyword in eval(golden_labels):
                keyword_pos = keyword["pos"]
                if not keyword_pos or text[keyword_pos[0]:keyword_pos[1]] != keyword["name"] or text[keyword_pos[1]].isalpha():
                    match = re.compile(r"\b({})\b".format(
                        re.escape(keyword["name"]))).search(text)
                    if match is not None:
                        start, end = match.span()
                        if "{}:{}".format(start, end) not in vis:
                            vis.add("{}:{}".format(start, end))
                            keyword["pos"] = [start, end]
                            keywords.append(
                                {"name": keyword["name"], "type": keyword["type"] if "type" in keyword else "", "pos": keyword["pos"]})
                else:
                    if "{}:{}".format(keyword_pos[0], keyword_pos[1]) not in vis:
                        vis.add("{}:{}".format(keyword_pos[0], keyword_pos[1]))
                        keywords.append(
                            {"name": keyword["name"], "type": keyword["type"] if "type" in keyword else "", "pos": keyword["pos"]})
        elif task_name.startswith("RE"):
            for relation in eval(golden_labels):
                for keyword in [relation["head"], relation["tail"]]:
                    keyword_pos = keyword["pos"]
                    if not keyword_pos or text[keyword_pos[0]:keyword_pos[1]] != keyword["name"] or text[keyword_pos[1]].isalpha():
                        match = re.compile(r"\b({})\b".format(
                            re.escape(keyword["name"]))).search(text)
                        if match is not None:
                            start, end = match.span()
                            if "{}:{}".format(start, end) not in vis:
                                vis.add("{}:{}".format(start, end))
                                keyword["pos"] = [start, end]
                                keywords.append(
                                    {"name": keyword["name"], "type": keyword["type"] if "type" in keyword else "", "pos": keyword["pos"]})
                    else:
                        if "{}:{}".format(keyword_pos[0], keyword_pos[1]) not in vis:
                            vis.add("{}:{}".format(
                                keyword_pos[0], keyword_pos[1]))
                            keywords.append(
                                {"name": keyword["name"], "type": keyword["type"] if "type" in keyword else "", "pos": keyword["pos"]})
        elif task_name.startswith("EET") or task_name.startswith("EEA"):
            for event in eval(golden_labels):
                if event["trigger"]:
                    trigger_pos = event["pos"]
                    if not trigger_pos or text[trigger_pos[0]:trigger_pos[1]] != event["trigger"] or text[trigger_pos[1]].isalpha():
                        match = re.compile(r"\b({})\b".format(
                            re.escape(event["trigger"]))).search(text)
                        if match is not None:
                            trigger_start, trigger_end = match.span()
                            if "{}:{}".format(trigger_start, trigger_end) not in vis:
                                vis.add("{}:{}".format(
                                    trigger_start, trigger_end))
                                keywords.append({"name": event["trigger"], "type": event["type"], "pos": [
                                                trigger_start, trigger_end]})
                    else:
                        if "{}:{}".format(trigger_pos[0], trigger_pos[1]) not in vis:
                            vis.add("{}:{}".format(
                                trigger_pos[0], trigger_pos[1]))
                            keywords.append(
                                {"name": event["trigger"], "type": event["type"], "pos": trigger_pos})

                for argument in event["arguments"]:
                    argument_pos = argument["pos"]
                    if not argument_pos or text[argument_pos[0]:argument_pos[1]] != argument["name"] or text[argument_pos[1]].isalpha():
                        match = re.compile(r"\b({})\b".format(
                            re.escape(argument["name"]))).search(text)
                        if match is not None:
                            argument_start, argument_end = match.span()
                            if "{}:{}".format(argument_start, argument_end) not in vis:
                                vis.add("{}:{}".format(
                                    argument_start, argument_end))
                                keywords.append({"name": argument["name"], "type": argument["role"], "pos": [
                                                argument_start, argument_end]})
                    else:
                        if "{}:{}".format(argument_pos[0], argument_pos[1]) not in vis:
                            vis.add("{}:{}".format(
                                argument_pos[0], argument_pos[1]))
                            keywords.append(
                                {"name": argument["name"], "type": argument["role"], "pos": argument_pos})

        text = text[:-1]
        aug_text = add_keywords_to_sentence_new_strategy(text, keywords)
        aug_texts.append(aug_text)
        aug_contents.append("{}\nInput: {}\nOutput: {}".format(contents.split(
            "\nInput:")[0].strip(), aug_text, contents.split("\nOutput:")[-1].strip()))
    examples["original_text"] = examples["text"]
    examples["original_contents"] = examples["contents"]
    examples["text"] = aug_texts
    examples["contents"] = aug_contents

    return examples


if __name__ == "__main__":
    corpus = get_corpus(
        "IE_INSTRUCTIONS/", shuffle=False)
    corpus = corpus.map(augment_corpus_with_keyword, batched=True, num_proc=4)

    train_dataset = get_dataset_by_split(
        "IE_INSTRUCTIONS/", "train", is_sample=True, shuffle=False)
    train_dataset = train_dataset.map(
        augment_with_keyword, batched=True, num_proc=4)

    test_dataset = get_dataset_by_split(
        "IE_INSTRUCTIONS/", "test", is_sample=False, shuffle=False)

    with gzip.open(os.path.join("data/", "passages.jsonl.gz"), "wt", encoding='utf-8', compresslevel=9) as f:
        for idx in range(len(corpus)):
            f.write(json.dumps(
                corpus[idx], ensure_ascii=False, separators=(',', ':')))
            f.write('\n')

    with gzip.open(os.path.join("data/", "train.jsonl.gz"), "wt", encoding='utf-8', compresslevel=9) as f:
        for idx in range(len(train_dataset)):
            f.write(json.dumps(
                train_dataset[idx], ensure_ascii=False, separators=(',', ':')))
            f.write('\n')

    with gzip.open(os.path.join("data/", "test.jsonl.gz"), "wt", encoding='utf-8', compresslevel=9) as f:
        for idx in range(len(test_dataset)):
            f.write(json.dumps(
                test_dataset[idx], ensure_ascii=False, separators=(',', ':')))
            f.write('\n')
