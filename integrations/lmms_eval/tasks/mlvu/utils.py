import ast
import os
import random
import re
import sys
from pathlib import Path

import yaml
from loguru import logger as eval_logger

hf_home = os.getenv("HF_HOME", "./~/.cache/huggingface")
# hf_home="/share/junjie/shuyan/lmms-eval/~/.cache/huggingface"
base_cache_dir = os.path.expanduser(hf_home)


with open(Path(__file__).parent / "mlvu_dev.yaml", "r") as f:
    raw_data_dev = f.readlines()
    safe_data_dev = []
    for i, line in enumerate(raw_data_dev):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_dev.append(line)
#cache_name_dev = yaml.safe_load("".join(safe_data_dev))["dataset_kwargs"]["cache_dir"]
#cache_dir_dev = os.path.join(base_cache_dir, cache_name_dev)
cache_dir_dev = os.environ.get("OTFS_MLVU_VIDEO_DIR", "./data/mlvu/videos")


with open(Path(__file__).parent / "mlvu_test.yaml", "r") as f:
    raw_data_test = f.readlines()
    safe_data_test = []
    for i, line in enumerate(raw_data_test):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_test.append(line)
cache_name_test = yaml.safe_load("".join(safe_data_test))["dataset_kwargs"]["cache_dir"]
cache_dir_test = os.path.join(base_cache_dir, cache_name_test)


def mlvu_doc_to_visual_dev(doc):
    video_path = doc["video"]
    video_path = os.path.join(cache_dir_dev, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def mlvu_doc_to_visual_test(doc):
    video_path = doc["video_name"]
    video_path = os.path.join(cache_dir_test, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def _get_doc_candidates(doc):
    candidates = doc.get("candidates")
    if isinstance(candidates, list) and candidates:
        return [str(candidate) for candidate in candidates]

    options = doc.get("options")
    if isinstance(options, dict) and options:
        ordered_keys = sorted(options.keys())
        return [str(options[key]) for key in ordered_keys]
    if isinstance(options, list) and options:
        return [str(option) for option in options]
    if isinstance(options, str):
        try:
            parsed_options = ast.literal_eval(options)
        except (ValueError, SyntaxError):
            parsed_options = None
        if isinstance(parsed_options, dict) and parsed_options:
            ordered_keys = sorted(parsed_options.keys())
            return [str(parsed_options[key]) for key in ordered_keys]
        if isinstance(parsed_options, list) and parsed_options:
            return [str(option) for option in parsed_options]

    candidates = []
    for i in range(5):
        option = doc.get(f"option{i}")
        if option is None or option == "N/A":
            break
        candidates.append(str(option))
    return candidates


def get_multi_choice_info(options):
    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    if response is None:
        response = ""
    response = str(response)
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []

    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        return random.choice(all_choices) if all_choices else ""

    if len(candidates) == 1:
        return candidates[0]

    start_indexes = []
    if index_ans:
        if ans_with_brack:
            for can in candidates:
                start_indexes.append(response.rfind(f"({can})"))
        else:
            for can in candidates:
                start_indexes.append(response.rfind(f" {can} "))
    else:
        for can in candidates:
            start_indexes.append(response.lower().rfind(index2ans[can].lower()))

    best_idx = max(range(len(candidates)), key=lambda idx: start_indexes[idx])
    return candidates[best_idx]


def _normalize_text(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _normalize_answer_to_choice(answer, index2ans):
    """Map a gold option text or an explicit choice label to its letter."""
    if answer is None:
        return ""

    answer_text = str(answer).strip()
    # Match option text first: the article "a" is not a choice label.
    normalized_answer = _normalize_text(answer_text)
    for choice, option in index2ans.items():
        if normalized_answer == _normalize_text(option):
            return choice

    match = re.fullmatch(r"(?:\(([A-Z])\)|([A-Z])[.)]?)", answer_text.upper())
    if match:
        choice = match.group(1) or match.group(2)
        if choice in index2ans:
            return choice

    return answer_text


def mlvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    question = doc["question"]
    candidates = _get_doc_candidates(doc)
    if candidates:
        options_text = "\n".join(f"{chr(ord('A') + i)}. {candidate}" for i, candidate in enumerate(candidates))
        question = f"{question}\n{options_text}"

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    full_prompt = pre_prompt + question + post_prompt
    return full_prompt


def extract_characters_regex(s):
    s = str(s).strip()
    if ")" in s:
        index = s.index(")")
        pred = s[index - 1 : index]
        return pred
    return s


def _normalize_task_type(task_type):
    task_aliases = {
        "findNeedle": "findNeedle",
        "needle": "findNeedle",
        "needleQA": "needleQA",
    }
    return task_aliases.get(task_type, task_type)


def mlvu_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    candidates = _get_doc_candidates(doc)

    if candidates:
        index2ans, all_choices = get_multi_choice_info(candidates)
        pred_ans = parse_multi_choice_response(pred, all_choices, index2ans)
        gold_ans = _normalize_answer_to_choice(doc.get("answer"), index2ans)
    else:
        pred_ans = extract_characters_regex(pred)
        gold_ans = doc.get("answer")

    task_type = _normalize_task_type(doc["question_type"])
    data_dict = {
        "question_id": doc.get("video", doc.get("video_name", doc["question"])),
        "task_type": task_type,
        "pred_answer": pred_ans,
        "answer": gold_ans,
    }

    return {"mlvu_percetion_score": data_dict}


def mlvu_aggregate_results_dev(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    TASK_TYPES = {"anomaly_reco", "count", "ego", "findNeedle", "order", "plotQA", "topic_reasoning"}
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = _normalize_task_type(result["task_type"])
        if task_type not in category2score:
            category2score[task_type] = {"correct": 0, "answered": 0}
            eval_logger.warning(f"Encountered unexpected MLVU dev task type: {task_type}")
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    task_category_scores = {}

    # Calculate and log accuracy for each task category
    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores[task_cate] = accuracy
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}%")

    # Average only across categories that actually appear in the evaluated split.
    answered_categories = [task_cate for task_cate, score in category2score.items() if task_cate in task_category_scores and score["answered"] > 0]
    if answered_categories:
        average_accuracy = sum(task_category_scores[task_cate] for task_cate in answered_categories) / len(answered_categories)
    else:
        average_accuracy = 0

    eval_logger.info(f"Average Performance Across Answered Task Categories: {average_accuracy:.1f}%")

    return average_accuracy


def mlvu_aggregate_results_test(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    TASK_TYPES = {"anomaly_reco", "count", "ego", "needleQA", "order", "plotQA", "sportsQA", "topic_reasoning", "tutorialQA"}
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = _normalize_task_type(result["task_type"])
        if task_type not in category2score:
            category2score[task_type] = {"correct": 0, "answered": 0}
            eval_logger.warning(f"Encountered unexpected MLVU test task type: {task_type}")
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    task_category_scores = {}

    # Calculate and log accuracy for each task category
    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores[task_cate] = accuracy
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}%")

    # Average only across categories that actually appear in the evaluated split.
    answered_categories = [task_cate for task_cate, score in category2score.items() if task_cate in task_category_scores and score["answered"] > 0]
    if answered_categories:
        average_accuracy = sum(task_category_scores[task_cate] for task_cate in answered_categories) / len(answered_categories)
    else:
        average_accuracy = 0

    eval_logger.info(f"Average Performance Across Answered Task Categories: {average_accuracy:.1f}%")

    return average_accuracy
