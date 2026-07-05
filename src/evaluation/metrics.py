import csv
import os
from typing import List, Dict

import numpy as np


def calculate_iou(box1, box2):
    """
    Вычисление IoU между двумя ограничивающими рамками.

    Формат:
    [x1, y1, x2, y2]
    """

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union


def evaluate_predictions(
    pred_boxes,
    pred_labels,
    pred_scores,
    gt_boxes,
    gt_labels,
    iou_threshold=0.5,
):
    """
    Оценка одной картинки.

    Возвращает:
        TP
        FP
        FN
        Mean IoU
    """

    matched_gt = set()

    tp = 0
    fp = 0
    ious = []

    order = np.argsort(-pred_scores)

    pred_boxes = pred_boxes[order]
    pred_labels = pred_labels[order]

    for pred_box, pred_label in zip(pred_boxes, pred_labels):

        best_iou = 0
        best_gt = -1

        for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):

            if i in matched_gt:
                continue

            if pred_label != gt_label:
                continue

            iou = calculate_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_iou >= iou_threshold:

            tp += 1
            matched_gt.add(best_gt)
            ious.append(best_iou)

        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    mean_iou = np.mean(ious) if len(ious) else 0.0

    return tp, fp, fn, mean_iou


def calculate_metrics(tp, fp, fn):
    """
    Precision
    Recall
    F1
    """

    precision = tp / (tp + fp) if (tp + fp) else 0

    recall = tp / (tp + fn) if (tp + fn) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    return precision, recall, f1


def evaluate_dataset(results: List[Dict]):

    """
    results содержит список словарей вида

    {
        pred_boxes,
        pred_labels,
        pred_scores,
        gt_boxes,
        gt_labels
    }
    """

    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_ious = []

    for sample in results:

        tp, fp, fn, mean_iou = evaluate_predictions(
            sample["pred_boxes"],
            sample["pred_labels"],
            sample["pred_scores"],
            sample["gt_boxes"],
            sample["gt_labels"],
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        all_ious.append(mean_iou)

    precision, recall, f1 = calculate_metrics(
        total_tp,
        total_fp,
        total_fn,
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Mean IoU": np.mean(all_ious),
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
    }


def save_metrics(metrics, save_path):
    """
    Сохранение результатов в CSV.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(["Metric", "Value"])

        for key, value in metrics.items():
            writer.writerow([key, value])

    print(f"Метрики сохранены: {save_path}")