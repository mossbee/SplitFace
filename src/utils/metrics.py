def accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0

def precision(predictions, labels):
    true_positive = ((predictions == 1) & (labels == 1)).sum().item()
    false_positive = ((predictions == 1) & (labels == 0)).sum().item()
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0

def recall(predictions, labels):
    true_positive = ((predictions == 1) & (labels == 1)).sum().item()
    false_negative = ((predictions == 0) & (labels == 1)).sum().item()
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

def f1_score(predictions, labels):
    prec = precision(predictions, labels)
    rec = recall(predictions, labels)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0