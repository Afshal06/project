def calculate_accuracy(y_true, y_pred):
    correct_predictions = (y_true == y_pred).sum()
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

def calculate_precision(y_true, y_pred):
    true_positives = ((y_true == 1) & (y_pred == 1)).sum()
    false_positives = ((y_true == 0) & (y_pred == 1)).sum()
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision

def calculate_recall(y_true, y_pred):
    true_positives = ((y_true == 1) & (y_pred == 1)).sum()
    false_negatives = ((y_true == 1) & (y_pred == 0)).sum()
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall

def calculate_f1_score(y_true, y_pred):
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score

def calculate_confusion_matrix(y_true, y_pred):
    true_positive = ((y_true == 1) & (y_pred == 1)).sum()
    true_negative = ((y_true == 0) & (y_pred == 0)).sum()
    false_positive = ((y_true == 0) & (y_pred == 1)).sum()
    false_negative = ((y_true == 1) & (y_pred == 0)).sum()
    return {
        'TP': true_positive,
        'TN': true_negative,
        'FP': false_positive,
        'FN': false_negative
    }