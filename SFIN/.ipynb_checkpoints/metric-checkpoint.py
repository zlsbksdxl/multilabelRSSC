import numpy as np
from sklearn.metrics import average_precision_score, hamming_loss

class MetricTracker:
    """跟踪训练过程中的指标"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_metrics(targets, probas):
    """计算多标签分类指标
    
    参数:
        targets: numpy数组，真实标签，形状为(N, c)
        probas: numpy数组，预测概率，形状为(N, c)
        
    返回:
        dict: 包含各种评估指标的字典
    """
    # 将概率转换为二进制预测（0/1）
    preds = (probas > 0.5).astype('int')
    y_true = targets
    y_pred = preds
    
    N, c = y_true.shape  # N: 样本数量, c: 类别数量
    
    # 计算平均精度均值(mAP)
    aps = []
    for j in range(c):
        if np.any(y_true[:, j]):
            ap = average_precision_score(y_true[:, j], probas[:, j])
            aps.append(ap)
    mAP = np.mean(aps) if aps else 0.0
    
    # 1. 基于样本的指标计算 (Example-Based)
    sample_precision = np.zeros(N)
    sample_recall = np.zeros(N)
    sample_f1 = np.zeros(N)
    
    for i in range(N):
        # 第i个样本的真正例数
        tp_e = np.sum(np.logical_and(y_true[i], y_pred[i]))
        # 第i个样本预测为正的总标签数
        pred_positive = np.sum(y_pred[i])
        # 第i个样本真实为正的总标签数
        true_positive = np.sum(y_true[i])
        
        # 计算样本精确率 P_e^i
        if pred_positive > 0:
            sample_precision[i] = tp_e / pred_positive
        # 计算样本召回率 R_e^i
        if true_positive > 0:
            sample_recall[i] = tp_e / true_positive
        # 计算样本F1值 F1_e^i
        if sample_precision[i] + sample_recall[i] > 0:
            sample_f1[i] = 2 * sample_precision[i] * sample_recall[i] / (sample_precision[i] + sample_recall[i])
    
    # 计算平均值
    P_e = np.mean(sample_precision)
    R_e = np.mean(sample_recall)
    F1_e = np.mean(sample_f1)
    
    # 2. 基于类别的指标计算 (Class-Based)
    class_precision = np.zeros(c)
    class_recall = np.zeros(c)
    class_f1 = np.zeros(c)
    
    for j in range(c):
        # 第j个类别的真正例数
        tp_c = np.sum(np.logical_and(y_true[:, j], y_pred[:, j]))
        # 第j个类别预测为正的总样本数
        pred_positive = np.sum(y_pred[:, j])
        # 第j个类别真实为正的总样本数
        true_positive = np.sum(y_true[:, j])
        
        # 计算类别精确率 P_c^j
        if pred_positive > 0:
            class_precision[j] = tp_c / pred_positive
        # 计算类别召回率 R_c^j
        if true_positive > 0:
            class_recall[j] = tp_c / true_positive
        # 计算类别F1值 F1_c^j
        if class_precision[j] + class_recall[j] > 0:
            class_f1[j] = 2 * class_precision[j] * class_recall[j] / (class_precision[j] + class_recall[j])
    
    # 计算平均值
    P_c = np.mean(class_precision)
    R_c = np.mean(class_recall)
    F1_c = np.mean(class_f1)
    
    # 计算Hamming Loss
    hl = hamming_loss(y_true, y_pred)
    
    return {
        # 原有的mAP指标
        "mAP": float(mAP),
        
        # 基于样本的指标 (Example-Based)
        "P_e": float(P_e),  # 样本精确率
        "R_e": float(R_e),  # 样本召回率
        "F1_e": float(F1_e),  # 样本F1值
        
        # 基于类别的指标 (Class-Based)
        "P_c": float(P_c),  # 类别精确率
        "R_c": float(R_c),  # 类别召回率
        "F1_c": float(F1_c),  # 类别F1值
        
        # 汉明损失
        "HL": float(hl)
    }