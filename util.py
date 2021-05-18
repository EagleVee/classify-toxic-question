import torch
from sklearn.metrics import f1_score, roc_auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def margin_score(targets, predictions):
    return ((targets == 1) * (1 - predictions) + (targets == 0) * (predictions)).mean()


def report_perf(valid_dataset, predictions_va, threshold, idx, epoch_cur, desc='val set'):
    val_f1 = f1_score(valid_dataset.targets, predictions_va > threshold)
    val_auc = roc_auc_score(valid_dataset.targets, predictions_va)
    val_margin = margin_score(valid_dataset.targets, predictions_va)
    print('idx {} epoch {} {} f1 : {:.4f} auc : {:.4f} margin : {:.4f}'.format(
        idx,
        epoch_cur,
        desc,
        val_f1,
        val_auc,
        val_margin))


def get_gpu_memory_usage(device_id):
    return round(torch.cuda.max_memory_allocated(device_id) / 1000 / 1000)


def avg(loss_list):
    if len(loss_list) == 0:
        return 0
    else:
        return sum(loss_list) / len(loss_list)
