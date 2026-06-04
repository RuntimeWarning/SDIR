import os
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def get_hit_miss_counts(prediction, truth, thresholds=[30, 40, 50]):
    total_tp = np.zeros((prediction.shape[0], len(thresholds)), dtype=np.int32)
    total_fp = np.zeros((prediction.shape[0], len(thresholds)), dtype=np.int32)
    total_tn = np.zeros((prediction.shape[0], len(thresholds)), dtype=np.int32)
    total_fn = np.zeros((prediction.shape[0], len(thresholds)), dtype=np.int32)
    thresholds = np.array(thresholds, dtype=np.float32)
    for seq in range(len(prediction)):
        for id in range(len(thresholds)):
            # convert to masks for comparison
            y_pred_mask = prediction[seq] >= thresholds[id]
            y_true_mask = truth[seq] >= thresholds[id]
            tn, fp, fn, tp = np.bincount(y_true_mask.reshape(-1) * 2 + y_pred_mask.reshape(-1), minlength=4)
            total_tp[seq][id] = tp
            total_fp[seq][id] = fp
            total_tn[seq][id] = tn
            total_fn[seq][id] = fn
    return total_tn, total_fp, total_fn, total_tp


class Evaluation(object):
    # Shape: (seq_len, batch_size, height, width)
    def __init__(self, seq_len, value_scale=90.0, thresholds=[30, 40, 50]):
        self.value_scale = value_scale
        self._thresholds = np.array(thresholds)
        self._seq_len = seq_len
        self.begin()

    def begin(self):
        self._total_hits = np.zeros((self._seq_len, len(self._thresholds)), dtype=np.int32)
        self._total_misses = np.zeros((self._seq_len, len(self._thresholds)),  dtype=np.int32)
        self._total_false_alarms = np.zeros((self._seq_len, len(self._thresholds)), dtype=np.int32)
        self._total_correct_negatives = np.zeros((self._seq_len, len(self._thresholds)), dtype=np.int32)
        self._mse = np.zeros((self._seq_len, ), dtype=np.float32)
        self._mae = np.zeros((self._seq_len, ), dtype=np.float32)
        self._ssim = np.zeros((self._seq_len,), dtype=np.float32)
        self._psnr = np.zeros((self._seq_len,), dtype=np.float32)
        self._total_batch_num = 0

    def update(self, gt, pred):
        batch_size = gt.shape[1]
        assert gt.shape[0] == self._seq_len
        assert gt.shape == pred.shape
        self._total_batch_num += batch_size
        #TODO Save all the mse, mae, gdl, hits, misses, false_alarms and correct_negatives
        mse = (np.square(pred - gt)).sum(axis=(2, 3))
        mae = (np.abs(pred - gt)).sum(axis=(2, 3))
        self._mse += mse.sum(axis=1)
        self._mae += mae.sum(axis=1)
        # Calculate the hits, misses, false alarms and correct negatives
        correct_negatives, false_alarms, misses, hits = \
            get_hit_miss_counts(prediction=pred * self.value_scale, truth=gt * self.value_scale,
                                thresholds=self._thresholds)
        self._total_hits += hits
        self._total_misses += misses
        self._total_false_alarms += false_alarms
        self._total_correct_negatives += correct_negatives
        for i in range(self._seq_len):
            for j in range(batch_size):
                self._ssim[i] += compare_ssim(gt[i][j], 
                                              pred[i][j], 
                                              data_range=1.0) #np.max(gt[i][j])
                self._psnr[i] += compare_psnr(gt[i][j], 
                                              pred[i][j], 
                                              data_range=1.0)

    def calculate_stat(self):
        """
        Calculate all evaluation metrics.

        a: TP (Hits), b: FP (False Alarms), c: FN (Misses), d: TN (Correct Negatives)
        """
        eps = 1e-7  # Prevent division by zero.
        
        a = self._total_hits.astype(np.float64)
        b = self._total_false_alarms.astype(np.float64)
        c = self._total_misses.astype(np.float64)
        d = self._total_correct_negatives.astype(np.float64)
        
        # Basic categorical metrics.
        # POD (Probability of Detection) / Recall
        pod = a / (a + c + eps)
        # FAR (False Alarm Rate)
        far = b / (a + b + eps)
        # CSI (Critical Success Index) / Threat Score
        csi = a / (a + b + c + eps)
        
        # Statistical metrics.
        n = a + b + c + d
        # Accuracy
        accuracy = (a + d) / (n + eps)
        # Bias Score
        bias = (a + b) / (a + c + eps)
        
        # Skill scores.
        # Expected hits under random forecasts.
        aref = (a + b) / (n + eps) * (a + c)
        # GSS (Gilbert Skill Score) / ETS (Equitable Threat Score)
        gss = (a - aref) / (a + b + c - aref + eps)
        # HSS (Heidke Skill Score)
        hss = 2 * gss / (gss + 1 + eps)
        
        # Common machine-learning classification metrics.
        precision = a / (a + b + eps)
        recall = pod
        f1 = (2 * precision * recall) / (precision + recall + eps)
        
        # Image quality metrics averaged over all accumulated samples.
        mse = self._mse / (self._total_batch_num + eps)
        mae = self._mae / (self._total_batch_num + eps)
        ssim = self._ssim / (self._total_batch_num + eps)
        psnr = self._psnr / (self._total_batch_num + eps)
        
        return (pod, far, csi, hss, gss, mse, mae, 
                precision, f1, bias, ssim, accuracy, psnr)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        pod, far, csi, hss, gss, mse, mae, precision, \
        f1, bias, ssim, accuracy, psnr = self.calculate_stat()
        f = open(path + "/result.txt", 'w')
        f.write("Total Sequence Num: %d, Out Seq Len: %d\n"
                %(self._total_batch_num, self._seq_len))
        for i in range(len(self._thresholds)):
            f.write("Threshold = %g:\n" %self._thresholds[i])
            f.write("   POD: %s\n" %str(['{:.4f}'.format(elem) for elem in pod[:, i]]).replace('\'', ''))
            f.write("   FAR: %s\n" % str(['{:.4f}'.format(elem) for elem in far[:, i]]).replace('\'', ''))
            f.write("   CSI: %s\n" % str(['{:.4f}'.format(elem) for elem in csi[:, i]]).replace('\'', ''))
            f.write("   GSS: %s\n" % str(['{:.4f}'.format(elem) for elem in gss[:, i]]).replace('\'', ''))
            f.write("   HSS: %s\n" % str(['{:.4f}'.format(elem) for elem in hss[:, i]]).replace('\'', ''))
            f.write("   PRECISION: %s\n" % str(['{:.4f}'.format(elem) for elem in precision[:, i]]).replace('\'', ''))
            f.write("   Accuracy: %s\n" % str(['{:.4f}'.format(elem) for elem in accuracy[:, i]]).replace('\'', ''))
            f.write("   F1: %s\n" % str(['{:.4f}'.format(elem) for elem in f1[:, i]]).replace('\'', ''))
            f.write("   POD stat: avg %.4f/final %.4f\n" %(pod[:, i].mean(), pod[-1, i]))
            f.write("   FAR stat: avg %.4f/final %.4f\n" %(far[:, i].mean(), far[-1, i]))
            f.write("   CSI stat: avg %.4f/final %.4f\n" %(csi[:, i].mean(), csi[-1, i]))
            f.write("   GSS stat: avg %.4f/final %.4f\n" %(gss[:, i].mean(), gss[-1, i]))
            f.write("   HSS stat: avg %.4f/final %.4f\n" % (hss[:, i].mean(), hss[-1, i]))
            f.write("   PRECISION stat: avg %.4f/final %.4f\n" % (precision[:, i].mean(), precision[-1, i]))
            f.write("   Accuracy stat: avg %.4f/final %.4f\n" % (accuracy[:, i].mean(), accuracy[-1, i]))
            f.write("   F1 stat: avg %.4f/final %.4f\n" % (f1[:, i].mean(), f1[-1, i]))
        f.write("MSE: %s\n" % str(['{:.4f}'.format(elem) for elem in mse]).replace('\'', ''))
        f.write("MAE: %s\n" % str(['{:.4f}'.format(elem) for elem in mae]).replace('\'', ''))
        f.write("SSIM: %s\n" % str(['{:.4f}'.format(elem) for elem in ssim]).replace('\'', ''))
        f.write("PSNR: %s\n" % str(['{:.4f}'.format(elem) for elem in psnr]).replace('\'', ''))
        f.write("MSE stat: avg %.4f/final %.4f\n" % (mse.mean(), mse[-1]))
        f.write("MAE stat: avg %.4f/final %.4f\n" % (mae.mean(), mae[-1]))
        f.write("SSIM stat: avg %.4f/final %.4f\n" % (ssim.mean(), ssim[-1])) 
        f.write("PSNR stat: avg %.4f/final %.4f\n" % (psnr.mean(), psnr[-1])) 
        f.close()
