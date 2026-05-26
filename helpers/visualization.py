import os
import matplotlib
import numpy as np
matplotlib.use('Agg')
from imageio import imwrite
from matplotlib import colors
import matplotlib.pyplot as plt

datasets_params = {
    'cikm': {'PIXEL_SCALE': 90.0,
             'THRESHOLDS': [20, 30, 35, 40],
             'COLOR_MAP': np.array([
                                    [0, 0, 0, 0],
                                    [0, 236, 236, 255],
                                    [1, 160, 246, 255],
                                    [1, 0, 246, 255],
                                    [0, 239, 0, 255],
                                    [0, 200, 0, 255],
                                    [0, 144, 0, 255],
                                    [255, 255, 0, 255],
                                    [231, 192, 0, 255],
                                    [255, 144, 2, 255],
                                    [255, 0, 0, 255],
                                    [166, 0, 0, 255],
                                    [101, 0, 0, 255],
                                    [255, 0, 255, 255],
                                    [153, 85, 201, 255],
                                    [255, 255, 255, 255]
                                   ]) / 255,
             'BOUNDS': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 'PIXEL_SCALE'],
            },
    'sevir': {'COLOR_MAP': [[0, 0, 0],
                            [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
                            [0.1568627450980392, 0.7450980392156863, 0.1568627450980392],
                            [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
                            [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
                            [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
                            [0.9607843137254902, 0.9607843137254902, 0.0],
                            [0.9294117647058824, 0.6745098039215687, 0.0],
                            [0.9411764705882353, 0.43137254901960786, 0.0],
                            [0.6274509803921569, 0.0, 0.0],
                            [0.9058823529411765, 0.0, 1.0]],
              'PIXEL_SCALE': 255.0,
              'BOUNDS': [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 'PIXEL_SCALE'],
              'THRESHOLDS': [16, 74, 133, 160, 181, 219],
             },
    'shanghai': {'PIXEL_SCALE': 90.0,
                 'THRESHOLDS': [20, 30, 35, 40],
                 'COLOR_MAP': np.array([
                                        [0, 0, 0, 0],
                                        [0, 236, 236, 255],
                                        [1, 160, 246, 255],
                                        [1, 0, 246, 255],
                                        [0, 239, 0, 255],
                                        [0, 200, 0, 255],
                                        [0, 144, 0, 255],
                                        [255, 255, 0, 255],
                                        [231, 192, 0, 255],
                                        [255, 144, 2, 255],
                                        [255, 0, 0, 255],
                                        [166, 0, 0, 255],
                                        [101, 0, 0, 255],
                                        [255, 0, 255, 255],
                                        [153, 85, 201, 255],
                                        [255, 255, 255, 255]
                                       ]) / 255,
                 'BOUNDS': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 'PIXEL_SCALE'],
                },
    'meteonet': {'PIXEL_SCALE': 90.0,
                 'THRESHOLDS': [12, 18, 24, 32],
                 'COLOR_MAP': ['lavender', 'indigo', 'mediumblue',
                               'dodgerblue', 'skyblue', 'cyan',
                               'olivedrab', 'lime', 'greenyellow', 
                               'orange', 'red', 'magenta', 'pink',],
                 'BOUNDS': [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 'PIXEL_SCALE'],
                 },
}


def gray2color(image, dataset):
    # 定义颜色映射和边界
    cmap = colors.ListedColormap(datasets_params[dataset]['COLOR_MAP'])
    norm = colors.BoundaryNorm(datasets_params[dataset]['BOUNDS'], cmap.N)
    # 将图像进行染色
    colored_image = cmap(norm(image))
    return colored_image

def visualization_color(gt, pred, path, itr, dataset='cikm'):
    datasets_params[dataset]['BOUNDS'][-1] = datasets_params[dataset]['PIXEL_SCALE']
    colored_pred = gray2color(pred * datasets_params[dataset]['PIXEL_SCALE'], dataset)
    colored_gt = gray2color(gt * datasets_params[dataset]['PIXEL_SCALE'], dataset)
    grid_pred = np.concatenate([np.concatenate([i for i in colored_pred], axis=-2)], axis=-3)
    grid_gt = np.concatenate([np.concatenate([i for i in colored_gt], axis=-2)], axis=-3)
    grid_concat = np.concatenate([grid_gt, grid_pred], axis=-3,)
    plt.imsave(os.path.join(path, f'{itr}.png'), grid_concat)

def generate_image(predict, image_path, image_id, dataset='cikm'):
    datasets_params[dataset]['BOUNDS'][-1] = datasets_params[dataset]['PIXEL_SCALE']
    predict = gray2color(predict * datasets_params[dataset]['PIXEL_SCALE'], dataset)
    for i in range(predict.shape[0]):
        # if np.sum(gt[i][0]>0) < 32768: # 128: 8192; 256: 32768
        #         continue
        path = os.path.join(image_path, str(image_id))
        if not os.path.exists(path):
            os.makedirs(path)
        for j in range(predict.shape[1]):
            plt.imsave(os.path.join(path, f'{j}.png'), predict[i][j])
        image_id += 1
    return image_id
# imwrite(os.path.join(path, f'{j}.png'), (predict[i][j] * 255).astype(np.uint8))