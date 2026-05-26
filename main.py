import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import re
import argparse
import numpy as np
from engine import Model
from tqdm.auto import tqdm
from datasets import get_datasets
from helpers.save_weights import SaveWeights



parser = argparse.ArgumentParser()
parser.add_argument('--img_channel', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--depth', type=int, default=8)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--max_epoches', type=int, default=201)
parser.add_argument('--save_dir', type=str, default='model_data')
parser.add_argument('--generate_image', type=bool, default=False)
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument("--lr-beta1", type=float, default=0.95, help="learning rate beta 1")
parser.add_argument("--lr-beta2", type=float, default=0.99, help="learning rate beta 2")
parser.add_argument("--l2-norm", type=float, default=0, help="l2 norm weight decay")
parser.add_argument("--mixed_precision",type=str, default='fp16', help="mixed precision training")

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--is_train', type=bool, default=True)# True for training, False for testing
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--datasets', type=str, default='cikm') # cikm, shanghai, sevir
parser.add_argument('--output_length', type=int, default=10) # cikm: 10, shanghai & sevir: 20
parser.add_argument('--input_length', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model_name', type=str, default='SDIR')
parser.add_argument('--visualization', type=bool, default=True)
parser.add_argument('--frequency_stride', type=int, default=16) # cikm: 16, shanghai & sevir: 32
parser.add_argument('--img_size', type=int, default=128) # cikm: 128, shanghai & sevir: 256
parser.add_argument('--patch_size', type=int, default=4) # cikm: 4, shanghai & sevir: 8
parser.add_argument('--thresholds', type=str, default='[20.0, 30.0, 35.0, 40.0]')
parser.add_argument('--value_scale', type=float, default=90.0)
args = parser.parse_args()
args.tied = True
args.thresholds = [16, 74, 133, 160, 181, 219] if args.datasets=='sevir' else eval(args.thresholds) 
args.value_scale = 255.0 if args.datasets=='sevir' else 90.0

def train_wrapper(model):
    save_weights = SaveWeights(verbose=True)
    start_epoch = 1
    if args.pretrained_model:
        model.load(args.pretrained_model)
        start_epoch = re.findall(r'/(\d+)/', args.pretrained_model)
        start_epoch = int(start_epoch[0])+1 if start_epoch else 1
    train_input_handle = get_datasets(name=args.datasets, opt='train', # cikm, knmi, tianchi, inspur
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers, 
                                      shuffle=True)
    train_input_handle = model.accelerator.prepare(train_input_handle)
    for epoch in range(start_epoch, args.max_epoches):
        total_loss = []
        train_pbar = tqdm(train_input_handle,
                          total=len(train_input_handle),
                          disable=not model.accelerator.is_main_process)
        for ims in train_pbar:
            loss = model.train(ims)
            train_pbar.set_description('epoch: {} train loss: {:.4f}'.format(epoch, loss))
            total_loss.append(loss)
        mean_loss = np.mean(total_loss)
        model.accelerator.wait_for_everyone()
        if model.accelerator.is_main_process:
            model_dict = model.accelerator.get_state_dict(model.network)
            save_weights(mean_loss, model_dict, args.model_name+'_'+args.datasets, epoch)
        
def test_wrapper(model):
    model.load(args.pretrained_model)
    match = re.findall(r'/(\d+)/', args.pretrained_model)
    epoch = match[0] if match else "1"
    test_input_handle = get_datasets(name=args.datasets, opt='test', 
                                     batch_size=args.batch_size, 
                                     num_workers=args.num_workers, 
                                     shuffle=False)
    test_input_handle = model.accelerator.prepare(test_input_handle)
    model.test(test_input_handle, "test_"+epoch)

if __name__ == '__main__':
    print('Initializing models')
    model = Model(args)
    total = sum([param.nelement() for param in model.network.parameters()])
    model.accelerator.print("Main Model Parameters: %.2fM" % (total/1e6))
    if args.is_train:
        train_wrapper(model)
    else:
        test_wrapper(model)
    model.accelerator.end_training()