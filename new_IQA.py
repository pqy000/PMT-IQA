import os
import argparse
import random
import numpy as np
import time
from optuna.visualization import *
from datetime import datetime
from MultiTaskIQASolver import MultiTaskIQASolver
import optuna
import warnings
warnings.filterwarnings('ignore')
import logging
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# main_path = os.environ['HOME']
# main_path += "/zhangyu"
main_path = "."

def objective(trial):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='live',
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=3,
                        help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=3,
                        help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    # parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=9, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=20, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=3, help='Train-test times')
    parser.add_argument('--lambda1', dest='lambda1', type=float, default=1.0, help='weight for mos loss')
    parser.add_argument('--lambda2', dest='lambda2', type=float, default=1.0, help='weight for crossentropy loss')
    parser.add_argument('--debug', dest='debug', type=bool, default=False, help='if print debug information')
    parser.add_argument('--dyn_weight', dest='dyn_weight', type=bool, default=False, help='if using dynamic weights')
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='adam', help='Optimizer type')
    config = parser.parse_args()

    folder_path = {
        'live': main_path + '/image_data/LIVE/',  #
        'csiq': main_path + '/image_data/CSIQ/',  #
        # 'tid2013': main_path + '/tid2013',
        'livec': main_path + '/image_data/ChallengeDB_release/',  # 保存
        'koniq-10k': main_path + '/image_data/koniq/',  # 保存
        'bid': main_path + '/image_data/BID/',  # 保存
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'bid': list(range(0, 586)),
    }
    sel_num = img_num[config.dataset]
    dirs = "result"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    srcc_all = np.zeros(config.train_test_num, dtype=np.float)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float)
    log_dir = './{}/{}'.format("result",config.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file2print_detail = open("{}/train_detail.log".format(log_dir), 'a+')
    print(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), file=file2print_detail)
    print("#" * 70, file=file2print_detail)
    print("Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC", file=file2print_detail)
    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num), file=file2print_detail)
    file2print_detail.flush()

    config.lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    config.batch_size = trial.suggest_int("batch_size", 9, 12, step=1)
    config.lambda1 = trial.suggest_float("lambda1", 0.9, 1.1, log=True)
    config.lambda2 = trial.suggest_float("lambda2", 0.9, 1.1, log=True)

    # for i in range(config.train_test_num):
    #     print('Round %d' % (i + 1))
        # Randomly select 80% images for training and the rest for testing
    random.shuffle(sel_num)
    train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
    test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

    solver = MultiTaskIQASolver(config, folder_path[config.dataset], train_index, test_index, config.optimizer,
                                config.debug, config.dyn_weight)
        # srcc_all[i], plcc_all[i] = solver.train()
    srcc, plcc = solver.train()
    # srcc_med = np.median(srcc_all)
    # plcc_med = np.median(plcc_all)
    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc, plcc), file=file2print_detail)

    return srcc

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "example-study"
storage_name = "sqlite:///{}.db".format(study_name)
# sampler = optuna.samplers.CmaEsSampler(use_separable_cma=True)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

#draw figure
fig1=plot_optimization_history(study)
fig2=plot_intermediate_values(study)
fig7=plot_slice(study)
fig9=plot_param_importances(study)
if not os.path.exists("image"):
    os.makedirs("image")
fig1.write_image("image/1.jpg", format="jpg")
fig2.write_image("image/2.jpg", format="jpg")
fig7.write_image("image/7.jpg", format="jpg")
fig9.write_image("image/9.jpg", format="jpg")

study = optuna.create_study(study_name, storage=study_name, load_if_exists=True)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
print(df)
print("Best params:", study.best_params)
print("Best Trial:", study.best_value)
