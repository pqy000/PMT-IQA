import torch
from scipy import stats
import numpy as np
import models
import data_loader
import time


class MultiTaskIQASolver(object):
    """Solver for training and testing hyperIQA"""

    def __init__(self, config, path, train_idx, test_idx, optimizer='adam', debug=False, usedyn_weight=False):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.optimizer = optimizer
        self.log_dir = './{}/{}'.format("result", config.dataset)
        self.debug = debug
        self.usedyn_weight = usedyn_weight
        self.file2print_detail = open("{}/train_detail.log".format(self.log_dir), 'a+')

        # self.model_multitask = models.MultiTaskIQANet(28, 128, 128, 64, 32, 16, 10).cuda()
        self.model_multitask = models.MultiTaskIQANet(28, 112, 112, 56, 28, 14, 10).cuda()
        # self.model_multitask = models.MultiTaskIQANet(16, 112, 56, 56, 28, 14, 11).cuda()
        # self.model_multitask = models.MultiTaskIQANet(16, 224, 112, 56, 28, 14, 11).cuda()
        self.model_multitask.train(True)
        # print(self.model_multitask)

        self.l1_loss = torch.nn.L1Loss().cuda()
        # self.l1_loss = torch.nn.MSELoss().cuda()
        self.ce_loss = torch.nn.CrossEntropyLoss().cuda()

        self.lambda1 = config.lambda1
        self.lambda2 = config.lambda2
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        if optimizer == 'adam':
            self.solver = torch.optim.Adam(self.model_multitask.parameters(), lr=self.lr,
                                           weight_decay=self.weight_decay)
        elif optimizer == 'sgd':
            self.solver = torch.optim.SGD(self.model_multitask.parameters(), lr=self.lr, momentum=0.9)
        else:
            print('Unknown optimizer', optimizer)
            exit(0)

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size,
                                              config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num,
                                             istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0
        # print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            epoch_loss2 = []
            pred_scores = []
            gt_scores = []
            pred_clses = []
            gt_cls = []

            tm_st = time.time()
            if self.debug:
                tm_s = time.time()
            for img_list, label_list, label_cls_list in self.train_data:
                if self.debug:
                    tm_e = time.time()
                    print('load data time:', tm_e - tm_s)
                K = len(img_list)
                img = torch.cat(img_list, 0).cuda()
                label = torch.cat(label_list, 0).cuda()
                label_cls = torch.cat(label_cls_list, 0).cuda()
                self.solver.zero_grad()
                out = self.model_multitask(img)  # 'out' contains mos and probabilistics for each class
                if self.debug:
                    tm_s1 = time.time()
                    print('forward time:', tm_s1 - tm_e, file=self.file2print_detail)

                # Quality prediction
                pred = out['mos']  # predicted mos
                pred_cls = out['cls']  # predicted probabilistics for each class
                pred_scores = pred_scores + pred.tolist()
                gt_scores = gt_scores + label.tolist()
                pred_clses = pred_clses + [pred_cls.tolist()]
                gt_cls = gt_cls + [label_cls.tolist()]
                if self.debug:
                    # print(pred.squeeze())
                    # print(label)
                    print('=======================>')

                loss = self.l1_loss(pred.squeeze(), label.float().detach())
                if self.debug:
                    print(loss.item())
                epoch_loss.append(loss.item())
                loss2 = self.ce_loss(pred_cls, label_cls)
                # print(loss.item(), loss2.item())
                epoch_loss2.append(loss2.item())
                if self.usedyn_weight:
                    # progress weight
                    # tmp = 5 / (t + np.exp(-10)) - 1 / (self.epochs - 1 - t + np.exp(-10))
                    self.lambda1 = t / (self.epochs - 1)
                    self.lambda2 = 1 - self.lambda1

                loss = self.lambda1 * loss + self.lambda2 * loss2
                loss.backward()
                self.solver.step()
                if self.debug:
                    tm_s = time.time()
                    print('backward time:', tm_s - tm_s1)

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            tm_en = time.time() - tm_st
            print("Epoch:{%4.4f}" % (tm_en), file=self.file2print_detail)
            print("Epoch:{%4.4f}" % (tm_en))
            test_srcc, test_plcc = self.test(self.test_data)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %(t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc), file=self.file2print_detail)
            # print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %(t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))

            # Update optimizer
            # lr = self.lr / pow(10, (t // 6))
            lr = self.lr / pow(10, (t // 6))
            # if t > 8:
            #     self.lrratio = 1
            # self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
            #               {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
            #               ]
            # self.solver = torch.optim.Adam(self.model_multitask.parameters(), lr=lr, weight_decay=self.weight_decay)
            if self.optimizer == 'adam':
                self.solver = torch.optim.Adam(self.model_multitask.parameters(), lr=lr,
                                               weight_decay=self.weight_decay)
            elif self.optimizer == 'sgd':
                self.solver = torch.optim.SGD(self.model_multitask.parameters(), lr=lr, momentum=0.9)
        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc), file=self.file2print_detail)
        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))
        print("*" * 100)
        print("*" * 100, file=self.file2print_detail)
        self.file2print_detail.flush()

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model_multitask.train(False)
        pred_scores = []
        gt_scores = []

        for img_list, label_list, label_cls_list in data:
            # Data.
            img = torch.cat(img_list, 0).cuda()
            label = torch.cat(label_list, 0).cuda()

            out = self.model_multitask(img)
            pred = out['mos']

            pred_scores = pred_scores + pred.view(-1).tolist()
            gt_scores = gt_scores + label.tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model_multitask.train(True)
        return test_srcc, test_plcc
