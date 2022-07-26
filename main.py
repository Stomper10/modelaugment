import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3'
import time
import random
import argparse
import datetime
import numpy as np
import pandas as pd
from scipy import stats
from config import Config
from dataset import ADNI_Dataset
from models.unet import UNet
from models.densenet import densenet121
from modelaug import AugModel
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torchsummary import summary
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score

if __name__ == "__main__":
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    print('[main.py started at {0}]'.format(nowDatetime))
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "modelaug", "evaluation"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly!")
    parser.add_argument("--pretrained_path", type=str, required=False,
                        help="Set the pretrained path if you want.")
    parser.add_argument("--train_num", type=int, required=True,
                        help="Set the number of training samples.")
    parser.add_argument("--task_name", type=str, required=False,
                        help="Set the name of the fine-tuning task. (e.g. AD/MCI)")
    parser.add_argument("--stratify", type=str, choices=["strat", "balan"], required=False,
                        help="Set training samples are stratified or balanced for fine-tuning task.")
    parser.add_argument("--random_seed", type=int, required=False, default=0,
                        help="Random seed for reproduction.")
    args = parser.parse_args()
    config = Config(args)
        
    # Control randomness for reproduction
    if config.random_seed != None:
        random_seed = config.random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        os.environ["PYTHONHASHSEED"] = str(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Sampling
    labels = pd.read_csv(config.label)
    if config.mode == 'pretraining':
        print('Task: Pretriaing for {0}'.format(config.task_name))
        print('Task type: {0}'.format(config.task_type))
        print('N = {0}'.format(config.train_num))
    elif config.mode == 'modelaug':
        print('Task: Model augmentation for {0}'.format(config.task_name))
        print('Task type: {0}'.format(config.task_type))
        print('N = {0}'.format(config.train_num))
    else: # config.mode == 'evaluation':
        print('Task: Evaluation for {0}'.format(config.task_name))
        print('Task type: {0}'.format(config.task_type))
        print('N = {0}'.format(config.train_num))
        
    if config.task_type == 'cls':
        print('Policy: {0}'.format(config.stratify))
        task_include = config.task_name.split('/')
        assert len(task_include) == 2, 'Set two labels.'
        assert config.num_classes == 2, 'Set config.num_classes == 2'

        data_1 = labels[labels[config.label_name] == task_include[0]]
        data_2 = labels[labels[config.label_name] == task_include[1]]

        if config.stratify == 'strat':
            ratio = len(data_1) / (len(data_1) + len(data_2))
            len_1_train = round(config.train_num*ratio)
            len_2_train = config.train_num - len_1_train
            len_1_valid = round(int(config.train_num*config.valid_ratio)*ratio)
            len_2_valid = int(config.train_num*config.valid_ratio) - len_1_valid
            assert config.train_num*(1+config.valid_ratio) < (len(data_1) + len(data_2)), 'Not enough valid data. Set smaller --train_num or smaller config.valid_ratio in config.py.'
            train1, valid1, test1 = np.split(data_1.sample(frac=1, random_state=random_seed), 
                                             [len_1_train, len_1_train + len_1_valid])
            train2, valid2, test2 = np.split(data_2.sample(frac=1, random_state=random_seed), 
                                             [len_2_train, len_2_train + len_2_valid])
            label_train = pd.concat([train1, train2]).sample(frac=1, random_state=random_seed)
            label_valid = pd.concat([valid1, valid2]).sample(frac=1, random_state=random_seed)
            label_test = pd.concat([test1, test2]).sample(frac=1, random_state=random_seed)
            assert len(label_test) >= 100, 'Not enough test data. (Total: {0})'.format(len(label_test))
        else: # config.stratify == 'balan'
            limit = len(data_1) if len(data_1) <= len(data_2) else len(data_2)
            data_1 = data_1.sample(frac=1, random_state=random_seed)[:limit]
            data_2 = data_2.sample(frac=1, random_state=random_seed)[:limit]
            len_1_train = round(config.train_num*0.5)
            len_2_train = config.train_num - len_1_train
            len_1_valid = round(int(config.train_num*config.valid_ratio)*0.5)
            len_2_valid = int(config.train_num*config.valid_ratio) - len_1_valid
            assert config.train_num*(1+config.valid_ratio) < limit*2, 'Not enough data to make balanced set.'
            train1, valid1, test1 = np.split(data_1.sample(frac=1, random_state=random_seed), 
                                             [len_1_train, len_1_train + len_1_valid])
            train2, valid2, test2 = np.split(data_2.sample(frac=1, random_state=random_seed), 
                                             [len_2_train, len_2_train + len_2_valid])
            label_train = pd.concat([train1, train2]).sample(frac=1, random_state=random_seed)
            label_valid = pd.concat([valid1, valid2]).sample(frac=1, random_state=random_seed)
            label_test = pd.concat([test1, test2]).sample(frac=1, random_state=random_seed)
            assert len(label_test) >= 100, 'Not enough test data. (Total: {0})'.format(len(label_test))

        print('\nTrain data info:\n{0}\nTotal: {1}\n'.format(label_train[config.label_name].value_counts().sort_index(), len(label_train)))
        print('Valid data info:\n{0}\nTotal: {1}\n'.format(label_valid[config.label_name].value_counts().sort_index(), len(label_valid)))
        print('Test data info:\n{0}\nTotal: {1}\n'.format(label_test[config.label_name].value_counts().sort_index(), len(label_test)))

        label_train[config.label_name].replace({task_include[0]: 0, task_include[1]: 1}, inplace=True)
        label_valid[config.label_name].replace({task_include[0]: 0, task_include[1]: 1}, inplace=True)
        label_test[config.label_name].replace({task_include[0]: 0, task_include[1]: 1}, inplace=True)

    else: # config.task_type = 'reg'
        task_include = config.task_name.split('/')
        assert len(task_include) == 1, 'Set only one label.'
        assert config.num_classes == 1, 'Set config.num_classes == 1'

        labels = pd.read_csv(config.label)
        labels = labels[(np.abs(stats.zscore(labels[config.label_name])) < 3)] # remove outliers w.r.t. z-score > 3
        assert config.train_num*(1+config.valid_ratio) <= len(labels), 'Not enough valid data. Set smaller --train_num or smaller config.valid_ratio in config.py.'

        label_train, label_valid, label_test = np.split(labels.sample(frac=1, random_state=random_seed), 
                                                        [config.train_num, int(config.train_num*(1+config.valid_ratio))])
        
        print('\nTrain data info:\nTotal: {0}\n'.format(len(label_train)))
        print('Valid data info:\nTotal: {0}\n'.format(len(label_valid)))
        print('Test data info:\nTotal: {0}\n'.format(len(label_test)))
    
    # Dataset
    if config.mode == 'pretraining':
        dataset_train1 = ADNI_Dataset(config, label_train, aug=True)
        dataset_train2 = ADNI_Dataset(config, label_train, aug=False)
        dataset_train = ConcatDataset([dataset_train1, dataset_train2])
    elif config.mode == 'evaluation':
        dataset_train1 = ADNI_Dataset(config, label_train, aug=False)
        dataset_train2 = ADNI_Dataset(config, label_train, aug=False)
        dataset_train = ConcatDataset([dataset_train1, dataset_train2])
    dataset_val = ADNI_Dataset(config, label_valid, aug=False)
    dataset_test = ADNI_Dataset(config, label_test, aug=False)

    loader_train = DataLoader(dataset_train,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=config.pin_mem,
                              num_workers=config.num_cpu_workers)
    loader_val = DataLoader(dataset_val,
                            batch_size=config.batch_size,
                            shuffle=False,
                            pin_memory=config.pin_mem,
                            num_workers=config.num_cpu_workers)
    loader_test = DataLoader(dataset_test,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=config.pin_mem,
                             num_workers=config.num_cpu_workers)

    # Training
    if config.mode == 'pretraining':
        if config.model == 'DenseNet': ###
            net = densenet121(mode='classifier', drop_rate=0.0, num_classes=config.num_classes)
            #print(net)
            #summary(net.to('cuda'), (1, 80, 80, 80))
        elif config.model == 'UNet':
            net = UNet(config.num_classes, mode='classif')
        else:
            raise ValueError('Unkown model: %s'%config.model)
    elif config.mode == 'modelaug':
        if config.model == 'UNet': ###
            net = UNet(1, mode='seg')
            print(net)
            summary(net.to('cuda'), (1, 80, 80, 80))
        else:
            raise ValueError('Unkown model: %s'%config.model)
    else: # config.mode == 'evaluation':
        if config.model == 'DenseNet': ###
            net = densenet121(mode='classifier', drop_rate=0.0, num_classes=config.num_classes)
            print(net)
            summary(net.to('cuda'), (1, 80, 80, 80))
        elif config.model == 'UNet':
            net = UNet(config.num_classes, mode='classif')
        else:
            raise ValueError('Unkown model: %s'%config.model)

    if config.mode == 'pretraining':
        if config.task_type == 'cls':
            loss = CrossEntropyLoss()
        else: # config.task_type == 'reg':
            loss = MSELoss()
    elif config.mode == 'modelaug':
        loss = MSELoss(reduction='none')
    else: # config.mode == 'evaluation':
        if config.task_type == 'cls':
            loss = CrossEntropyLoss()
        else: # config.task_type == 'reg':
            loss = MSELoss()

    model = AugModel(net, loss, loader_train, loader_val, loader_test, config)
    
    # Inference
    if config.mode == 'pretraining':
        outGT, outPRED = model.pretraining()
    elif config.mode == 'modelaug':
        inputs_aug_images = model.modelaug()
    else: # config.mode == 'evaluation':
        outGT, outPRED, inputs_images = model.evaluation()
    
    if config.mode != 'modelaug':
        if config.mode == 'evaluation':
            inputs_images = inputs_images.cpu().numpy()
            np.save('./images/{0}_images{1}'.format(config.task_name.replace('/', ''), config.random_seed), inputs_images)
        if config.task_type == 'cls':
            outGTnp = outGT.cpu().numpy()
            outPREDnp = outPRED.cpu().numpy()
            print('\n<<< Test Results: AUROC >>>')
            outAUROC = []
            for i in range(config.num_classes):
                outAUROC.append(roc_auc_score(outGTnp[:, i], outPREDnp[:, i]))
            aurocMean = np.array(outAUROC).mean()
            print('MEAN', ': {:.4f}'.format(aurocMean))
        else: # config.task_type == 'reg':
            outGTnp = outGT.cpu().numpy()
            outPREDnp = outPRED.cpu().numpy()
            print('\n<<< Test Results >>>')
            print('MSE: {:.2f}'.format(mean_squared_error(outGTnp, outPREDnp)))
            print('MAE: {:.2f}'.format(mean_absolute_error(outGTnp, outPREDnp)))
            print('RMSE: {:.2f}'.format(np.sqrt(mean_squared_error(outGTnp, outPREDnp))))
            print('R2-score: {:.4f}'.format(r2_score(outGTnp, outPREDnp)))
    else:
        inputs_aug_images = inputs_aug_images.cpu().numpy()
        np.save('./images/{0}_aug_images{1}'.format(config.task_name.replace('/', ''), config.random_seed), inputs_aug_images)

    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    print('[main.py finished at {0}]'.format(nowDatetime))
