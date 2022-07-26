class Config:

    def __init__(self, args):

        for name, value in vars(args).items():
            setattr(self, name, value)

        # self.mode = args.mode
        # self.train_num = args.train_num
        # self.task_name = args.task_name
        # self.stratify = args.stratify
        # self.random_seed = args.random_seed
        # self.pretrained_path = args.pretrained_path
        self.data = '../yAwareContrastiveLearning/adni_t1s_baseline'
        self.label = './csv/fsdat_baseline.csv'
        self.label_name = 'Dx.new'
        self.task_type = 'cls' # 'reg'
        self.valid_ratio = 0.25
        self.input_size = (1, 80, 80, 80)
        self.batch_size = 8
        self.pin_mem = True
        self.num_cpu_workers = 8
        self.num_classes = 2 # AD vs CN or MCI vs CN or AD vs MCI or reg
        self.cuda = True

        if self.mode == 'pretraining':
            self.model = 'DenseNet'
            self.nb_epochs = 100
            self.patience = 20
            self.lr = 1e-4
            self.weight_decay = 5e-5
            self.tf = 'crop'

        elif self.mode == 'modelaug':
            self.model = 'UNet'
            self.nb_epochs = 100
            self.patience = 20
            self.lr = 1e-4
            self.weight_decay = 5e-5
            #self.representation_path = './weights/DenseNet121_BHB-10K_yAwareContrastive.pth'

        elif self.mode == 'evaluation':
            self.model = 'DenseNet'
            self.nb_epochs = 100
            self.patience = 20
            self.lr = 1e-4
            self.weight_decay = 5e-5
            #self.augmentator_path = './ckpts/ADNI_ADCN_300_aug.pt'
