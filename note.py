import os
import numpy as np

dir_name = './shell'
txt_list = [x for x in os.listdir(dir_name) if 'txt' in x]
acc_dict = dict()
for txt_file in txt_list:
    if 'aug' not in txt_file:
        acc_list = []
        with open('{0}/{1}'.format(dir_name, txt_file), 'rt') as myfile:
            for myline in myfile:
                if 'MEAN :' in myline:
                    acc_list.append(float(myline[7:13]))

        mean_file = np.round(np.mean(acc_list), 2)
        std_file = np.round(np.std(acc_list), 2)

        acc_dict[txt_file] = [mean_file, std_file]
        #print(acc_dict[txt_file])

sorted(acc_dict.items())

[('ADCN_pre_none.txt',   [0.89, 0.01]), 
 ('ADCN_pre_cutout.txt', [0.88, 0.02]), 
 ('ADCN_eval.txt',       [0.80, 0.04]), 

 ('ADMCI_pre_none.txt',  [0.71, 0.03]), 
 ('ADMCI_pre_cutout.txt',[0.71, 0.04]), 
 ('ADMCI_eval.txt',      [0.67, 0.07]), 
 
 ('MCICN_pre_none.txt',  [0.59, 0.03]),
 ('MCICN_pre_cutout.txt',[0.60, 0.03]), 
 ('MCICN_eval.txt',      [0.58, 0.04])]

# embedding vector shape (1, 128)

[('ADCN_pre_none.txt',   [0.89, 0.01]), 
 ('ADCN_pre_all.txt',    [0.87, 0.02]), 
 ('ADCN_pre_cutout.txt', [0.88, 0.02]), 
 ('ADCN_eval.txt',       [0.88, 0.02]), # pretrained
 
 ('ADMCI_pre_none.txt',  [0.71, 0.03]), 
 ('ADMCI_pre_all.txt',   [0.68, 0.03]), 
 ('ADMCI_pre_cutout.txt',[0.71, 0.04]), 
 ('ADMCI_eval.txt',      [0.70, 0.04]), # pretrained
 
 ('MCICN_pre_none.txt',  [0.59, 0.03]),
 ('MCICN_pre_all.txt',   [0.57, 0.03]), 
 ('MCICN_pre_cutout.txt',[0.60, 0.03]), 
 ('MCICN_eval.txt',      [0.57, 0.04])] # pretrained

# embedding vector shape (2,2,2,1024)

[('ADCN_eval.txt',            [0.88, 0.02]), 
 ('ADCN_pre_all.txt',         [0.87, 0.02]), 
 ('ADCN_pre_crop.txt',        [nan, nan]), 
 ('ADCN_pre_cutout.txt',      [0.88, 0.02]), 
 ('ADCN_pre_none.txt',        [0.89, 0.01]), 
 ('ADCN_pre_weight_none.txt', [0.89, 0.02]), 

 ('ADMCI_eval.txt',           [0.70, 0.04]), 
 ('ADMCI_pre_all.txt',        [0.68, 0.03]), 
 ('ADMCI_pre_crop.txt',       [nan, nan]), 
 ('ADMCI_pre_cutout.txt',     [0.71, 0.04]), 
 ('ADMCI_pre_none.txt',       [0.71, 0.03]), 
 ('ADMCI_pre_weight_none.txt',[0.70, 0.02]), 

 ('MCICN_eval.txt',           [0.57, 0.04]), 
 ('MCICN_pre_all.txt',        [0.57, 0.03]), 
 ('MCICN_pre_crop.txt',       [nan, nan]), 
 ('MCICN_pre_cutout.txt',     [0.60, 0.03]), 
 ('MCICN_pre_none.txt',       [0.59, 0.03]), 
 ('MCICN_pre_weight_none.txt',[0.59, 0.02])]


[('ADCN_Aug.txt', [nan, nan]), 
 ('ADCN_eval.txt', [0.87, 0.02]), 
 ('ADCN_pre_None.txt', [0.88, 0.0]), 
 ('ADCN_pre_all.txt', [0.87, 0.02]), 
 ('ADCN_pre_crop.txt', [0.89, 0.0]), 
 ('ADCN_pre_cutout.txt', [0.88, 0.02]), 
 ('ADCN_pre_none.txt', [0.89, 0.01]), 
 ('ADCN_pre_weight_crop.txt', [0.89, 0.0]), 
 ('ADCN_pre_weight_none.txt', [0.89, 0.02]), 
 
 ('ADMCI_Aug.txt', [nan, nan]), 
 ('ADMCI_eval.txt', [0.71, 0.03]), 
 ('ADMCI_pre_None.txt', [0.76, 0.0]), 
 ('ADMCI_pre_all.txt', [0.68, 0.03]), 
 ('ADMCI_pre_crop.txt', [0.71, 0.02]), 
 ('ADMCI_pre_cutout.txt', [0.71, 0.04]), 
 ('ADMCI_pre_none.txt', [0.71, 0.03]), 
 ('ADMCI_pre_weight_crop.txt', [0.78, 0.01]), 
 ('ADMCI_pre_weight_none.txt', [0.7, 0.02]), 
 
 ('MCICN_Aug.txt', [nan, nan]), 
 ('MCICN_eval.txt', [0.57, 0.05]), 
 ('MCICN_pre_None.txt', [0.62, 0.0]), 
 ('MCICN_pre_all.txt', [0.57, 0.03]), 
 ('MCICN_pre_crop.txt', [0.58, 0.03]), 
 ('MCICN_pre_cutout.txt', [0.6, 0.03]), 
 ('MCICN_pre_none.txt', [0.59, 0.03]), 
 ('MCICN_pre_weight_crop.txt', [0.6, 0.01]), 
 ('MCICN_pre_weight_none.txt', [0.59, 0.02])]


 [('ADCN_Aug.txt', [nan, nan]), 
  ('ADCN_eval.txt', [0.87, 0.02]), 
  ('ADCN_eval2.txt', [0.87, 0.0]), 
  ('ADCN_pre_None.txt', [0.88, 0.0]), 
  ('ADCN_pre_all.txt', [0.87, 0.02]), 
  ('ADCN_pre_crop.txt', [0.89, 0.01]), 
  ('ADCN_pre_cutout.txt', [0.88, 0.02]), 
  ('ADCN_pre_none.txt', [0.89, 0.01]), 
  ('ADCN_pre_weight_crop.txt', [0.89, 0.0]), 
  ('ADCN_pre_weight_none.txt', [0.89, 0.02]), 
  ('ADCN_PRE_cutout.txt', [0.88, 0.02]),
  
  ('ADMCI_Aug.txt', [nan, nan]), 
  ('ADMCI_eval.txt', [0.71, 0.03]), 
  ('ADMCI_pre_None.txt', [0.76, 0.0]), 
  ('ADMCI_pre_all.txt', [0.68, 0.03]), 
  ('ADMCI_pre_crop.txt', [0.7, 0.04]), 
  ('ADMCI_pre_cutout.txt', [0.71, 0.04]), 
  ('ADMCI_pre_none.txt', [0.71, 0.03]), 
  ('ADMCI_pre_weight_crop.txt', [0.78, 0.01]), 
  ('ADMCI_pre_weight_none.txt', [0.7, 0.02]), 
  
  ('MCICN_Aug.txt', [nan, nan]), 
  ('MCICN_eval.txt', [0.57, 0.05]), 
  ('MCICN_pre_None.txt', [0.62, 0.0]), 
  ('MCICN_pre_all.txt', [0.57, 0.03]), 
  ('MCICN_pre_crop.txt', [0.58, 0.03]), 
  ('MCICN_pre_cutout.txt', [0.6, 0.03]), 
  ('MCICN_pre_none.txt', [0.59, 0.03]), 
  ('MCICN_pre_weight_crop.txt', [0.6, 0.01]), 
  ('MCICN_pre_weight_none.txt', [0.59, 0.02]),
  ('MCICN_PRE_cutout.txt', [0.57, 0.05])]
