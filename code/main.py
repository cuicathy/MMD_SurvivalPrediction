import os
import pickle
from data_loader import *
from options import parse_args
from utils import complete_incomplete_data_selection
from train_test import train, test_complete_incomplete
from evaluation import evaluate_missingModa, evaluate_completeModa
import numpy as np
import torch
## Random Seeds
torch.cuda.manual_seed_all(2019)
torch.manual_seed(2019)
random.seed(2019)
np.random.seed(2019)
os.environ['PYTHONHASHSEED'] = str(2019)

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

opt = parse_args()
if opt.exp_num < 0:
    exp_number = 159
else:
    exp_number = opt.exp_num
fold_range=[i+1 for i in range(15)] # Use 15 train-val-test splits
'''
if exp_number == 159:
    print('noFast-allviews')
    opt.required_modality = ['demo', 'path', 'rad', 'omic']
    #opt.init_type = 'max'
    opt.random_drop_views = True
    opt.use_embedding = True
    opt.recon = True
    opt.recon_loss_weight = 1
'''
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
if not os.path.exists(opt.checkpoints_dir): os.makedirs(opt.checkpoints_dir)
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)): os.makedirs(
    os.path.join(opt.checkpoints_dir, opt.exp_name))
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)): os.makedirs(
    os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))

### 1. Initializes Data
# Load data
opt.rad_dir = os.path.join(opt.dataroot, 'radiology')
if opt.task == 'surv':
    if opt.use_embedding == True:
        data_cv_path = os.path.join(opt.dataroot, 'gbmlgg15cv_embedding3.pkl')
        test_data_cv_path = os.path.join(opt.dataroot, 'gbmlgg15cv_patches_embedding3.pkl')
data_cv_mask_path = os.path.join(opt.dataroot, 'mask_gbmlgg15cv3.pkl')
test_data_cv_mask_path = os.path.join(opt.dataroot, 'mask_gbmlgg15cv_patches3.pkl')
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_mask = pickle.load(open(data_cv_mask_path, 'rb'))
results = []
data_cv_splits = data_cv['cv_splits']
data_cv_mask_splits = data_cv_mask['cv_splits']
opt.device_specific = device

print("Using device:", device)
print(opt)
Available_idx_file_path = os.path.join(opt.dataroot, 'img_availability.csv')
print('Available_idx_file_path:', Available_idx_file_path)
print('data_cv_splits:', data_cv_path)
print('data_mask_cv_splits:', data_cv_mask_path)
data_cv_splits_completed, mask_cv_splits_completed = complete_incomplete_data_selection(opt, data_cv_splits=data_cv_splits, data_cv_mask=data_cv_mask_splits, available_idx_file_path=Available_idx_file_path, required_modality = opt.required_modality)

### 2. Training Sets-Up Main Loop
for k, data in data_cv_splits_completed.items():
    if k in fold_range:
        print("*******************************************")
        print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits_completed.items())))
        print("*******************************************")
        if os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name,
                                       '%s_%d_patch_pred_train.pkl' % (opt.model_name, k))):
            print("Train-Test Split already made.")
            continue
        ### 2.1 Trains Model
        print('# Training DATA:', len(data['train']['x_path']))
        print('# Val DATA:', len(data['val']['x_path']))
        print('# Testing DATA:', len(data['test']['x_path']))
        model, optimizer, metric_logger = train(opt, data, mask_cv_splits_completed[k], device, k)
print('######################################Testing###############################################')

### 3. Testing
test_complete_incomplete(opt, test_data_cv_path, test_data_cv_mask_path, fold_range)
### 4. Evaluation
model = [opt.model_name]
if opt.task == 'surv':
    model_appendidx_namelist = ['rad_demo', 'path_miss']
    _, results_selectedViews = evaluate_missingModa(model3=model, model_appendidx_namelist=model_appendidx_namelist,
                                                                              fold_range=fold_range)
    _, results_allviews = evaluate_completeModa(model3 = model, fold_range=fold_range)
    All_results = results_selectedViews.append(results_allviews, ignore_index=True)
    print(All_results['C-Index'])
    print('********************')
    All_results.to_csv(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, 'SurvPred_Eval.csv'), index=False)
    print(All_results['C-Index'])