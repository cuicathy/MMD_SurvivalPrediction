import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from data_loader import PathgraphomicFastDatasetLoader, PathgraphomicDatasetLoader
from networks import define_net, define_reg, define_optimizer, define_scheduler
from utils import CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, count_parameters, dfs_unfreeze
import pickle
import os
import logging
from utils import complete_incomplete_data_selection

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def _init_fn(worker_id):
    np.random.seed(int(2019))

def train(opt, data, mask, device, k):
    model = define_net(opt, k)
    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)
    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))
    print("Activation Type:", opt.act_type)
    print("Optimizer Type:", opt.optimizer_type)
    print("Regularization Type:", opt.reg_type)

    use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_patch == True else ('_', 'all_st')
    custom_data_loader = PathgraphomicDatasetLoader(opt, data, mask, split='train', mode=opt.mode)

    train_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=True,
                                               drop_last=True, num_workers=0, worker_init_fn=seed_worker,generator=g)

    metric_logger = {'train': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': [], 'grad_acc': []},
                     'test': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': [], 'grad_acc': []}}

    num_epoch_no_improvement = 0
    patience_epoch = 10
    epochs_cnt = 0
    start_early_stop = 0
    cindex_val = 0
    best_eval_metric_value = -float('inf')

    for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):
        epochs_cnt = epochs_cnt + 1
        opt.epoch_count_training = epochs_cnt
        print('epoch:', epochs_cnt, 'num_epoch_no_improvement:', num_epoch_no_improvement)

        model = model.to(device)
        model.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
        loss_epoch, grad_acc_epoch = 0, 0
        for batch_idx, (x_name, x_path, x_omic, x_rad, x_demo, x_radiomics, censor, survtime, x_masks, x_keep_masks) in enumerate(train_loader):

            censor = censor.to(device) if "surv" in opt.task else censor

            x_masks['mask_path'] = x_masks['mask_path'].to(device)
            x_masks['mask_rad'] = x_masks['mask_rad'].to(device)
            x_masks['mask_omic'] = x_masks['mask_omic'].to(device)
            x_masks['mask_demo'] = x_masks['mask_demo'].to(device)

            output,pred = model(x_path=x_path.to(device), x_omic=x_omic.to(device), x_rad=x_rad.to(device), x_demo=x_demo.to(device),x_radiomics=x_radiomics.to(device), x_masks=x_masks, x_keep_masks=x_keep_masks.to(device))
            loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
            loss_reg = define_reg(opt, model)

            loss = opt.lambda_cox * loss_cox + opt.lambda_reg * loss_reg + loss_add(opt, output)
            loss_epoch += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if opt.task == "surv":
                risk_pred_all = np.concatenate(
                    (risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
                censor_all = np.concatenate(
                    (censor_all, censor.detach().cpu().numpy().reshape(-1)))
                survtime_all = np.concatenate(
                    (survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
            if opt.verbose > 0 and opt.print_every > 0 and (
                    batch_idx % opt.print_every == 0 or batch_idx + 1 == len(train_loader)):
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch + 1, opt.niter + opt.niter_decay, batch_idx + 1, len(train_loader), loss.item()))
        scheduler.step()

        if epochs_cnt > start_early_stop and (opt.measure):
            loss_epoch /= len(train_loader.dataset)
            loss_train, cindex_train, pvalue_train, surv_acc_train, pred_train = test(opt, model, data, mask,'train', device)
            loss_val, cindex_val, pvalue_val, surv_acc_val, pred_val = test(opt, model, data, mask,'val', device)
            loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test = test(opt, model, data, mask,'test', device)
            if opt.verbose > 0:
                if opt.task == 'surv':
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_train, 'C-Index', cindex_train))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'C-Index', cindex_test))
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Val', loss_val, 'C-Index', cindex_val))

        if epochs_cnt > start_early_stop + 1:
            current_eval_matric = cindex_val
            if best_eval_metric_value <= current_eval_matric:
                best_eval_metric_value = current_eval_matric
                #best_acc = cindex_test
                num_epoch_no_improvement = 0
                ### 3.3 Saves Model
                if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                    model_state_dict = model.module.cpu().state_dict()
                else:
                    model_state_dict = model.cpu().state_dict()

                torch.save({
                    'split': k,
                    'opt': opt,
                    'epoch': epochs_cnt,
                    'data': [],
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': []},
                    os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k)))

                pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name,
                                                         '%s_%d%spred_test1.pkl' % (opt.model_name, k, use_patch)),'wb'))

            else:
                num_epoch_no_improvement = num_epoch_no_improvement + 1
            if num_epoch_no_improvement == patience_epoch:
                print("Early Stopping")
                print(epochs_cnt)
                break
    return model, optimizer, metric_logger



def loss_add(opt, output):
    add_loss = 0
    if opt.recon == True:
        add_loss = add_loss + output["recon_loss"]* opt.recon_loss_weight
    return add_loss


def test(opt, model, data, mask, split, device):
    name_list = []
    model.eval()
    custom_data_loader = PathgraphomicFastDatasetLoader(opt, data, mask, split,
                                                                   mode=opt.mode)

    test_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_test_size, shuffle=False,
                                              drop_last=False, num_workers=0,
                                              worker_init_fn=seed_worker, generator=g)
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    loss_test = 0

    for batch_idx, (x_name, x_path, x_omic, x_rad, x_demo, x_radiomics, censor, survtime, x_masks,
                    x_keep_masks) in enumerate(test_loader):

        censor = censor.to(device) if "surv" in opt.task else censor
        x_masks['mask_path'] = x_masks['mask_path'].to(device)
        x_masks['mask_rad'] = x_masks['mask_rad'].to(device)
        x_masks['mask_omic'] = x_masks['mask_omic'].to(device)
        x_masks['mask_demo'] = x_masks['mask_demo'].to(device)
        output, pred = model(x_path=x_path.to(device), x_omic=x_omic.to(device),
                             x_rad=x_rad.to(device), x_demo=x_demo.to(device), x_radiomics=x_radiomics.to(device),
                             x_masks=x_masks, x_keep_masks=x_keep_masks.to(device))
        loss_cox = CoxLoss(survtime, censor, pred, device)
        loss_reg = define_reg(opt, model)
        loss = opt.lambda_cox * loss_cox + opt.lambda_reg * loss_reg + loss_add(opt, output)
        loss_test += loss.data.item()
        if opt.task == "surv":
            risk_pred_all = np.concatenate(
                (risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate(
                (survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
        name_list = name_list + list(x_name)
    loss_test /= len(test_loader.dataset)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all)
    pred_test = [risk_pred_all, survtime_all, censor_all, name_list]
    return loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test


def test_missingModa(opt, model, data, mask, split, device):
    model.eval()
    custom_data_loader = PathgraphomicFastDatasetLoader(opt, data, mask, split,
                                                        mode=opt.mode)
    test_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=1, shuffle=False,
                                              drop_last=False)
    pred_rad_demo = test_two_view_fun(test_loader, model, opt, device, 'rad_demo')
    pred_path_missing_demo = test_three_view_fun(test_loader, model, opt, device, 'path_missing')
    return pred_rad_demo, pred_path_missing_demo


def test_two_view_fun(test_loader, model, opt, device, view):
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    name_list = []
    for batch_idx, (x_name, x_path, x_omic, x_rad, x_demo, x_radiomics, censor, survtime, x_masks,
                    x_keep_masks) in enumerate(test_loader):
        if view == 'path_demo':
            x_omic = torch.zeros(x_omic.shape)
            x_rad = torch.zeros(x_rad.shape)
            x_radiomics = torch.zeros(x_radiomics.shape)
            x_keep_masks= torch.ones(x_keep_masks.shape)
            x_keep_masks[:, 3] = 0
            x_keep_masks[:, 1] = 0

        elif view == 'omic_demo':
            x_path = torch.zeros(x_path.shape)
            x_rad = torch.zeros(x_rad.shape)
            x_radiomics = torch.zeros(x_radiomics.shape)
            x_keep_masks= torch.ones(x_keep_masks.shape)
            x_keep_masks[:, 1] = 0
            x_keep_masks[:, 0] = 0
        elif view == 'rad_demo':
            x_path = torch.zeros(x_path.shape)
            x_omic = torch.zeros(x_omic.shape)
            x_keep_masks= torch.ones(x_keep_masks.shape)
            x_keep_masks[:, 3] = 0
            x_keep_masks[:, 0] = 0
        elif view == 'path_omic':
            x_demo = torch.zeros(x_demo.shape)
            x_rad = torch.zeros(x_rad.shape)
            x_radiomics = torch.zeros(x_radiomics.shape)
            x_keep_masks= torch.ones(x_keep_masks.shape)
            x_keep_masks[:, 2] = 0
            x_keep_masks[:, 1] = 0
        elif view == 'path_rad':
            x_omic = torch.zeros(x_omic.shape)
            x_demo = torch.zeros(x_demo.shape)
            x_keep_masks= torch.ones(x_keep_masks.shape)
            x_keep_masks[:, 3] = 0
            x_keep_masks[:, 2] = 0
        elif view == 'rad_omic':
            x_path = torch.zeros(x_path.shape)
            x_demo = torch.zeros(x_demo.shape)
            x_keep_masks= torch.ones(x_keep_masks.shape)
            x_keep_masks[:, 0] = 0
            x_keep_masks[:, 2] = 0
        censor = censor.to(device) if "surv" in opt.task else censor
        x_masks['mask_path'] = x_masks['mask_path'].to(device)
        x_masks['mask_rad'] = x_masks['mask_rad'].to(device)
        x_masks['mask_omic'] = x_masks['mask_omic'].to(device)
        x_masks['mask_demo'] = x_masks['mask_demo'].to(device)
        output, pred = model(x_path=x_path.to(device), x_omic=x_omic.to(device),
                             x_rad=x_rad.to(device), x_demo=x_demo.to(device), x_radiomics=x_radiomics.to(device),
                             x_masks=x_masks, x_keep_masks=x_keep_masks.to(device))
        if opt.task == "surv":
            risk_pred_all = np.concatenate(
                (risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate(
                (survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
        name_list.append(x_name)
    pred_test = [risk_pred_all, survtime_all, censor_all, name_list]
    return pred_test

def test_three_view_fun(test_loader, model, opt, device, view):
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    name_list = []
    for batch_idx, (x_name, x_path, x_omic, x_rad, x_demo, x_radiomics, censor, survtime, x_masks,
                    x_keep_masks) in enumerate(test_loader):
        if view == 'path_missing':
            x_path = torch.zeros(x_path.shape)
            x_keep_masks= torch.ones(x_keep_masks.shape)
            x_keep_masks[:,0] = 0
        elif view == 'demo_missing':
            x_demo = torch.zeros(x_demo.shape)
            x_keep_masks= torch.ones(x_keep_masks.shape)
            x_keep_masks[:,2] = 0
        elif view == 'omic_missing':
            x_omic = torch.zeros(x_omic.shape)
            x_keep_masks= torch.ones(x_keep_masks.shape)
            x_keep_masks[:,3] = 0
        elif view == 'rado_missing':
            x_rad = torch.zeros(x_rad.shape)
            x_radiomics = torch.zeros(x_radiomics.shape)
            x_keep_masks= torch.ones(x_keep_masks.shape)
            x_keep_masks[:,1] = 0
        censor = censor.to(device) if "surv" in opt.task else censor
        x_masks['mask_path'] = x_masks['mask_path'].to(device)
        x_masks['mask_rad'] = x_masks['mask_rad'].to(device)
        x_masks['mask_omic'] = x_masks['mask_omic'].to(device)
        x_masks['mask_demo'] = x_masks['mask_demo'].to(device)
        output, pred = model(x_path=x_path.to(device), x_omic=x_omic.to(device),
                             x_rad=x_rad.to(device), x_demo=x_demo.to(device), x_radiomics=x_radiomics.to(device),
                             x_masks=x_masks, x_keep_masks=x_keep_masks.to(device))
        if opt.task == "surv":
            risk_pred_all = np.concatenate(
                (risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate(
                (survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
        name_list.append(x_name)
    pred_test = [risk_pred_all, survtime_all, censor_all, name_list]
    return pred_test

def test_complete_incomplete(opt, data_cv_path, data_cv_mask_path, fold_range=[i + 1 for i in range(15)]):
    opt.random_drop_views = False
    opt.use_patch = True
    ######################################################
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    print("Using device:", device)
    if not os.path.exists(opt.checkpoints_dir): os.makedirs(opt.checkpoints_dir)
    if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)): os.makedirs(
        os.path.join(opt.checkpoints_dir, opt.exp_name))
    if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)): os.makedirs(
        os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))
    ### 1. Initializes Data
    use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_patch == True else ('_', 'all_st')
    print("Loading %s" % data_cv_path)
    data_cv = pickle.load(open(data_cv_path, 'rb'))
    data_cv_mask = pickle.load(open(data_cv_mask_path, 'rb'))
    data_cv_splits = data_cv['cv_splits']
    data_cv_mask_splits = data_cv_mask['cv_splits']
    results = []

    Available_idx_file_path = os.path.join(opt.dataroot, 'img_availability.csv')
    print('Available_idx_file_path:', Available_idx_file_path)
    print('data_cv_splits:', data_cv_path)
    print('data_mask_cv_splits:', data_cv_mask_path)
    data_cv_splits, mask_cv_splits = complete_incomplete_data_selection(opt, data_cv_splits=data_cv_splits, data_cv_mask=data_cv_mask_splits, available_idx_file_path=Available_idx_file_path)
    # Save for evaluation
    save_data_cv = {}
    save_data_cv['cv_splits'] = data_cv_splits
    save_data_cv['data_pd'] = data_cv['data_pd']

    ### 2. Sets-Up Main Loop
    for k, data in data_cv_splits.items():
        if k in fold_range:
            print("*******************************************")
            print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
            print("*******************************************")
            if opt.use_embedding == True:
                data['test']['x_omic'] = np.array(data['test']['x_omic']).squeeze(axis=1)
                data['test']['x_rad'] = np.array(data['test']['x_rad']).squeeze(axis=1)
                data['test']['x_demo'] = np.array(data['test']['x_demo']).squeeze(axis=1)

            print('# Testing DATA:', len(data['test']['x_path']))
            print('# Testing patient:', set(data['test']['x_patname']))
            load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name,
                                     '%s_%d.pt' % (opt.model_name, k))
            model_ckpt = torch.load(load_path, map_location=device)

            model_state_dict = model_ckpt['model_state_dict']
            if hasattr(model_state_dict, '_metadata'): del model_state_dict._metadata

            model = define_net(opt, None)
            if isinstance(model, torch.nn.DataParallel): model = model.module
            print('Loading the model from %s' % load_path)
            model.load_state_dict(model_state_dict)

            ### 2.2 Evalutes Train + Test Error, and Saves Model
            loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test = test(opt, model, data, mask_cv_splits[k], 'test', device)
            pred_rad_demo, pred_test_path_miss = test_missingModa(
                opt, model, data, mask_cv_splits[k], 'test', device)
            pickle.dump(pred_rad_demo, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name,
                        '%s_%d%spred_test_rad_demo.pkl' % (opt.model_name, k, use_patch)),'wb'))
            pickle.dump(pred_test_path_miss, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name,
                        '%s_%d%spred_test_path_miss.pkl' % (opt.model_name, k, use_patch)),'wb'))

            if opt.task == 'surv':
                print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
                logging.info(
                    "[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
                results.append(cindex_test)
            ### 3.3 Saves Model
            pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name,
                                                     '%s_%d%spred_test.pkl' % (opt.model_name, k, use_patch)), 'wb'))
    print('Split Results:', results)
    print("Average:", np.array(results).mean())
    pickle.dump(results,
                open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name),
                     'wb'))


