import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.optim.lr_scheduler as lr_scheduler

from utils import *
import torchvision.models as models

################
# Network Utils
################
def define_optimizer(opt, model):
    optimizer = None
    if opt.optimizer_type == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=opt.final_lr)
    elif opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),
                                     weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay,
                                        initial_accumulator_value=0.1)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
    return optimizer

def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_reg(opt, model):
    loss_reg = None

    if opt.reg_type == 'none':
        loss_reg = 0
    elif opt.reg_type == 'path':
        loss_reg = regularize_path_weights(model=model)
    elif opt.reg_type == 'mm':
        loss_reg = regularize_MM_weights(model=model)
    elif opt.reg_type == 'all':
        loss_reg = regularize_weights(model=model)
    elif opt.reg_type == 'omic':
        loss_reg = regularize_MM_omic(model=model)
    else:
        raise NotImplementedError('reg method [%s] is not implemented' % opt.reg_type)
    return loss_reg


def define_net(opt, k):
    act = define_act_layer(act_type=opt.act_type)
    init_max = True if opt.init_type == "max" else False
    if opt.mode == "path":
        net = get_vgg(path_dim=opt.path_dim, act=act, label_dim=opt.label_dim)
    elif opt.mode == "rad_omic":
        net = Rad_Net_omic(opt, act=act, label_dim=opt.label_dim,
                           rad_dim=opt.rad_dim)
    elif opt.mode == "demo":
        net = demo_MaxNet(omic_dim=opt.demo_dim, act=act,
                          label_dim=opt.label_dim, init_max=init_max)
    elif opt.mode == "omic":
        net = MaxNet(opt=opt, input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate,
                         act=act,
                         label_dim=opt.label_dim, init_max=init_max)
    elif opt.mode == "Multimodal":
        net = Multimodal_fusion(opt=opt, act=act, k=k, init_max=init_max)
    else:
        raise NotImplementedError('model [%s] is not implemented' % opt.model)
    print('opt.init_type', opt.init_type)
    return init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)

def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    print('act_type', act_type)
    return act_layer

def masked_mean(data, masks):
    num = sum((X * mask[:, None].float() for X, mask in zip(data, masks)))
    denom = sum((mask for mask in masks))[:, None].float()
    return num / denom

############
# Path Model
############
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class PathNet(nn.Module):

    def __init__(self, features, path_dim=32, act=None, num_classes=1):
        super(PathNet, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, path_dim),
            nn.ReLU(True),
            nn.Dropout(0.05)
        )

        self.linear = nn.Linear(path_dim, num_classes)
        self.act = act

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        dfs_freeze(self.features)

    def forward(self, **kwargs):
        x = kwargs['x_path']
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        features = self.classifier(x)
        hazard = self.linear(features)

        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def get_vgg(arch='vgg19_bn', cfg='E', act=None, batch_norm=True, label_dim=1, pretrained=True, progress=True, **kwargs):
    model = PathNet(make_layers(cfgs[cfg], batch_norm=batch_norm), act=act, num_classes=label_dim, **kwargs)

    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        for key in list(pretrained_dict.keys()):
            if 'classifier' in key: pretrained_dict.pop(key)

        model.load_state_dict(pretrained_dict, strict=False)
        print("Initializing Path Weights")

    return model

############
# Demo Model
############
class demo_MaxNet(nn.Module):
    def __init__(self, input_dim=9, omic_dim=32, dropout_rate=0.1, act=None, label_dim=1, init_max=True):
        super(demo_MaxNet, self).__init__()
        hidden = [32, 32]
        self.act = act

        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], omic_dim),
            nn.ReLU())

        self.encoder = nn.Sequential(encoder1, encoder2, encoder3)
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))

        if init_max: init_max_weights(self)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs['x_demo']
        features = self.encoder(x)
        out = self.classifier(features)
        if self.act is not None:
            out = self.act(out)
            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift
        return features, out


class Rad_Net_omic(nn.Module):
    # Set up the network structure
    def __init__(self, opt, rad_dim=32, act=None, label_dim=1):
        super(Rad_Net_omic, self).__init__()
        self.opt = opt
        self.act = act
        self.net = models.resnet18(pretrained=True)
        weight = self.net.conv1.weight.clone()
        self.net.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.net.conv1.weight[:, 0:3, :, :] = nn.Parameter(weight)
            self.net.conv1.weight[:, 3] = self.net.conv1.weight[:, 0]
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, opt.mmhid)
        dfs_freeze(self.net)
        for param in self.net.fc.parameters():
            param.requires_grad = True

        self.fc1 = nn.Sequential(nn.Linear(opt.mmhid + opt.input_size_radiomics, opt.mmhid))
        self.bn1 = nn.BatchNorm1d(opt.mmhid)
        self.fc2 = nn.Sequential(nn.Linear(opt.mmhid, rad_dim),nn.ReLU(True))
        self.relu_res = nn.ReLU(True)
        self.dropout_res = nn.Dropout(0.05)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Sequential(nn.Linear(rad_dim, label_dim))
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs['x_rad']
        x_add = kwargs['x_radiomics']
        x = self.net(x)
        # New
        x = self.dropout(x)
        fea_combine = torch.cat((x, x_add), axis=1)
        fea_combine = self.fc1(fea_combine)
        fea_combine = self.relu_res(fea_combine)
        fea_combine = self.dropout_res(fea_combine)
        features = self.fc2(fea_combine)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard

############
# Omic Model
############
class MaxNet(nn.Module):
    def __init__(self, opt, input_dim=80, omic_dim=32, dropout_rate=0.25, act=None, label_dim=1, init_max=True):
        super(MaxNet, self).__init__()
        hidden = [64, 48, 32, 32]
        self.act = act
        self.opt = opt

        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim),
            nn.ReLU())

        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))

        if init_max: init_max_weights(self)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs['x_omic']
        features = self.encoder(x)
        out = self.classifier(features)
        if self.act is not None:
            out = self.act(out)
            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift
        return features, out

class Multimodal_fusion(nn.Module):

    def __init__(self, opt, act, k, init_max):
        super(Multimodal_fusion, self).__init__()
        self.opt = opt
        self.hidden_dim = self.opt.mmhid
        self.fc1 = nn.Sequential(nn.Linear(self.opt.path_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc2 = nn.Sequential(nn.Linear(self.opt.rad_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc3 = nn.Sequential(nn.Linear(self.opt.demo_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc4 = nn.Sequential(nn.Linear(self.opt.omic_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.act = act

        if opt.recon == True:
            self.fc1_decode = nn.Sequential(nn.Linear(self.opt.mmhid, self.opt.mmhid), nn.ReLU(), nn.Linear(self.opt.mmhid, self.opt.path_dim))
            self.fc2_decode = nn.Sequential(nn.Linear(self.opt.mmhid, self.opt.mmhid), nn.ReLU(), nn.Linear(self.opt.mmhid, self.opt.rad_dim))
            self.fc3_decode = nn.Sequential(nn.Linear(self.opt.mmhid, self.opt.mmhid), nn.ReLU(), nn.Linear(self.opt.mmhid, self.opt.demo_dim))
            self.fc4_decode = nn.Sequential(nn.Linear(self.opt.mmhid, self.opt.mmhid), nn.ReLU(), nn.Linear(self.opt.mmhid, self.opt.omic_dim))

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
        self.fuse_fc = nn.Sequential(nn.Linear(self.hidden_dim, opt.mmhid), nn.ReLU(), nn.Dropout(0.1),
                                        nn.Linear(opt.mmhid, opt.mmhid), nn.ReLU(), nn.Dropout(0.1))


        self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
        self.KL_loss = nn.KLDivLoss(log_target=True)
        #####################################End2end initialization
        if self.opt.task == 'surv':
            omic_model_name = 'omic_surv_final_relu2_32_new_positive002'
            demo_model_name = 'demo_surv_final_relu2_32_new002'
            rad_model_name = 'rad_surv_32_all_again'
            path_model_name = 'path_surv_reproduce'

        if opt.use_embedding == False:
            self.path_net = get_vgg(path_dim=opt.path_dim, act=act, label_dim=opt.label_dim)
            if k is not None:
                pt_fname = '_%d.pt' % k
                if self.opt.unimodal_pretrain == True:
                    best_path_ckpt = torch.load(os.path.join(opt.checkpoints_dir, opt.exp_name, path_model_name, path_model_name + pt_fname),
                                                map_location=torch.device('cpu'))
                    self.path_net.load_state_dict(best_path_ckpt['model_state_dict'])
        if opt.use_embedding == False:
            self.omic_net = MaxNet(opt=opt, input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate,
                         act=act,
                         label_dim=opt.label_dim, init_max=init_max)
            if k is not None:
                pt_fname = '_%d.pt' % k
                if self.opt.unimodal_pretrain == True:

                    best_path_ckpt = torch.load(os.path.join(opt.checkpoints_dir, opt.exp_name, omic_model_name, omic_model_name + pt_fname),
                                                map_location=torch.device('cpu'))
                    self.omic_net.load_state_dict(best_path_ckpt['model_state_dict'])


        if opt.use_embedding == False:
            self.demo_net = demo_MaxNet(omic_dim=opt.demo_dim, act=act,
                          label_dim=opt.label_dim, init_max=init_max)
            if k is not None:
                pt_fname = '_%d.pt' % k
                if self.opt.unimodal_pretrain == True:
                    best_path_ckpt = torch.load(os.path.join(opt.checkpoints_dir, opt.exp_name, demo_model_name,demo_model_name + pt_fname),
                                                map_location=torch.device('cpu'))
                    self.demo_net.load_state_dict(best_path_ckpt['model_state_dict'])
                    print('load demo_net model_state_dict')

        if opt.use_embedding == False:
            self.rad_net = Rad_Net_omic(opt, 'Multi_channel_ResNet', act=act, label_dim=opt.label_dim,
                           rad_dim=opt.rad_dim)
            if k is not None:
                pt_fname = '_%d.pt' % k
                if self.opt.unimodal_pretrain == True:
                    best_path_ckpt = torch.load(os.path.join(opt.checkpoints_dir, opt.exp_name, rad_model_name,rad_model_name + pt_fname),
                                                map_location=torch.device('cpu'))
                    self.rad_net.load_state_dict(best_path_ckpt['model_state_dict'])

    def forward(self, **kwargs):
        x_masks = kwargs['x_masks']
        x_keep_masks = kwargs['x_keep_masks']
        if self.opt.use_embedding == False:
            x_path_fea, _ = self.path_net(x_path=kwargs['x_path'])
            x_demo_fea, _ = self.demo_net(x_demo=kwargs['x_demo'])
            x_omic_fea, _ = self.omic_net(x_omic=kwargs['x_omic'])
            x_rad_fea,_ = self.rad_net(x_rad=kwargs['x_rad'], x_radiomics=kwargs['x_radiomics'])
        else:
            x_path_fea = kwargs['x_path']
            x_rad_fea = kwargs['x_rad']
            x_demo_fea = kwargs['x_demo']
            x_omic_fea = kwargs['x_omic']

        x_path_fea = x_path_fea.view(x_path_fea.shape[0], -1)
        x_path_fea_encode = F.relu(self.fc1(x_path_fea))

        x_rad_fea = x_rad_fea.view(x_rad_fea.shape[0], -1)
        x_rad_fea_encode = F.relu(self.fc2(x_rad_fea))

        x_demo_fea = x_demo_fea.view(x_demo_fea.shape[0], -1)
        x_demo_fea_encode = F.relu(self.fc3(x_demo_fea))

        x_omic_fea = x_omic_fea.view(x_omic_fea.shape[0], -1)
        x_omic_fea_encode = F.relu(self.fc4(x_omic_fea))

        mean = masked_mean((x_path_fea_encode, x_rad_fea_encode, x_demo_fea_encode, x_omic_fea_encode),
                           (x_keep_masks[:, 0], x_keep_masks[:, 1], x_keep_masks[:, 2], x_keep_masks[:, 3]))
        if self.opt.recon == True:
            x_path_fea_recon = F.relu(self.fc1_decode(mean))
            x_rad_fea_recon = F.relu(self.fc2_decode(mean))
            x_demo_fea_recon = F.relu(self.fc3_decode(mean))
            x_omic_fea_recon = F.relu(self.fc4_decode(mean))
            shape_m, shape_n = x_path_fea_recon.shape
            x_mask_matrix = torch.stack((x_masks["mask_path"][:, None], x_masks["mask_rad"][:, None], x_masks["mask_demo"][:, None], x_masks["mask_omic"][:, None]),dim=1)
            recon1_batch = torch.sum(((x_path_fea_recon - x_path_fea) * x_masks["mask_path"][:, None]) ** 2.0, 1)
            recon2_batch = torch.sum(((x_rad_fea_recon - x_rad_fea) * x_masks["mask_rad"][:, None]) ** 2.0,1)
            recon3_batch = torch.sum(((x_demo_fea_recon - x_demo_fea) * x_masks["mask_demo"][:, None]) ** 2.0,1)
            recon4_batch = torch.sum(((x_omic_fea_recon - x_omic_fea) * x_masks["mask_omic"][:, None]) ** 2.0,1)

            recon1 = recon1_batch/torch.sum(x_masks["mask_path"][:, None]) if torch.sum(x_masks["mask_path"][:, None]) > 0 else torch.tensor(0)
            recon2 = recon2_batch/torch.sum(
                    x_masks["mask_rad"][:, None]) if torch.sum(x_masks["mask_rad"][:, None]) > 0 else torch.tensor(0)
            recon3 = recon3_batch/ torch.sum(
                    x_masks["mask_demo"][:, None]) if torch.sum(x_masks["mask_demo"][:, None]) > 0 else torch.tensor(0)
            recon4 = recon4_batch/ torch.sum(
                    x_masks["mask_omic"][:, None]) if torch.sum(x_masks["mask_omic"][:, None]) > 0 else torch.tensor(0)
            reconstruction_loss = (torch.sum(
                ((recon1_batch + recon2_batch + recon3_batch + recon4_batch).unsqueeze(1)) / torch.sum(x_mask_matrix,
                                                                                                       1))) / shape_m
        else:
            reconstruction_loss = torch.tensor([0])
            recon1 = torch.tensor(0)
            recon2 = torch.tensor(0)
            recon3 = torch.tensor(0)
            recon4 = torch.tensor(0)
        mean_update1 = mean
        features = self.fuse_fc(mean_update1)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return {"recon_loss": reconstruction_loss.unsqueeze(0),
                "recon_path": torch.sum(recon1), "recon_rad": torch.sum(recon2), "recon_demo": torch.sum(recon3),
                "recon_omic":torch.sum(recon4)}, hazard
