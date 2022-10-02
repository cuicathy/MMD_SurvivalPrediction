import argparse
import os
### Parser

def parse_args():
    parser = argparse.ArgumentParser()
    #####################################################################################################################
    # key arguments that you may want to adjust
    parser.add_argument('--dataroot', default='../data', help="datasets")
    parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints',
                        help='models are saved here')
    parser.add_argument('--exp_name', type=str, default='surv_15',
                        help='name of the project. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--mode', type=str, default='multimodal', help='mode (multimoda/path/rad_omic/omic/demo)')
    parser.add_argument('--model_name', type=str, default='Test1', help='Name of the model')

    parser.add_argument('--use_embedding', type=bool, default=True,
                        help="True - use extracted features for the second stage")
    parser.add_argument('--use_patch', type=bool, default=False,
                        help="True - patch-wise data and results, require aggregation in the end for a patient. Always use True for testing, and False for training.")
    parser.add_argument('--random_drop_views', type=bool, default=True, help="Modality dropout")
    parser.add_argument('--recon', type=bool, default=True, help="Embedding reconstruction")
    parser.add_argument('--exp_num', type=int, default=-1)
    parser.add_argument('--keep_ratio', type=float, default=0.5,
                        help='probability to keep the modality (when random_drop_views == True)')
    parser.add_argument('--unimodal_pretrain', type=bool, default=True,
                        help='Use pretrained weights of unimodalities')
    parser.add_argument('--recon_loss_weight', type=float, default=1, help="weights of reconstruction loss")
    parser.add_argument('--required_modality', type=list, default=[], help='Each sample used for training must have all required_modality available. '
                                                                           'If required_modality == [], all samples can be used (one modality is available at least)'
                        'If required_modality == ["demo", "path", "rad", "omic"], samples with all modalities available can be used')
    #######################################################################################################################################

    parser.add_argument('--use_vgg_features', type=int, default=0, help='Use pretrained embeddings')
    parser.add_argument('--use_rnaseq', type=int, default=1, help='Use RNAseq data.')

    parser.add_argument('--model', type=str, default="Multimodal", help='model to use')
    parser.add_argument('--task', type=str, default='surv', help='This code only supports the surv task')
    parser.add_argument('--act_type', type=str, default='Sigmoid', help='activation function')
    parser.add_argument('--input_size_omic', type=int, default=80, help="input_size for omic vector")
    parser.add_argument('--input_size_path', type=int, default=512, help="input_size for path images")
    parser.add_argument('--input_size_demo', type=int, default=9, help="input_size for omic vector")
    parser.add_argument('--input_size_rad', type=int, default=120, help="input_size for rad images")
    parser.add_argument('--input_size_radiomics', type=int, default=318,
                        help="input_size for radiomics features")
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--save_at', type=int, default=20, help="adsfasdf")
    parser.add_argument('--label_dim', type=int, default=1, help='size of output')
    parser.add_argument('--measure', default=1, type=int, help='disables measure while training (make program faster)')
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--print_every', default=0, type=int)

    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.9, help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--beta2', type=float, default=0.999, help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--lr_policy', default='linear', type=str, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--finetune', default=1, type=int, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--final_lr', default=0.1, type=float, help='Used for AdaBound')
    parser.add_argument('--reg_type', default='none', type=str, help="regularization type")
    parser.add_argument('--niter', type=int, default=10, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--epoch_count', type=int, default=1, help='start of epoch')
    parser.add_argument('--epoch_count_training', type=int, default=1, help='start of epoch')
    parser.add_argument('--batch_size', type=int, default=8, help="Number of batches to train for. Default: 32")
    parser.add_argument('--batch_test_size', type=int, default=8,
                        help="Number of batches to test for. Default: 1")

    parser.add_argument('--lambda_cox', type=float, default=1)
    parser.add_argument('--lambda_reg', type=float, default=0)
    parser.add_argument('--lambda_nll', type=float, default=1)

    parser.add_argument('--fusion_type', type=str, default="pofusion", help='concat | pofusion')
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--use_bilinear', type=int, default=1)
    parser.add_argument('--path_gate', type=int, default=1)
    parser.add_argument('--grph_gate', type=int, default=1)
    parser.add_argument('--omic_gate', type=int, default=1)

    parser.add_argument('--path_dim', type=int, default=32)
    parser.add_argument('--grph_dim', type=int, default=32)
    parser.add_argument('--demo_dim', type=int, default=32)
    parser.add_argument('--omic_dim', type=int, default=32)
    parser.add_argument('--rad_dim', type=int, default=32)

    parser.add_argument('--path_scale', type=int, default=1)
    parser.add_argument('--grph_scale', type=int, default=1)
    parser.add_argument('--omic_scale', type=int, default=1)
    parser.add_argument('--mmhid', type=int, default=128, help='dimension of hidden layers in fusion')
    parser.add_argument('--init_type', type=str, default='none',
                        help='network initialization [normal | xavier | kaiming | orthogonal | max]. Max seems to work well')

    parser.add_argument('--dropout_rate', default=0.25, type=float,
                        help='0 - 0.25. Increasing dropout_rate helps overfitting. Some people have gone as high as 0.5. You can try adding more regularization')
    parser.add_argument('--pooling_ratio', default=0.2, type=float, help='pooling ratio for SAGPOOl')
    parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Used for Adam. L2 Regularization on weights. I normally turn this off if I am using L1. You should try')
    opt = parser.parse_known_args()[0]
    opt = parse_gpuids(opt)
    ori_message(parser, opt)
    return opt


def ori_message(parser, opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    return message


def print_options(opt, log_file_name, message):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format(log_file_name))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
    return file_name


def parse_gpuids(opt):
    import torch
    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    return opt


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
