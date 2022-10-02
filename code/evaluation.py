from tqdm import tqdm
from lifelines.statistics import logrank_test
import numpy as np
import pandas as pd
import scipy
from utils import *

def getPValAggSurv_GBMLGG_Binary_missingView(ckpt_name='../checkpoints/surv_15/', #
                                        model='pathgraphomic_fusion', percentile=[50],
                                        data_cv_path='../data/patches_testing_reference.pkl',
                                        single_view='rad', fold_range=[i+1 for i in range(15)]):
    data = getDataAggSurv_GBMLGG_missingView(ckpt_name=ckpt_name, model=model, data_cv_path=data_cv_path,
                                            single_view=single_view, fold_range=fold_range)
    p = np.percentile(data['Hazard'], percentile)
    data.insert(0, 'grade_pred', [hazard2grade(hazard, p) for hazard in data['Hazard']])
    T_low, T_high = data['Survival months'][data['grade_pred'] == 0], data['Survival months'][data['grade_pred'] == 1]
    E_low, E_high = data['censored'][data['grade_pred'] == 0], data['censored'][data['grade_pred'] == 1]

    low_vs_high = logrank_test(durations_A=T_low, durations_B=T_high, event_observed_A=E_low,
                               event_observed_B=E_high).p_value
    return np.array([low_vs_high])


def getPValAggSurv_GBMLGG_Multi_missingView(ckpt_name='../checkpoints/', #
                                       model='pathgraphomic_fusion', percentile=[33, 66], data_cv_path='',
                                       single_view='rad', fold_range=[i+1 for i in range(15)]):
    data = getDataAggSurv_GBMLGG_missingView(ckpt_name=ckpt_name, model=model, data_cv_path=data_cv_path,
                                            single_view=single_view, fold_range=fold_range)
    p = np.percentile(data['Hazard'], percentile)
    if p[0] == p[1]: p[0] = 2.99997
    data.insert(0, 'grade_pred', [hazard2grade(hazard, p) for hazard in data['Hazard']])
    T_low, T_mid, T_high = data['Survival months'][data['grade_pred'] == 0], data['Survival months'][
        data['grade_pred'] == 1], data['Survival months'][data['grade_pred'] == 2]
    E_low, E_mid, E_high = data['censored'][data['grade_pred'] == 0], data['censored'][data['grade_pred'] == 1], \
                           data['censored'][data['grade_pred'] == 2]
    low_vs_mid = logrank_test(durations_A=T_low, durations_B=T_mid, event_observed_A=E_low,
                              event_observed_B=E_mid).p_value
    mid_vs_high = logrank_test(durations_A=T_mid, durations_B=T_high, event_observed_A=E_mid,
                               event_observed_B=E_high).p_value
    return np.array([low_vs_mid, mid_vs_high])


def getPValAggSurv_GBMLGG_Binary(ckpt_name='../checkpoints/',#
                                 model='pathgraphomic_fusion', percentile=[50],
                                 data_cv_path='./data/patches_testing_reference.pkl', fold_range=[i+1 for i in range(15)]):
    data = getDataAggSurv_GBMLGG(ckpt_name=ckpt_name, model=model, data_cv_path=data_cv_path, fold_range=fold_range)
    p = np.percentile(data['Hazard'], percentile)
    data.insert(0, 'grade_pred', [hazard2grade(hazard, p) for hazard in data['Hazard']])
    T_low, T_high = data['Survival months'][data['grade_pred'] == 0], data['Survival months'][data['grade_pred'] == 1]
    E_low, E_high = data['censored'][data['grade_pred'] == 0], data['censored'][data['grade_pred'] == 1]

    low_vs_high = logrank_test(durations_A=T_low, durations_B=T_high, event_observed_A=E_low,
                               event_observed_B=E_high).p_value
    return np.array([low_vs_high])


def getPValAggSurv_GBMLGG_Multi(ckpt_name='../checkpoints/',#
                                model='pathgraphomic_fusion', percentile=[33, 66], data_cv_path='', fold_range = [i+1 for i in range(15)]):
    data = getDataAggSurv_GBMLGG(ckpt_name=ckpt_name, model=model, data_cv_path=data_cv_path, fold_range=fold_range)
    p = np.percentile(data['Hazard'], percentile)
    if p[0] == p[1]: p[0] = 2.99997
    data.insert(0, 'grade_pred', [hazard2grade(hazard, p) for hazard in data['Hazard']])
    T_low, T_mid, T_high = data['Survival months'][data['grade_pred'] == 0], data['Survival months'][
        data['grade_pred'] == 1], data['Survival months'][data['grade_pred'] == 2]
    E_low, E_mid, E_high = data['censored'][data['grade_pred'] == 0], data['censored'][data['grade_pred'] == 1], \
                           data['censored'][data['grade_pred'] == 2]
    low_vs_mid = logrank_test(durations_A=T_low, durations_B=T_mid, event_observed_A=E_low,
                              event_observed_B=E_mid).p_value
    mid_vs_high = logrank_test(durations_A=T_mid, durations_B=T_high, event_observed_A=E_mid,
                               event_observed_B=E_high).p_value
    return np.array([low_vs_mid, mid_vs_high])


def getPredAggSurv_GBMLGG(ckpt_name='../checkpoints/', model='pathgraphomic_fusion',#
                          split='test', use_rnaseq=True, agg_type='Hazard_mean',
                          data_cv_path='../data/patches_testing_reference.pkl', fold_range=[i+1 for i in range(15)]):
    results = []
    use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1)

    data_cv = pickle.load(open(data_cv_path, 'rb'))
    for k in fold_range:
        pred = pickle.load(open(ckpt_name + '/%s/%s_%d%spred_%s.pkl' % (model, model, k, use_patch, split), 'rb'))
        surv_all = pd.DataFrame(np.stack(np.array(pred))).T
        surv_all.columns = ['Hazard', 'Survival months', 'censored', 'TCGA ID']
        data_cv_splits = data_cv['cv_splits']
        data_cv_split_k = data_cv_splits[k]
        assert np.all(data_cv_split_k[split]['t'] == pred[1])
        all_dataset = data_cv['data_pd'].drop('TCGA ID', axis=1)
        all_dataset_regstrd = all_dataset.loc[data_cv_split_k[split][
            'x_patname']]
        assert np.all(np.array(all_dataset_regstrd['Survival months']) == pred[1])
        assert np.all(np.array(all_dataset_regstrd['censored']) == pred[2])
        all_dataset_regstrd.insert(loc=0, column='Hazard', value=np.array(surv_all['Hazard']))
        all_dataset_regstrd["Hazard"] = pd.to_numeric(all_dataset_regstrd["Hazard"], downcast="float")
        hazard_agg = all_dataset_regstrd.groupby('TCGA ID').agg({'Hazard': ['mean', 'median', max]})
        hazard_agg.columns = ["_".join(x) for x in hazard_agg.columns.ravel()]
        hazard_agg = hazard_agg[[agg_type]]
        hazard_agg.columns = ['Hazard']
        all_dataset_hazard = hazard_agg.join(all_dataset, how='inner')
        cin = CIndex_lifeline(all_dataset_hazard['Hazard'], all_dataset_hazard['censored'],
                              all_dataset_hazard['Survival months'])
        results.append(cin)
    return results


def getDataAggSurv_GBMLGG_missingView(ckpt_name='../checkpoints/surv_15/', #
                                     model='pathgraphomic_fusion',
                                     split='test', use_rnaseq=True, agg_type='Hazard_mean', zscore=False,
                                     data_cv_path='', single_view='_rad', fold_range=[i+1 for i in range(15)]):  ###CC

    data = []
    if 'cox' in model:
        for k in fold_range:
            pred = pickle.load(
                open(ckpt_name + '/%s/%s_%d_pred_%s_%s.pkl' % (model, model, k, split, single_view), 'rb'))
            data.append(pred)
        return pd.concat(data)
    else:
        data_cv = pickle.load(open(data_cv_path, 'rb'))
        use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1)

        for k in fold_range:
            temp = ckpt_name + '/%s/%s_%d%spred_%s_%s.pkl' % (model, model, k, use_patch, split, single_view)
            pred = pickle.load(open(temp, 'rb'))
            #surv_all = pd.DataFrame(np.stack(np.delete(np.array(pred), [3]))).T
            surv_all = pd.DataFrame(np.stack(np.array(pred))).T
            surv_all.columns = ['Hazard', 'Survival months', 'censored', 'TCGA ID']
            data_cv_splits = data_cv['cv_splits']
            data_cv_split_k = data_cv_splits[k]
            assert np.all(data_cv_split_k[split]['t'] == pred[1])  # Data is correctly registered
            all_dataset = data_cv['data_pd'].drop('TCGA ID', axis=1)
            all_dataset_regstrd = all_dataset.loc[data_cv_split_k[split][
                'x_patname']]  # Subset of "all_datasets" (metadata) that is registered with "pred" (predictions)
            assert np.all(np.array(all_dataset_regstrd['Survival months']) == pred[1])
            assert np.all(np.array(all_dataset_regstrd['censored']) == pred[2])
            all_dataset_regstrd.insert(loc=0, column='Hazard', value=np.array(surv_all['Hazard']))

            all_dataset_regstrd[['Hazard', 'censored', 'Survival months']] = all_dataset_regstrd[
                ['Hazard', 'censored', 'Survival months']].apply(
                pd.to_numeric)
            hazard_agg = all_dataset_regstrd.groupby('TCGA ID').agg({'Hazard': ['mean', 'median', max]})
            hazard_agg.columns = ["_".join(x) for x in hazard_agg.columns.ravel()]
            hazard_agg = hazard_agg[[agg_type]]
            hazard_agg.columns = ['Hazard']
            all_dataset_hazard = hazard_agg.join(all_dataset, how='inner')
            all_dataset_hazard['split'] = k
            if zscore: all_dataset_hazard['Hazard'] = scipy.stats.zscore(np.array(all_dataset_hazard['Hazard']))
            data.append(all_dataset_hazard)

        data = pd.concat(data)
        return data


def getDataAggSurv_GBMLGG(ckpt_name='../checkpoints/', model='pathgraphomic_fusion',
                          split='test', use_rnaseq=True, agg_type='Hazard_mean', zscore=False, data_cv_path='',
                          fold_range=[i + 1 for i in range(15)]):

    data = []
    if 'cox' in model:
        for k in fold_range:
            pred = pickle.load(open(ckpt_name + '/%s/%s_%d_pred_%s.pkl' % (model, model, k, split), 'rb'))
            data.append(pred)
        return pd.concat(data)
    else:
        data_cv = pickle.load(open(data_cv_path, 'rb'))
        use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1)

        for k in fold_range:
            temp = ckpt_name + '/%s/%s_%d%spred_%s.pkl' % (model, model, k, use_patch, split)
            pred = pickle.load(open(temp, 'rb'))
            surv_all = pd.DataFrame(np.stack(np.array(pred))).T
            surv_all.columns = ['Hazard', 'Survival months', 'censored', 'TCGA ID']
            data_cv_splits = data_cv['cv_splits']
            data_cv_split_k = data_cv_splits[k]
            assert np.all(data_cv_split_k[split]['t'] == pred[1])  # Data is correctly registered
            all_dataset = data_cv['data_pd'].drop('TCGA ID', axis=1)
            all_dataset_regstrd = all_dataset.loc[data_cv_split_k[split][
                'x_patname']]  # Subset of "all_datasets" (metadata) that is registered with "pred" (predictions)
            assert np.all(np.array(all_dataset_regstrd['Survival months']) == pred[1])
            assert np.all(np.array(all_dataset_regstrd['censored']) == pred[2])
            all_dataset_regstrd.insert(loc=0, column='Hazard', value=np.array(surv_all['Hazard']))
            all_dataset_regstrd["Hazard"] = pd.to_numeric(all_dataset_regstrd["Hazard"], downcast="float")
            hazard_agg = all_dataset_regstrd.groupby('TCGA ID').agg({'Hazard': ['mean', 'median', max]})
            hazard_agg.columns = ["_".join(x) for x in hazard_agg.columns.ravel()]
            hazard_agg = hazard_agg[[agg_type]]
            hazard_agg.columns = ['Hazard']
            all_dataset_hazard = hazard_agg.join(all_dataset, how='inner')
            all_dataset_hazard['split'] = k
            if zscore: all_dataset_hazard['Hazard'] = scipy.stats.zscore(np.array(all_dataset_hazard['Hazard']))
            data.append(all_dataset_hazard)
        data = pd.concat(data)
        return data



### Survival Outcome Prediction
def hazard2grade(hazard, p):
    for i in range(len(p)):
        if hazard < p[i]:
            return i
    return len(p)


def CI_pm(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return str("{0:.4f} ± ".format(m) + "{0:.3f}".format(h))


def CI_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return str("{0:.3f}, ".format(m - h) + "{0:.3f}".format(m + h))


def p(n):
    def percentile_(x):
        return np.percentile(x, n)

    percentile_.__name__ = 'p%s' % n
    return percentile_


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def getPredAggSurv_GBMLGG_sim(ckpt_name='../checkpoints/', model='pathgraphomic_fusion',
                              split='test', use_rnaseq=True, agg_type='Hazard_mean',
                              data_cv_path='../data/patches_testing_reference.pkl',
                              single_view='_rad', fold_range=[i + 1 for i in range(15)]):
    results = []
    use_patch, roi_dir, use_vgg_features = ('_patch_', 'all_st_patches_512', 1)

    data_cv = pickle.load(open(data_cv_path, 'rb'))
    for k in fold_range:
        pred = pickle.load(open(ckpt_name + '/%s/%s_%d%spred_%s_%s.pkl' % (model, model, k, use_patch, split, single_view), 'rb'))
        surv_all = pd.DataFrame(np.stack(np.array(pred))).T
        surv_all.columns = ['Hazard', 'Survival months', 'censored', 'TCGA ID']

        surv_all[['Hazard', 'censored', 'Survival months']] = surv_all[['Hazard', 'censored', 'Survival months']].apply(
            pd.to_numeric)
        hazard_agg = surv_all.groupby('TCGA ID').agg({'Hazard': ['mean', 'median', max]})  # .apply(pd.to_numeric)
        surv_gt_agg = surv_all.groupby('TCGA ID').agg({'censored': ['mean'], 'Survival months': ['mean']})
        surv_gt_agg.columns = ['censored', 'Survival months']
        hazard_agg.columns = ["_".join(x) for x in hazard_agg.columns.ravel()]
        hazard_agg = hazard_agg[[agg_type]]
        hazard_agg.columns = ['Hazard']
        all_dataset_hazard = hazard_agg.join(surv_gt_agg, how='inner')

        cin = CIndex_lifeline(all_dataset_hazard['Hazard'], all_dataset_hazard['censored'],
                              all_dataset_hazard['Survival months'])
        results.append(cin)
    return results

def evaluate_completeModa(model3, ckpt_name='../checkpoints/surv_15/',
                           fold_range=[i + 1 for i in range(15)]):
    print('*******************evaluate_allviews_test***************************')
    #data_cv_path_patches = '../data/patches_testing_reference.pkl'
    data_cv_path_patches = '../data/gbmlgg15cv_patches_embedding3.pkl'
    pvalue_surv_binary = [np.array(getPValAggSurv_GBMLGG_Binary(ckpt_name=ckpt_name, model=model, percentile=[50],
                                                                data_cv_path=data_cv_path_patches, fold_range=fold_range)) for model in
                          tqdm(model3)]
    pvalue_surv_binary = pd.DataFrame(np.array(pvalue_surv_binary))
    pvalue_surv = pd.concat([pvalue_surv_binary], axis=1)
    pvalue_surv.index = model3
    pvalue_surv.columns = ['P-Value (<50% vs. >50%)']

    cv_surv = [np.array(getPredAggSurv_GBMLGG(ckpt_name=ckpt_name, model=model, data_cv_path=data_cv_path_patches, fold_range=fold_range)) for
               model in tqdm(model3)]
    cv_surv = pd.DataFrame(np.array(cv_surv))
    cv_surv.columns = ['Split %s' % str(k) for k in fold_range]
    cv_surv.index = model3
    cv_surv['C-Index'] = [CI_pm(cv_surv.loc[model]) for model in model3]
    cv_surv_all = cv_surv[['C-Index']].join(pvalue_surv, how='inner')
    pd.set_option('display.max_columns', None)
    print(pd.DataFrame(cv_surv))
    print(pd.DataFrame(cv_surv_all))
    cv_surv.insert(0, "Inputs", 'allviews', True)
    cv_surv = cv_surv.join(pvalue_surv)
    temp_results = cv_surv
    return cv_surv_all, temp_results

def evaluate_missingModa(model3, ckpt_name='../checkpoints/surv_15/',
                                               model_appendidx_namelist=[],
                                               fold_range=[i + 1 for i in range(15)]):
    #data_cv_path_patches = '../data/patches_testing_reference.pkl' # Data Reference
    data_cv_path_patches = '../data/gbmlgg15cv_patches_embedding3.pkl'
    print('*****************singleViews******************')
    #temp_results = []
    for idx, view_name in enumerate(model_appendidx_namelist):
        print('###', view_name)
        pvalue_surv_binary = [
            np.array(getPValAggSurv_GBMLGG_Binary_missingView(ckpt_name=ckpt_name, model=model, percentile=[50],
                                                         data_cv_path=data_cv_path_patches, single_view=view_name, fold_range=fold_range)) for
            model in tqdm(model3)]
        pvalue_surv_binary = pd.DataFrame(np.array(pvalue_surv_binary))
        pvalue_surv = pd.concat([pvalue_surv_binary], axis=1)
        pvalue_surv.index = model3
        pvalue_surv.columns = ['P-Value (<50% vs. >50%)']

        cv_surv = [np.array(
            getPredAggSurv_GBMLGG_sim(ckpt_name=ckpt_name, model=model, data_cv_path=data_cv_path_patches,
                                      single_view=view_name, fold_range=fold_range)) for model in tqdm(model3)]
        cv_surv = pd.DataFrame(np.array(cv_surv))
        cv_surv.columns = ['Split %s' % str(k) for k in fold_range]
        cv_surv.index = model3
        cv_surv['C-Index'] = [CI_pm(cv_surv.loc[model]) for model in model3]
        cv_surv_all = cv_surv[['C-Index']]
        pd.set_option('display.max_columns', None)

        print(pd.DataFrame(cv_surv_all))

        if idx == 0:
            cv_surv.insert(0, "Inputs", view_name, True)
            cv_surv = cv_surv.join(pvalue_surv)
            temp_results = cv_surv
        else:
            cv_surv.insert(0, "Inputs", view_name, True)
            cv_surv = cv_surv.join(pvalue_surv)
            temp_results = pd.concat([temp_results, cv_surv], axis=0)

    return cv_surv_all, temp_results
