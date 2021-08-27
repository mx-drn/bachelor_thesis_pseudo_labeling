import pandas as pd
import numpy as np
import ast


def average_train_data_and_calc_std(data_class):
    overall_dict = {
        'tradi': {key + 1: {
            'train_acc': [],
            'train_precision': [],
            'train_recall': [],
            'train_f1': [],
            'train_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_loss': []
        } for key in range(10)},
        'nssdl': {key + 1: {
            'train_acc': [],
            'train_precision': [],
            'train_recall': [],
            'train_f1': [],
            'train_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_loss': []
        } for key in range(10)},
        'nssdl_strong': {key + 1: {
            'train_acc': [],
            'train_precision': [],
            'train_recall': [],
            'train_f1': [],
            'train_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_loss': []
        } for key in range(10)},
        'mixtext': {key + 1: {
            'train_acc': [],
            'train_precision': [],
            'train_recall': [],
            'train_f1': [],
            'train_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_loss': []
        } for key in range(10)}
    }

    for kind in list(overall_dict.keys()):
        for fold in range(10):
            df_temp = pd.read_excel(
                f"./histories/data_classes/{data_class}/fold_{fold + 1}/{kind}/training_data_from_fold.xlsx")

            for index, row in df_temp.iterrows():
                overall_dict[kind][row['epoch']]['train_acc'].append(row['train_acc'])
                overall_dict[kind][row['epoch']]['train_precision'].append(row['train_precision'])
                overall_dict[kind][row['epoch']]['train_recall'].append(row['train_recall'])
                overall_dict[kind][row['epoch']]['train_f1'].append(row['train_f1'])
                overall_dict[kind][row['epoch']]['train_loss'].append(row['train_loss'])
                overall_dict[kind][row['epoch']]['val_acc'].append(row['val_acc'])
                overall_dict[kind][row['epoch']]['val_precision'].append(row['val_precision'])
                overall_dict[kind][row['epoch']]['val_recall'].append(row['val_recall'])
                overall_dict[kind][row['epoch']]['val_f1'].append(row['val_f1'])
                overall_dict[kind][row['epoch']]['val_loss'].append(row['val_loss'])

    tradi_averaged_train = [{
        'model': 'tradi',
        'epoch': epoch + 1,
        'train_acc': round(np.mean(overall_dict['tradi'][epoch + 1]['train_acc']), 4),
        'train_precision': round(np.mean(overall_dict['tradi'][epoch + 1]['train_precision']), 4),
        'train_recall': round(np.mean(overall_dict['tradi'][epoch + 1]['train_recall']), 4),
        'train_f1': round(np.mean(overall_dict['tradi'][epoch + 1]['train_f1']), 4),
        'train_loss': round(np.mean(overall_dict['tradi'][epoch + 1]['train_loss']), 4),
        'val_acc': round(np.mean(overall_dict['tradi'][epoch + 1]['val_acc']), 4),
        'val_precision': round(np.mean(overall_dict['tradi'][epoch + 1]['val_precision']), 4),
        'val_recall': round(np.mean(overall_dict['tradi'][epoch + 1]['val_recall']), 4),
        'val_f1': round(np.mean(overall_dict['tradi'][epoch + 1]['val_f1']), 4),
        'val_loss': round(np.mean(overall_dict['tradi'][epoch + 1]['val_loss']), 4)
    } for epoch in range(10)]

    nssdl_averaged_train = [{
        'model': 'nssdl',
        'epoch': epoch + 1,
        'train_acc': round(np.mean(overall_dict['nssdl'][epoch + 1]['train_acc']), 4),
        'train_precision': round(np.mean(overall_dict['nssdl'][epoch + 1]['train_precision']), 4),
        'train_recall': round(np.mean(overall_dict['nssdl'][epoch + 1]['train_recall']), 4),
        'train_f1': round(np.mean(overall_dict['nssdl'][epoch + 1]['train_f1']), 4),
        'train_loss': round(np.mean(overall_dict['nssdl'][epoch + 1]['train_loss']), 4),
        'val_acc': round(np.mean(overall_dict['nssdl'][epoch + 1]['val_acc']), 4),
        'val_precision': round(np.mean(overall_dict['nssdl'][epoch + 1]['val_precision']), 4),
        'val_recall': round(np.mean(overall_dict['nssdl'][epoch + 1]['val_recall']), 4),
        'val_f1': round(np.mean(overall_dict['nssdl'][epoch + 1]['val_f1']), 4),
        'val_loss': round(np.mean(overall_dict['nssdl'][epoch + 1]['val_loss']), 4)
    } for epoch in range(10)]

    nssdl_strong_averaged_train = [{
        'model': 'nssdl_strong',
        'epoch': epoch + 1,
        'train_acc': round(np.mean(overall_dict['nssdl_strong'][epoch + 1]['train_acc']), 4),
        'train_precision': round(np.mean(overall_dict['nssdl_strong'][epoch + 1]['train_precision']), 4),
        'train_recall': round(np.mean(overall_dict['nssdl_strong'][epoch + 1]['train_recall']), 4),
        'train_f1': round(np.mean(overall_dict['nssdl_strong'][epoch + 1]['train_f1']), 4),
        'train_loss': round(np.mean(overall_dict['nssdl_strong'][epoch + 1]['train_loss']), 4),
        'val_acc': round(np.mean(overall_dict['nssdl_strong'][epoch + 1]['val_acc']), 4),
        'val_precision': round(np.mean(overall_dict['nssdl_strong'][epoch + 1]['val_precision']), 4),
        'val_recall': round(np.mean(overall_dict['nssdl_strong'][epoch + 1]['val_recall']), 4),
        'val_f1': round(np.mean(overall_dict['nssdl_strong'][epoch + 1]['val_f1']), 4),
        'val_loss': round(np.mean(overall_dict['nssdl_strong'][epoch + 1]['val_loss']), 4)
    } for epoch in range(10)]

    mixtext_averaged_train = [{
        'model': 'mixtext',
        'epoch': epoch + 1,
        'train_acc': round(np.mean(overall_dict['mixtext'][epoch + 1]['train_acc']), 4),
        'train_precision': round(np.mean(overall_dict['mixtext'][epoch + 1]['train_precision']), 4),
        'train_recall': round(np.mean(overall_dict['mixtext'][epoch + 1]['train_recall']), 4),
        'train_f1': round(np.mean(overall_dict['mixtext'][epoch + 1]['train_f1']), 4),
        'train_loss': round(np.mean(overall_dict['mixtext'][epoch + 1]['train_loss']), 4),
        'val_acc': round(np.mean(overall_dict['mixtext'][epoch + 1]['val_acc']), 4),
        'val_precision': round(np.mean(overall_dict['mixtext'][epoch + 1]['val_precision']), 4),
        'val_recall': round(np.mean(overall_dict['mixtext'][epoch + 1]['val_recall']), 4),
        'val_f1': round(np.mean(overall_dict['mixtext'][epoch + 1]['val_f1']), 4),
        'val_loss': round(np.mean(overall_dict['mixtext'][epoch + 1]['val_loss']), 4)
    } for epoch in range(10)]

    tradi_averaged_train = pd.DataFrame(tradi_averaged_train)
    nssdl_averaged_train = pd.DataFrame(nssdl_averaged_train)
    nssdl_strong_averaged_train = pd.DataFrame(nssdl_strong_averaged_train)
    mixtext_averaged_train = pd.DataFrame(mixtext_averaged_train)

    tradi_averaged_train.to_excel(f"./histories/data_classes/{data_class}/average/tradi/averaged_train_data.xlsx")
    nssdl_averaged_train.to_excel(f"./histories/data_classes/{data_class}/average/nssdl/averaged_train_data.xlsx")
    nssdl_strong_averaged_train.to_excel(
        f"./histories/data_classes/{data_class}/average/nssdl_strong/averaged_train_data.xlsx")
    mixtext_averaged_train.to_excel(f"./histories/data_classes/{data_class}/average/mixtext/averaged_train_data.xlsx")

    print(f"Traindata of class: {data_class} was averaged and saved successfully!")


#average_train_data_and_calc_std(10)
#average_train_data_and_calc_std(50)
#average_train_data_and_calc_std(250)
#average_train_data_and_calc_std(500)
#average_train_data_and_calc_std(1300)