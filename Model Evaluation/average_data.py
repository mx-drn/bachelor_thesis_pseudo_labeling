import pandas as pd
import numpy as np
import ast


def average_data_and_calc_std(data_class):
    traditional = {
        'average_acc': [],
        'average_prec': [],
        'average_recall': [],
        'average_f1': [],
        'average_loss': [],
        'average_time': [],
        'average_frequency': {
            0: [],
            1: [],
            2: []
        }
    }

    nssdl = {
        'average_acc': [],
        'average_prec': [],
        'average_recall': [],
        'average_f1': [],
        'average_loss': [],
        'average_time': [],
        'average_frequency': {
            0: [],
            1: [],
            2: []
        }
    }

    nssdl_strong = {
        'average_acc': [],
        'average_prec': [],
        'average_recall': [],
        'average_f1': [],
        'average_loss': [],
        'average_time': [],
        'average_frequency': {
            0: [],
            1: [],
            2: []
        }
    }

    mixtext = {
        'average_acc': [],
        'average_prec': [],
        'average_recall': [],
        'average_f1': [],
        'average_loss': [],
        'average_time': [],
        'average_frequency': {
            0: [],
            1: [],
            2: []
        }
    }

    for fold in range(10):
        df_temp = pd.read_excel(f"./{data_class}/class_{data_class}_test_data_from_fold_{fold}.xlsx")

        for index, row in df_temp.iterrows():
            if row['model'] == 'tradi':
                traditional['average_acc'].append(row['test_acc'])
                traditional['average_prec'].append(row['test_precision'])
                traditional['average_recall'].append(row['test_recall'])
                traditional['average_f1'].append(row['test_f1'])
                traditional['average_loss'].append(row['test_loss'])
                traditional['average_time'].append(row['train_time'])

                freq_dict = ast.literal_eval(row['frequency of precited labels'])

                traditional['average_frequency'][0].append(freq_dict[0]) if 0 in list(freq_dict.keys()) else \
                    traditional['average_frequency'][0].append(0)
                traditional['average_frequency'][1].append(freq_dict[1]) if 1 in list(freq_dict.keys()) else \
                    traditional['average_frequency'][1].append(0)
                traditional['average_frequency'][2].append(freq_dict[2]) if 2 in list(freq_dict.keys()) else \
                    traditional['average_frequency'][1].append(0)
            elif row['model'] == 'nssdl':
                nssdl['average_acc'].append(row['test_acc'])
                nssdl['average_prec'].append(row['test_precision'])
                nssdl['average_recall'].append(row['test_recall'])
                nssdl['average_f1'].append(row['test_f1'])
                nssdl['average_loss'].append(row['test_loss'])
                nssdl['average_time'].append(row['train_time'])

                freq_dict = ast.literal_eval(row['frequency of precited labels'])

                nssdl['average_frequency'][0].append(freq_dict[0]) if 0 in list(freq_dict.keys()) else \
                    nssdl['average_frequency'][0].append(0)
                nssdl['average_frequency'][1].append(freq_dict[1]) if 1 in list(freq_dict.keys()) else \
                    nssdl['average_frequency'][1].append(0)
                nssdl['average_frequency'][2].append(freq_dict[2]) if 2 in list(freq_dict.keys()) else \
                    nssdl['average_frequency'][1].append(0)
            elif row['model'] == 'nssdl_strong':
                nssdl_strong['average_acc'].append(row['test_acc'])
                nssdl_strong['average_prec'].append(row['test_precision'])
                nssdl_strong['average_recall'].append(row['test_recall'])
                nssdl_strong['average_f1'].append(row['test_f1'])
                nssdl_strong['average_loss'].append(row['test_loss'])
                nssdl_strong['average_time'].append(row['train_time'])

                freq_dict = ast.literal_eval(row['frequency of precited labels'])

                nssdl_strong['average_frequency'][0].append(freq_dict[0]) if 0 in list(freq_dict.keys()) else \
                    nssdl_strong['average_frequency'][0].append(0)
                nssdl_strong['average_frequency'][1].append(freq_dict[1]) if 1 in list(freq_dict.keys()) else \
                    nssdl_strong['average_frequency'][1].append(0)
                nssdl_strong['average_frequency'][2].append(freq_dict[2]) if 2 in list(freq_dict.keys()) else \
                    nssdl_strong['average_frequency'][1].append(0)
            elif row['model'] == 'mixtext':
                mixtext['average_acc'].append(row['test_acc'])
                mixtext['average_prec'].append(row['test_precision'])
                mixtext['average_recall'].append(row['test_recall'])
                mixtext['average_f1'].append(row['test_f1'])
                mixtext['average_loss'].append(row['test_loss'])
                mixtext['average_time'].append(row['train_time'])

                freq_dict = ast.literal_eval(row['frequency of precited labels'])

                mixtext['average_frequency'][0].append(freq_dict[0]) if 0 in list(freq_dict.keys()) else \
                    mixtext['average_frequency'][0].append(0)
                mixtext['average_frequency'][1].append(freq_dict[1]) if 1 in list(freq_dict.keys()) else \
                    mixtext['average_frequency'][1].append(0)
                mixtext['average_frequency'][2].append(freq_dict[2]) if 2 in list(freq_dict.keys()) else \
                    mixtext['average_frequency'][1].append(0)
            else:
                raise ("Row does not inherit valid model.")

    tradi_avg_freq = {
        0: round(np.mean(traditional['average_frequency'][0]), 4),
        1: round(np.mean(traditional['average_frequency'][1]), 4),
        2: round(np.mean(traditional['average_frequency'][2]), 4)
    }

    tradi_std_freq = {
        0: round(np.std(traditional['average_frequency'][0]), 4),
        1: round(np.std(traditional['average_frequency'][1]), 4),
        2: round(np.std(traditional['average_frequency'][2]), 4)
    }

    nssdl_avg_freq = {
        0: round(np.mean(nssdl['average_frequency'][0]), 4),
        1: round(np.mean(nssdl['average_frequency'][1]), 4),
        2: round(np.mean(nssdl['average_frequency'][2]), 4)
    }

    nssdl_std_freq = {
        0: round(np.std(nssdl['average_frequency'][0]), 4),
        1: round(np.std(nssdl['average_frequency'][1]), 4),
        2: round(np.std(nssdl['average_frequency'][2]), 4)
    }

    nssdl_strong_avg_freq = {
        0: round(np.mean(nssdl_strong['average_frequency'][0]), 4),
        1: round(np.mean(nssdl_strong['average_frequency'][1]), 4),
        2: round(np.mean(nssdl_strong['average_frequency'][2]), 4)
    }

    nssdl_strong_std_freq = {
        0: round(np.std(nssdl_strong['average_frequency'][0]), 4),
        1: round(np.std(nssdl_strong['average_frequency'][1]), 4),
        2: round(np.std(nssdl_strong['average_frequency'][2]), 4)
    }

    mixtext_avg_freq = {
        0: round(np.mean(mixtext['average_frequency'][0]), 4),
        1: round(np.mean(mixtext['average_frequency'][1]), 4),
        2: round(np.mean(mixtext['average_frequency'][2]), 4)
    }

    mixtext_std_freq = {
        0: round(np.std(mixtext['average_frequency'][0]), 4),
        1: round(np.std(mixtext['average_frequency'][1]), 4),
        2: round(np.std(mixtext['average_frequency'][2]), 4)
    }

    test_dict = {
        'tradi': {
            'avg': {
                'acc': round(np.mean(traditional['average_acc']), 4),
                'precision': round(np.mean(traditional['average_prec']), 4),
                'recall': round(np.mean(traditional['average_recall']), 4),
                'f1': round(np.mean(traditional['average_f1']), 4)
            },
            'std': {
                'acc': round(np.std(traditional['average_acc']), 4),
                'precision': round(np.std(traditional['average_prec']), 4),
                'recall': round(np.std(traditional['average_recall']), 4),
                'f1': round(np.std(traditional['average_f1']), 4)
            }
        },
        'nssdl': {
            'avg': {
                'acc': round(np.mean(nssdl['average_acc']), 4),
                'precision': round(np.mean(nssdl['average_prec']), 4),
                'recall': round(np.mean(nssdl['average_recall']), 4),
                'f1': round(np.mean(nssdl['average_f1']), 4)
            },
            'std': {
                'acc': round(np.std(nssdl['average_acc']), 4),
                'precision': round(np.std(nssdl['average_prec']), 4),
                'recall': round(np.std(nssdl['average_recall']), 4),
                'f1': round(np.std(nssdl['average_f1']), 4)
            }
        },
        'nssdl_strong': {
            'avg': {
                'acc': round(np.mean(nssdl_strong['average_acc']), 4),
                'precision': round(np.mean(nssdl_strong['average_prec']), 4),
                'recall': round(np.mean(nssdl_strong['average_recall']), 4),
                'f1': round(np.mean(nssdl_strong['average_f1']), 4)
            },
            'std': {
                'acc': round(np.std(nssdl_strong['average_acc']), 4),
                'precision': round(np.std(nssdl_strong['average_prec']), 4),
                'recall': round(np.std(nssdl_strong['average_recall']), 4),
                'f1': round(np.std(nssdl_strong['average_f1']), 4)
            }
        },
        'mixtext': {
            'avg': {
                'acc': round(np.mean(mixtext['average_acc']), 4),
                'precision': round(np.mean(mixtext['average_prec']), 4),
                'recall': round(np.mean(mixtext['average_recall']), 4),
                'f1': round(np.mean(mixtext['average_f1']), 4)
            },
            'std': {
                'acc': round(np.std(mixtext['average_acc']), 4),
                'precision': round(np.std(mixtext['average_prec']), 4),
                'recall': round(np.std(mixtext['average_recall']), 4),
                'f1': round(np.std(mixtext['average_f1']), 4)
            }
        }
    }

    averaged_tests = [{
        'model': 'tradi',
        'acc': f"avg: {round(np.mean(traditional['average_acc']), 4)} || std: {round(np.std(traditional['average_acc']), 4)}",
        'precision': f"avg: {round(np.mean(traditional['average_prec']), 4)} || std: {round(np.std(traditional['average_prec']), 4)}",
        'recall': f"avg: {round(np.mean(traditional['average_recall']), 4)} || std: {round(np.std(traditional['average_recall']), 4)}",
        'f1': f"avg: {round(np.mean(traditional['average_f1']), 4)} || std: {round(np.std(traditional['average_f1']), 4)}",
        'loss': f"avg: {round(np.mean(traditional['average_loss']), 4)} || std: {round(np.std(traditional['average_loss']), 4)}",
        'time': f"avg: {round(np.mean(traditional['average_time']), 4)} || std: {round(np.std(traditional['average_time']), 4)}",
        'frequency of precited labels': f"avg: {tradi_avg_freq} || std: {tradi_std_freq}"
    },

        {
            'model': 'nssdl',
            'acc': f"avg: {round(np.mean(nssdl['average_acc']), 4)} || std: {round(np.std(nssdl['average_acc']), 4)}",
            'precision': f"avg: {round(np.mean(nssdl['average_prec']), 4)} || std: {round(np.std(nssdl['average_prec']), 4)}",
            'recall': f"avg: {round(np.mean(nssdl['average_recall']), 4)} || std: {round(np.std(nssdl['average_recall']), 4)}",
            'f1': f"avg: {round(np.mean(nssdl['average_f1']), 4)} || std: {round(np.std(nssdl['average_f1']), 4)}",
            'loss': f"avg: {round(np.mean(nssdl['average_loss']), 4)} || std: {round(np.std(nssdl['average_loss']), 4)}",
            'time': f"avg: {round(np.mean(nssdl['average_time']), 4)} || std: {round(np.std(nssdl['average_time']), 4)}",
            'frequency of precited labels': f"avg: {nssdl_avg_freq} || std: {nssdl_std_freq}"
        },

        {
            'model': 'nssdl_strong',
            'acc': f"avg: {round(np.mean(nssdl_strong['average_acc']), 4)} || std: {round(np.std(nssdl_strong['average_acc']), 4)}",
            'precision': f"avg: {round(np.mean(nssdl_strong['average_prec']), 4)} || std: {round(np.std(nssdl_strong['average_prec']), 4)}",
            'recall': f"avg: {round(np.mean(nssdl_strong['average_recall']), 4)} || std: {round(np.std(nssdl_strong['average_recall']), 4)}",
            'f1': f"avg: {round(np.mean(nssdl_strong['average_f1']), 4)} || std: {round(np.std(nssdl_strong['average_f1']), 4)}",
            'loss': f"avg: {round(np.mean(nssdl_strong['average_loss']), 4)} || std: {round(np.std(nssdl_strong['average_loss']), 4)}",
            'time': f"avg: {round(np.mean(nssdl_strong['average_time']), 4)} || std: {round(np.std(nssdl_strong['average_time']), 4)}",
            'frequency of precited labels': f"avg: {nssdl_strong_avg_freq} || std: {nssdl_strong_std_freq}"
        },

        {
            'model': 'mixtext',
            'acc': f"avg: {round(np.mean(mixtext['average_acc']), 4)} || std: {round(np.std(mixtext['average_acc']), 4)}",
            'precision': f"avg: {round(np.mean(mixtext['average_prec']), 4)} || std: {round(np.std(mixtext['average_prec']), 4)}",
            'recall': f"avg: {round(np.mean(mixtext['average_recall']), 4)} || std: {round(np.std(mixtext['average_recall']), 4)}",
            'f1': f"avg: {round(np.mean(mixtext['average_f1']), 4)} || std: {round(np.std(mixtext['average_f1']), 4)}",
            'loss': f"avg: {round(np.mean(mixtext['average_loss']), 4)} || std: {round(np.std(mixtext['average_loss']), 4)}",
            'time': f"avg: {round(np.mean(mixtext['average_time']), 4)} || std: {round(np.std(mixtext['average_time']), 4)}",
            'frequency of precited labels': f"avg: {mixtext_avg_freq} || std: {mixtext_std_freq}"
        }]

    averaged_tests = pd.DataFrame(averaged_tests)

    averaged_tests.to_excel(f"./{data_class}/averaged_class_{data_class}_test_data.xlsx")

    print(f"Testdata of class: {data_class} was averaged and saved successfully!")

    all_times = [
        {
            'data_class': data_class,
            'model': 'tradi',
            'time': round(np.mean(traditional['average_time']), 4)
        },
        {
            'data_class': data_class,
            'model': 'nssdl',
            'time': round(np.mean(nssdl['average_time']), 4)
        },
        {
            'data_class': data_class,
            'model': 'nssdl_strong',
            'time': round(np.mean(nssdl_strong['average_time']), 4)
        },
        {
            'data_class': data_class,
            'model': 'mixtext',
            'time': round(np.mean(mixtext['average_time']), 4)
        }
    ]

    return all_times, test_dict


time10, test_dict10 = average_data_and_calc_std(10)
time50, test_dict50 = average_data_and_calc_std(50)
time250, test_dict250 = average_data_and_calc_std(250)
time500, test_dict500 = average_data_and_calc_std(500)
time1300, test_dict1300 = average_data_and_calc_std(1300)

overall_average = []

for kind in ['tradi', 'nssdl', 'nssdl_strong', 'mixtext']:
    avg_acc = 0
    avg_prec = 0
    avg_recall = 0
    avg_f1 = 0

    std_acc = 0
    std_prec = 0
    std_recall = 0
    std_f1 = 0

    for dict in [test_dict10, test_dict50, test_dict250, test_dict500, test_dict1300]:
        avg_acc += dict[kind]['avg']['acc']
        avg_prec += dict[kind]['avg']['precision']
        avg_recall += dict[kind]['avg']['recall']
        avg_f1 += dict[kind]['avg']['f1']

        std_acc += dict[kind]['std']['acc']
        std_prec += dict[kind]['std']['precision']
        std_recall += dict[kind]['std']['recall']
        std_f1 += dict[kind]['std']['f1']

    overall_average.append({
            'model': kind,
            'acc': f"avg: {round(avg_acc/5, 4)} || std: {round(std_acc/5, 4)}",
            'precision': f"avg: {round(avg_prec/5, 4)} || std: {round(std_prec/5, 4)}",
            'recall': f"avg: {round(avg_recall/5, 4)} || std: {round(std_recall/5, 4)}",
            'f1': f"avg: {round(avg_f1/5, 4)} || std: {round(std_f1/5, 4)}"
        })

overall_average = pd.DataFrame(overall_average)

overall_average.to_excel(f"./averaged_test_data_over_all_classes.xlsx")

print(f"All Testdata was averaged and saved successfully!")

#all_times = time10 + time50 + time250 + time500 + time1300
#all_times = pd.DataFrame(all_times)
#all_times.to_excel("./all_training_times.xlsx")

