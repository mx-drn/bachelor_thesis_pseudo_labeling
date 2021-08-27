import pandas as pd


def build_acc_dataframe_for_signifcance_testing(data_class):
    to_boxplot = {
        'tradi': [],
        'nssdl': [],
        'nssdl_strong': [],
        'mixtext': []
    }

    for fold in range(10):
        df_temp = pd.read_excel(f"./{data_class}/class_{data_class}_test_data_from_fold_{fold}.xlsx")

        for index, row in df_temp.iterrows():
            if row['model'] == 'tradi':
                to_boxplot['tradi'].append(row['test_acc'])
            elif row['model'] == 'nssdl':
                to_boxplot['nssdl'].append(row['test_acc'])
            elif row['model'] == 'nssdl_strong':
                to_boxplot['nssdl_strong'].append(row['test_acc'])
            elif row['model'] == 'mixtext':
                to_boxplot['mixtext'].append(row['test_acc'])
            else:
                raise ("Row does not inherit valid model.")


    all_accs = [{
        'traditional': to_boxplot['tradi'][i],
        'nssdl': to_boxplot['nssdl'][i],
        'nssdl_strong': to_boxplot['nssdl_strong'][i],
        'mixtext': to_boxplot['mixtext'][i]
    } for i in range(10)]

    all_accs = pd.DataFrame(all_accs)
    all_accs.reset_index(drop=True, inplace=True)

    all_accs.to_excel(f"./{data_class}/all_accs_{data_class}_test_data.xlsx", index=False)

    print(f"Accuracies of class: {data_class} were extracted and saved successfully!")

def build_f1_dataframe_for_signifcance_testing(data_class):
    to_boxplot = {
        'tradi': [],
        'nssdl': [],
        'nssdl_strong': [],
        'mixtext': []
    }

    for fold in range(10):
        df_temp = pd.read_excel(f"./{data_class}/class_{data_class}_test_data_from_fold_{fold}.xlsx")

        for index, row in df_temp.iterrows():
            if row['model'] == 'tradi':
                to_boxplot['tradi'].append(row['test_f1'])
            elif row['model'] == 'nssdl':
                to_boxplot['nssdl'].append(row['test_f1'])
            elif row['model'] == 'nssdl_strong':
                to_boxplot['nssdl_strong'].append(row['test_f1'])
            elif row['model'] == 'mixtext':
                to_boxplot['mixtext'].append(row['test_f1'])
            else:
                raise ("Row does not inherit valid model.")


    all_accs = [{
        'traditional': to_boxplot['tradi'][i],
        'nssdl': to_boxplot['nssdl'][i],
        'nssdl_strong': to_boxplot['nssdl_strong'][i],
        'mixtext': to_boxplot['mixtext'][i]
    } for i in range(10)]

    all_accs = pd.DataFrame(all_accs)
    all_accs.reset_index(drop=True, inplace=True)

    all_accs.to_excel(f"./{data_class}/all_f1s_{data_class}_test_data.xlsx", index=False)

    print(f"F1-Scores of class: {data_class} were extracted and saved successfully!")


build_f1_dataframe_for_signifcance_testing(10)
build_f1_dataframe_for_signifcance_testing(50)
build_f1_dataframe_for_signifcance_testing(250)
build_f1_dataframe_for_signifcance_testing(500)
build_f1_dataframe_for_signifcance_testing(1300)
