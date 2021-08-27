import pandas as pd
import matplotlib.pyplot as plt


x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for data_class in [10, 50, 250, 500, 1300]:
    for kind in ['tradi', 'nssdl', 'nssdl_strong', 'mixtext']:
        learning_data = pd.read_excel(f"./histories/data_classes/{data_class}/average/{kind}/averaged_train_data.xlsx")

        y_train = []
        y_val = []
        y_train_acc = []
        y_val_acc = []

        for epoch in x:
            for index, row in learning_data.iterrows():
                if epoch == row['epoch']:
                    y_train.append(row['train_loss'])
                    y_val.append(row['val_loss'])
                    y_train_acc.append(row['train_acc'])
                    y_val_acc.append(row['val_acc'])

        plt.plot(x, y_train, label='Training loss')
        plt.plot(x, y_val, label='Validation loss')
        plt.plot(x, y_train_acc, label='Training accuracy')
        plt.plot(x, y_val_acc, label='Validation accuracy')

        plt.xlabel('Epochs')
        plt.ylabel('Loss / Accuracy')
        plt.savefig(f"./histories/data_classes/{data_class}/average/{kind}/averaged_train_data")
        plt.clf()