import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data as torch_data
import torch.nn.functional as F


def get_first_0_index(df_row_series):
    index = 2
    for _, value in df_row_series.iteritems():
        if value == 0:
            return index
        index += 1
    return len(df_row_series)+2


def sliding_window_split(raw_data, overlap=0, split_width=500):
    '''
    raw_data is the df_row Series with address and label
    return address, slices in np array, label
    '''

    # cut 0's
    zero_index = get_first_0_index(raw_data[2:])
    data = raw_data[2:zero_index]
    sample_width = data.shape[-1]
    index = 0
    splits = []

    while index + split_width < sample_width:
        splits.append(data[index:index + split_width])
        index += split_width - overlap

    if sample_width - index > split_width / 8:
        slice_from = -split_width if sample_width > split_width else 0
        splits.append(data[slice_from:])

    return raw_data[0], np.array(splits), raw_data[1]


def pad_zeros(np_array, build_width=500):
    '''for slice with smaller size, pad zeros and make them have standard size'''
    result = np.zeros(shape=(1,build_width), dtype=int)
    result[:np_array.shape[0],:np_array.shape[1]] = np_array
    return result


def build_dataset(dataframe, build_width=500):
    '''
        param: original datafeame
        return:
    '''
    all_data = np.zeros(shape=(1, build_width), dtype=int)  # containing all slices
    labels = []
    print('start building dataset')
    for index, row in dataframe.iterrows():
        print('processing ', (index / len(dataframe)) * 100, ', index ', index, ' / ', len(dataframe))
        curr_addr, cur_slices, cur_label = sliding_window_split(row)

        #         print('cur slices shape: ', cur_slices.shape)
        #         print('all data shape: ', all_data.shape)

        #         print(type(cur_slices.shape))
        #         print(type(cur_slices.shape[1]))
        #         print(type(cur_slices.shape[1]))
        #         print('cur slices content: ', cur_slices)

        if cur_slices.shape[0] > 0:
            if cur_slices.shape[1] < 500:
                #                 print('old cur slices: ', cur_slices)
                cur_slices = pad_zeros(cur_slices, build_width)
            #                 print('new cur slices: ', cur_slices)
            #                 print('new cur slices shape: ', cur_slices.shape)

            all_data = np.concatenate((all_data, cur_slices), axis=0)
            for i in range(cur_slices.shape[0]):
                labels.append(cur_label)
        else:
            #             print('zero found: ', cur_slices.shape)
            #             print('zero content: ', cur_slices)
            print('empty contract found: ', curr_addr)

    all_data = all_data[1:]
    return all_data, labels


# df = pd.read_csv('../dataset/op_int_equal_dataframe/op_int_equal_new.csv')
df = pd.read_csv('./datasets/op_int_equal_new.csv')

print(df.head())
print(df.shape)
print(df.label.value_counts())

# 0~62 ponzi test dataset
# 63~182 ponzi train dataset
# 183~4553 non ponzi train dataset
# 4554~-1 non ponzi test dataset

train_df = df.iloc[63:4553, :]
ponzi_test = df.iloc[0:63, :]
non_ponzi_test = df.iloc[4553:, :]

# get the test and train df
test_df = pd.concat([ponzi_test, non_ponzi_test])
print(test_df.shape)
train_df
print(train_df)


class PandasReader(torch_data.Dataset):
    def __init__(self, dataset, labels):
        self.data = dataset
        self.target = labels
        self.shape = self.data.shape

    def __getitem__(self, i):
        features = torch.from_numpy(self.data[i].astype(int))
        target = torch.tensor(self.target[i])
        return features, target

    def __len__(self):
        return self.data.shape[0]


class RNN(nn.Module):
    def __init__(self, input_dim, unique_codes):
        super(RNN, self).__init__()
        # self. emb = torch.nn.Embedding(unique_codes, 64)
        # self.lstm_1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)
        # self.fc_1 = nn.Linear(in_features=64, out_features=16)
        # self.fc_2 = nn.Linear(in_features=16, out_features=1)
        self.emb = torch.nn.Embedding(unique_codes, 32)
        self.lstm_1 = nn.LSTM(input_size=32, hidden_size=32, num_layers=2, batch_first=True)
        self.fc_1 = nn.Linear(in_features=32, out_features=8)
        self.fc_2 = nn.Linear(in_features=8, out_features=1)
        self.dropout = nn.Dropout()
        self.hidden = None

    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = self.emb(x)
        if x.size(0) < 90:

            import pdb; pdb.set_trace()
        y, self.hidden = self.lstm_1(x, self.hidden)
        y = self.dropout(y[:,-1])
        y = self.dropout(F.leaky_relu(self.fc_1(y)))
        y = self.dropout(F.leaky_relu(self.fc_2(y)))
        return y


def train(train_loader, val_loader, epochs):
    model = RNN(train_loader.dataset.shape[1], 262).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0

        count = 0

        for _, (features, target) in enumerate(train_loader):
            print(f'processing {count/len(train_loader)}, {count}/{len(train_loader)}')
            count += 1

            features = features.to(device)
            # import pdb; pdb.set_trace()
            target = target.to(device).double()

            print(40 * '-')
            print('Target')
            print(target)
            print(40 * '-')

            x = features
            y = model.forward(x).squeeze(dim=-1)
            loss = criterion(y, target)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            train_loss += loss.item()
            optimizer.step()
            print(f"EPOCH {epoch+1}: TRAIN LOSS - {train_loss}")
        train_loss = train_loss / len(train_loader)

        with torch.no_grad():
            model.eval()
            for _, (features, target) in enumerate(val_loader):
                x = features.unsqueeze(dim=0)
                preds = model.forward(x).squeeze(dim=0)
                loss = criterion(preds, target)
                val_loss += loss.item()
            model.train()
            val_loss = val_loss / len(val_loader)

        print(f"EPOCH {epoch+1}: TRAIN LOSS - {train_loss}, VAL LOSS - {val_loss}")

    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("USING DEVICE:", device)

#     torch.set_default_tensor_type("torch.DoubleTensor")
torch.set_default_dtype(torch.float64)

train_dataset, train_labels = build_dataset(train_df)
test_dataset, test_labels = build_dataset(test_df)
#     train_reader = PandasReader()

#     df_train = pd.read_csv("./train.csv").drop("Date", axis="columns")
#     train_loader = torch_data.DataLoader(PandasReader(train_df), batch_size=30, shuffle=True)

#     df_val = pd.read_csv("./val.csv").drop("Date", axis="columns")
#     val_loader = torch_data.DataLoader(PandasReader(test_df), batch_size=30, shuffle=False)

#     print("TRAINING DATA SHAPE:", train_loader.dataset.shape)
#     print("VALIDATION DATA SHAPE:", val_loader.dataset.shape)

#     model = train(train_loader, val_loader, epochs=50)

train_reader = PandasReader(train_dataset, train_labels)
test_reader = PandasReader(test_dataset, test_labels)

# train_loader = torch_data.DataLoader(train_reader, batch_size=30, shuffle=True)
# val_loader = torch_data.DataLoader(test_reader, batch_size=30, shuffle=False)
train_loader = torch_data.DataLoader(train_reader, batch_size=90, shuffle=True, drop_last=True)
val_loader = torch_data.DataLoader(test_reader, batch_size=90, shuffle=False, drop_last=True)

print("TRAINING DATA SHAPE:", train_loader.dataset.shape)
print("VALIDATION DATA SHAPE:", val_loader.dataset.shape)

# model = train(train_loader, val_loader, epochs=50)
model = train(train_loader, val_loader, epochs=2)