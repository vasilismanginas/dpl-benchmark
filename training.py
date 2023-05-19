import torch

from data import CaviarFrames, CAVIARDataset
from network import LSTM_net
from deepproblog.network import Network
from model import Model
from deepproblog.engines import ExactEngine
from deepproblog.dataset import DataLoader
from deepproblog.train import train_model


print()
window_size = 50
window_stride = 10
test_split = 0.25
shuffle_dataset = True

# weird:batching = False for the Network but batch_size still passed in the DataLoader
batch_size = 1

#create the problog chache file, with initial throwaway probabilistic fact
with open('cached_predicates.pl', 'w') as f:
    f.write('0.0::cached(tensor(train(0)),interacting(p1,p2),0).')

# define network and make it DeepProbLog-compatible
rnn = LSTM_net(num_classes=4, input_size=5, hidden_size=32, num_layers=1)
rnn_DPL = Network(rnn, "lstm_net", batching=False)
rnn_DPL.optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)

# define DPL model given DPL program and neural net
# TODO: write activity_detection.pl file
model = Model("problog_files/activity_detection_minimal.pl", [rnn_DPL])
model.set_engine(ExactEngine(model))

# TODO: write what this is and why
model.add_tensor_source(
    "train",
    CaviarFrames(
        window_size=window_size,
        window_stride=window_stride,
        subset="train",
        test_split=test_split,
        shuffle=shuffle_dataset,
    ),
)
model.add_tensor_source(
    "test",
    CaviarFrames(
        window_size=window_size,
        window_stride=window_stride,
        subset="test",  # input sequence contains data for 2 people interacting
        test_split=test_split,
        shuffle=shuffle_dataset,
    ),
)

# define the dataset, wrap in a DPL data loader, and train the system
dataset = CAVIARDataset(
    window_size=50,
    window_stride=10,
    subset="train",
    test_split=0.25,
    shuffle=True,
)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
train_model(
    model,
    loader,
    2,  # number of epochs
    log_iter=100,
    profile=0,
)
