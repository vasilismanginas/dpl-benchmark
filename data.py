import torch
from sklearn.model_selection import train_test_split
from typing import Mapping, Iterator
from problog.logic import Term
from deepproblog.dataset import Dataset, load
from caviar_utils import get_caviar_data, get_dataset_split, complex_event_mapping
from problog.logic import Term, Constant
from deepproblog.query import Query


class CaviarFrames(Mapping[Term, torch.Tensor]):
    def __init__(
        self, window_size, window_stride, subset, test_split=0.25, shuffle=True
    ):
        self.subset = subset
        self.dataset = CAVIARDataset(
            window_size,
            window_stride,
            subset,
            test_split,
            shuffle,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator:
        for i in range(self.dataset):
            yield self.dataset.feature_maps[i]

    def __getitem__(self, item):
        return self.dataset.feature_maps[int(item[0])]


class CAVIARDataset(Dataset):
    def __init__(
        self,
        window_size,
        window_stride,
        subset,
        test_split=0.25,
        shuffle=True,
    ):
        self.subset = subset
        self.window_size = window_size
        self.window_stride = window_stride

        # get input feature maps and CE labels from CAVIAR
        input_feature_maps, complex_event_labels = get_caviar_data(
            window_size,
            window_stride,
        )

        # get either the training or testing split
        self.feature_maps, self.CE_labels = get_dataset_split(
            input_feature_maps,
            complex_event_labels,
            subset,
            test_split,
            shuffle,
        )

        # self.print_CE_types(subset, complex_event_labels)

    def __len__(self):
        return len(self.CE_labels)

    def to_query(self, i):
        # this is a feature map including window_size=50 frames with 2 people
        feature_map = Term("tensor", Term(self.subset, Constant(i)))
        CE_labels = self.CE_labels[i]

        # we're currently doing the very dumb thing of having individual queries for
        # each timestep in the sequence (i.e. we evaluate the LSTM 50 times for the
        # same 50x10 tensor because we don't yet know how to perform 50 queries
        # within the same LSTM evaluation)
        query_timestep = i % self.window_size

        term = Term(
            "happens",
            feature_map,
            Constant(query_timestep),
        )

        term = Term(
            "holdsAt",
            feature_map,
            Term("interacting", Constant("p1"), Constant("p2")),
            Constant(query_timestep),
        )

        return Query(term)

    def print_CE_types(self, subset, complex_event_labels):
        all_CE_types_str = []
        all_CE_types_int = set(
            [CE for sequence in complex_event_labels for CE in sequence]
        )

        for CE in all_CE_types_int:
            all_CE_types_str.extend(
                [k for k, v in complex_event_mapping.items() if v == CE]
            )

        print(
            f"All complex event types in subset {subset}", "\n", all_CE_types_str, "\n"
        )


if __name__ == "__main__":
    dataset = CAVIARDataset(
        subset="train", window_size=50, window_stride=10, test_split=0.25
    )

    # dataset.to_query(1)

    print()
