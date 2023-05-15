import cached_path
import dataclasses
import typing
import os
import numpy as np
import math
from sklearn.model_selection import train_test_split
import torch

from xml.etree import ElementTree


complex_event_mapping = {
    "joining": 1,
    "interacting": 2,
    "moving": 3,
    "split up": 4,
    "leaving object": 5,
    "fighting": 6,
    "no_event": 7,
    "leaving victim": 8,
}


@dataclasses.dataclass
class BoundingBox:
    orientation: int
    height: int
    width: int
    x_position: int
    y_position: int
    simple_event: str

    def get_features_from_bb(self):
        return list(
            map(
                int,
                [
                    self.height,
                    self.width,
                    self.x_position,
                    self.y_position,
                    self.orientation,
                ],
            )
        )

    @classmethod
    def from_dict(cls, dictionary):
        return cls(
            height=dictionary["h"],
            width=dictionary["w"],
            x_position=dictionary["xc"],
            y_position=dictionary["yc"],
            orientation=dictionary["orientation"],
            simple_event=dictionary["simple_event"],
        )


@dataclasses.dataclass
class Group:
    subgroups: dict[tuple[int, int], str]


@dataclasses.dataclass
class CaviarFrame:
    bounding_boxes: dict[int, BoundingBox]
    group: Group


def load_caviar_data(subset_filenames: typing.Optional[list[str]] = None):
    # @param: subset_filenames --> the filenames that we wish to
    # load.

    raw_sequences = []

    # Fetch the data from online and store them for future use.
    # Once the archive is extracted data is stored within a
    # caviar_videos subdirectory.

    # caviar_root = os.path.join(
    #     cached_path.cached_path(
    #         "https://users.iit.demokritos.gr/~nkatz/caviar_videos.zip",
    #         extract_archive=True,
    #     ),
    #     "caviar_videos",
    # )

    caviar_root = "/home/yuzer/.cache/cached_path/3d7268fd95461fe356087696890c33afe4a1257e48773d5e3cc6e06d1f505a55.4baaf2515ddb1b1533af48a43c660a60fa029edfc3562069cb4afcbcdb9081e8-extracted/caviar_videos"

    # Parse the different xml files. We are going to use the
    # bounding boxes of different persons to predict the
    # labels so store the boxes and the labels

    # Do the parsing and store the different video sequences
    all_filenames = [
        filename.removesuffix(".xml")
        for filename in os.listdir(caviar_root)
        if filename.endswith(".xml")
    ]
    for filename in subset_filenames or all_filenames:
        with open(os.path.join(caviar_root, f"{filename}.xml"), "r") as input_file:
            frames = ElementTree.parse(input_file).findall("frame")

        frame_objects = []
        for frame in frames:
            # All of the this code is directly based on the xml structure and
            # should never crash but the type checker obviously has no idea and
            # thinks all of this information might not exist.
            # TODO: Add all runtime checks but for now just ignore the lines
            from typing import no_type_check

            @no_type_check
            def parse_frame(frame_) -> CaviarFrame:
                # Parse the bounding boxes in the frame and add the to a dictionary
                bounding_boxes = {
                    int(object.attrib["id"]): BoundingBox.from_dict(
                        object.find("box").attrib
                        | {"orientation": int(object.find("orientation").text)}
                        | {
                            "simple_event": object.find("hypothesislist")
                            .find("hypothesis")
                            .find("movement")
                            .text
                        }
                    )
                    for object in frame_.find("objectlist").findall("object")
                }

                # Parse the groups in the frame if they exist and register the complex events
                groups = {}

                for group in frame_.find("grouplist").findall("group"):
                    groups[tuple(map(int, group.find("members").text.split(",")))] = (
                        group.find("hypothesislist")
                        .find("hypothesis")
                        .find("situation")
                        .text
                    )

                return CaviarFrame(bounding_boxes, Group(groups))

            frame_objects.append(parse_frame(frame))

        # Now somehow we need to pass this data though a sequence
        # absorbing model which means we need constant feature dimensionality
        # per frame. What we do is for each group that performs a complex event
        # we extract its complete history. So if in the video there are three seperate
        # groups then we will create three seperate sequences each of which holds the
        # complete history of the bounding boxes of two people

        # First get the unique groups (each group is 2 people)
        unique_groups = set()
        for frame in frame_objects:
            unique_groups.update(frame.group.subgroups.keys())

        # For each group get the complete history (i.e. the bounding boxes
        # of the two people since the appeared in the video) and at each time
        # associate the frame with a label being either no_event, i.e. no complex
        # event in the frame or some complex event identifier
        group_raw_streams = []
        for group_id in unique_groups:
            group_raw_stream = []

            # Go and extract the history of each group. In each frame if both objects from
            # the group exist keep their bounding boxes and retrieve the frame label for those
            # two people
            for frame in frame_objects:
                if all(
                    object_id in frame.bounding_boxes.keys() for object_id in group_id
                ):
                    group_raw_stream.append(
                        (
                            tuple(
                                frame.bounding_boxes[object_id]
                                for object_id in group_id
                            ),
                            frame.group.subgroups.get(group_id, "no_event"),
                        )
                    )

            group_raw_streams.append(group_raw_stream)

        # Add the raw streams from the filename to the total raw_sequences list
        raw_sequences.extend(group_raw_streams)

    # keep only sequences with 2 bounding boxes (where only 2 people are present)
    return list(
        filter(
            lambda sequence: all(len(bbs) == 2 for bbs, _ in sequence), raw_sequences
        )
    )


def get_features_from_frame(frame):
    (bounding_boxes, CE) = frame
    (bb1, bb2) = bounding_boxes
    return bb1.get_features_from_bb() + bb2.get_features_from_bb()


def get_complex_event_from_frame(frame):
    (bounding_boxes, CE) = frame

    # maybe do this with sklearn's Label Encoder, not sure if it invertible
    # though which may be needed later on for interfacing with DPL part
    return complex_event_mapping[CE]


def generate_io_from_dataset(raw_sequences, window_size, window_stride):
    """
    Generates feature_maps and labels given the raw sequences of the
    CAVIAR dataset

    :param raw_sequences: 30 sequences of frames (videos) from CAVIAR
    :param window_size: The number of frames indicating the length of the sequences
        used as a feature map (a window_size=50 is 2sec worth of video given 25frames/s)
    :param window_stride: The stride of the sliding window

    :returns:
        input_feature_maps: np.array of size (#examples, window_size, num_features)
            for window_size=50, window_stride=10 this is (763examples, 50frames/sequence, 10features/frame)
        complex_event_labels: np.array of size (#examples, window_size, 1)
            for window_size=50, window_stride=10 this is (763examples, 50frames/sequence, 1CE/frame)
    """
    sequences = []

    # generate sequences of length window_size with window_stride
    for raw_sequence in raw_sequences:
        sequences.extend(
            [
                raw_sequence[i * window_stride : (i * window_stride) + window_size]
                for i in range(math.ceil(len(raw_sequence) / window_stride))
            ]
        )

    # filter out sequences with are not equal in length to window_size
    # (remainders from the end of a sequence)
    sequences = filter(lambda sequence: len(sequence) == 50, sequences)

    input_feature_maps = []
    complex_event_labels = []

    # for each of the sequences of length window_size, go to each sequence
    # and transform the frame into a list of features (10 features - 5 for
    # each person) and a label for a complex event
    for sequence in sequences:
        for _ in range(window_size):
            input_feature_maps.append(
                [get_features_from_frame(frame) for frame in sequence]
            )
            complex_event_labels.append(
                [get_complex_event_from_frame(frame) for frame in sequence]
            )

    # save data in a file so that it won't have to be generated again in
    # the future if the settings (window_size, window_stride) are the same
    file_name = f"window_size({window_size})_window_stride({window_stride})"
    file_path = os.path.join("data", file_name)
    print(f"file '{file_path}' generated, saving now...")
    np.savez(
        file=file_path,
        input_feature_maps=np.array(input_feature_maps),
        complex_event_labels=np.array(complex_event_labels),
    )

    return np.array(input_feature_maps), np.array(complex_event_labels)


def get_caviar_data(window_size, window_stride):
    file_name = f"window_size({window_size})_window_stride({window_stride}).npz"
    file_path = os.path.join("data", file_name)

    if os.path.exists(file_path):
        print(f"file '{file_path}' exists, loading now...")
        dataset = np.load(file_path)
        return (
            dataset["input_feature_maps"],
            dataset["complex_event_labels"],
        )
    else:
        print(f"file '{file_path}' doesn't exist, generating now...")
        raw_dataset = load_caviar_data()
        return generate_io_from_dataset(
            raw_dataset,
            window_size,
            window_stride,
        )


def get_dataset_split(
    input_feature_maps, complex_event_labels, subset, test_split, shuffle
):
    """
    Get train or test split from the dataset
    """

    X_train, X_test, y_train, y_test = train_test_split(
        input_feature_maps,
        complex_event_labels,
        test_size=test_split,
        shuffle=shuffle,
    )

    match subset:
        case "train":
            return (
                torch.Tensor(X_train),
                torch.Tensor(y_train),
            )

        case "test":
            return (
                torch.Tensor(X_test),
                torch.Tensor(y_test),
            )

        case _:
            raise ValueError("Parameter 'subset' should be either 'train' or 'test'")


if __name__ == "__main__":
    # ["mwt1gt", "mwt2gt", "mws1gt", "ms3ggt"]
    # raw_sequences = load_caviar_data()
    # set([len(frame) for sequence in raw_sequences for frame in sequence])
    # print(raw_sequences)

    window_size = 50
    window_stride = 10

    raw_sequences = load_caviar_data()

    input_feature_maps, complex_event_labels = generate_io_from_dataset(
        raw_sequences, window_size, window_stride
    )

    print()
