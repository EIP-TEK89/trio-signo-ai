import argparse
from dataclasses import dataclass, field
import time
import os

from src.gesture import ACTIVATED_GESTURES_PRESETS, ActiveGestures, ALL_GESTURES
import sys


@dataclass
class Args:
    data_augmentation: int = 1
    name: str = field(default_factory=lambda: f"trainset_{time.strftime('%d-%m-%Y_%H-%M-%S', time.localtime(time.time()))}")
    memory_frame: int = 15
    dataset_dir: str = "datasets"
    null_label: str | None = None
    balance_samples: bool = False
    one_sided: bool = False
    active_points: ActiveGestures = field(default_factory=lambda: ActiveGestures.buildWithPreset(ALL_GESTURES))
    sample_label: list[str] = field(default_factory=list)
    sign_detector: bool = False


def parse_args() -> Args:
    args = Args()

    a_param_description: str = ""
    for key, value in ACTIVATED_GESTURES_PRESETS.items():
        a_param_description += f"\n\t{key}: {value[1]}"

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=f"Creates a training dataset with the given labels and parameters."
    )
    parser.add_argument(
        '--data-augmentation', "-d",
        help='Number of subset to generate for each sample.',
        required=False,
        default=args.data_augmentation,
        type=int)
    parser.add_argument(
        '--name', "-n",
        help='Name of the trainset.',
        required=False,
        default=args.name,
        type=str)
    parser.add_argument(
        '--memory-frame', "-m",
        help='Number of frame in the past in the training set',
        required=False,
        default=args.memory_frame,
        type=int)
    parser.add_argument(
        '--dataset-dir',
        help='Folder the progam will look in to find the dataset to create the trainset.',
        required=False,
        default=args.dataset_dir,
        type=str)
    parser.add_argument(
        '--null-label', "-x",
        help='NULL dataset: Define the null labeled output for the model further training data.',
        required=False,
        default=args.null_label,
        type=str)
    parser.add_argument(
        '--balance-samples', "-b",
        help='Balance the number of samples between label in the training dataset.',
        required=False,
        action='store_true')
    parser.add_argument(
        '--one-sided', "-o",
        help='One sides all the sign making left and right hand the same',
        required=False,
        action='store_true')
    parser.add_argument(
        '--active-points', "-a",
        help=f"(e.g: only the right hand points can be set to active) (Default: all points are active):{a_param_description}",
        required=False,
        default=None,
        type=str,
        nargs='*')
    parser.add_argument(
        '--sample-label', "-s",
        help=f"List of dataset to use to generate the training dataset, the program will take the corresponding folder set in --dataset-dir.",
        required=False,
        default=args.sample_label,
        nargs='*',
        type=str)
    parser.add_argument(
        '--sign-detector',
        help='Creates a dataset optimized for sign detector model.',
        required=False,
        action='store_true')

    term_args: argparse.Namespace = parser.parse_args()

    assert type(term_args.data_augmentation) is int, "data-augmentation must be an integer"
    assert term_args.data_augmentation > 0, "data-augmentation must be greater than 0"
    args.data_augmentation = term_args.data_augmentation

    assert type(term_args.name) is str, "name must be a string"
    args.name = term_args.name

    assert type(term_args.memory_frame) is int, "memory-frame must be an integer"
    assert term_args.memory_frame > 0, "memory-frame must be greater than 0"
    args.memory_frame = term_args.memory_frame

    assert type(term_args.dataset_dir) is str, "dataset-dir must be a string"
    args.dataset_dir = term_args.dataset_dir

    assert term_args.null_label is None or type(term_args.null_label) is str, "null-label must be a string or None"
    args.null_label = term_args.null_label

    args.balance_samples = term_args.balance_samples
    args.one_sided = term_args.one_sided
    args.sign_detector = term_args.sign_detector
    if args.sign_detector:
        assert args.null_label is not None, "--null-label must be set when using --sign-detector"

    if term_args.active_points is not None:
        requested_active_gesture: list[ActiveGestures] = []
        for points in term_args.active_points:
            tmp: tuple[ActiveGestures, str] | None = ACTIVATED_GESTURES_PRESETS.get(points)
            if tmp is None:
                print(f"Unknown active point: {points}. Available points are: {', '.join(ACTIVATED_GESTURES_PRESETS.keys())}")
                sys.exit(1)
            requested_active_gesture.append(tmp[0])
        args.active_points = ActiveGestures.buildWithPreset(requested_active_gesture)

    assert type(term_args.sample_label) is list, "sample-label must be a list of strings"
    assert len(term_args.sample_label) > 0, "sample-label must contain at least one label"
    args.sample_label = term_args.sample_label
    if args.null_label is not None:
        args.sample_label.append(args.null_label)
    folders = os.listdir(args.dataset_dir)
    valid: bool = True
    for label in args.sample_label:
        if label not in folders:
            print(f"Label '{label}' not found in dataset directory '{args.dataset_dir}'.")
            valid = False
    if not valid:
        sys.exit(1)

    return args
