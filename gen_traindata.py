import os
import sys
import time
import copy
import random
from collections import deque

from src.gen_traindata.gen_static_data import gen_static_data
from src.gen_traindata.gen_dynamic_data import gen_dynamic_data
from src.gen_traindata.tools import rand_gesture
from src.gen_traindata.parse_args import parse_args, Args

from src.gesture import DataGestures, ActiveGestures, ALL_GESTURES, ACTIVATED_GESTURES_PRESETS
from src.datasample import DataSample
from src.datasamples import IDX_VALID_SAMPLE, DataSamples, DataSamplesInfo, IDX_INVALID_SAMPLE

VALID_LABEL: str = "_valid"

class ProgressionLogger:
    dataset_labels: list[str] = []
    label_id: int = 0

    treated_label_samples: int = 0
    to_treat_label_samples: int = 0

    data_augmentation_iteration: int = 0
    total_data_augmentation_iterations: int = 0

    created_samples: int = 0
    start_time: float = 0.0

    completed_cycle: int = 0
    total_cycle: int = 0

    def setStartTime(self):
        self.start_time = time.time()

    def getElapsedTime(self) -> float:
        return time.time() - self.start_time

    def __repr__(self):
        elapsed_time = self.getElapsedTime()
        one_cycle_time = 1
        if self.completed_cycle != 0:
            one_cycle_time = elapsed_time / self.completed_cycle
        remaining_time = one_cycle_time * (self.total_cycle - self.completed_cycle)
        remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
        dataset_labels_len = len(self.dataset_labels)

        string: str = \
          f"[Label ({self.dataset_labels[self.label_id]}): {str(self.label_id).zfill(
              len(str(dataset_labels_len)))}/{dataset_labels_len}] " + \
          f"[Sample: {str(self.treated_label_samples).zfill(
              len(str(self.to_treat_label_samples)))}/{self.to_treat_label_samples}] " + \
          f"[Subset: {str(self.data_augmentation_iteration).zfill(
              len(str(self.total_data_augmentation_iterations)))}/{self.total_data_augmentation_iterations}] " + \
          f"[Sample created: {self.created_samples}] " + \
          f"Time left: {remaining_time_str} {str(self.completed_cycle).zfill(
              len(str(self.total_cycle)))}/{self.total_cycle}"

        return string

    def print(self):
        print(f"\r\033[K{self.__repr__()}", end="\r", flush=True)

def getFormatedTime(seconds: float) -> str:
    """Format seconds into a string of the form HH:MM:SS."""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def create_subset(sample: DataSample,
                  nb_frame: int,
                  null_set: str | None = None,
                  active_points: ActiveGestures = ALL_GESTURES,
                  ) -> list[DataSample]:
    sub_sample: deque[DataSample] = deque()

    initial_samples: list[DataSample] = [sample]
    if sample.mirrorable:
        mirror_sample: DataSample = copy.deepcopy(sample)
        mirror_sample.mirror_sample(x=True, y=False, z=False)
        initial_samples.append(mirror_sample)

    for samp in initial_samples:
        # Be careful those function randomize undefined (set to None) points
        tmp_samples: deque[DataSample]
        if len(sample.gestures) == 1:
            tmp_samples = gen_static_data(samp, nb_frame, null_set)
        else:
            tmp_samples = gen_dynamic_data(samp, nb_frame, null_set)
        sub_sample.extend(tmp_samples)

    # # Randomize all point that are not defined
    # for samp in sub_sample:
    #     samp.setNonePointsRandomlyToRandomOrZero()

    # Create pure non valid data
    if null_set is not None:
        tmp_sample: DataSample
        for _ in range(2):
            tmp_sample = DataSample(null_set, [])

            target_nb_frame: int = random.randint(1, nb_frame)
            while len(tmp_sample.gestures) < target_nb_frame:
                if random.randint(0, 5) == 0:
                    tmp_sample.gestures.insert(-1, DataGestures())
                else:
                    tmp_sample.gestures.insert(-1, rand_gesture())
            sub_sample.append(tmp_sample)

        tmp_sample = DataSample(null_set, [])
        target_nb_frame: int = random.randint(1, nb_frame)
        while len(tmp_sample.gestures) < target_nb_frame:
            tmp_sample.gestures.insert(-1, DataGestures().setAllPointsToZero())
        sub_sample.append(tmp_sample)

    return list(sub_sample)


def summary_checker(dataset_name: str, null_label: str | None, labels: list[str], total_subsets: int, nb_frame: int, file_name: str, one_side: bool, active_gesture: ActiveGestures = ALL_GESTURES):
    print(f"Dataset name: {dataset_name}")
    print(f"Null label: {null_label}")
    print(f"Labels: {labels}")
    print(f"Total subsets: {total_subsets}")
    print(f"Number of frame: {nb_frame}")
    print(f"Active gesture: {active_gesture}")
    print(f"Output file: {file_name}")
    print(f"One side: {one_side}")
    answer = None
    while answer != "y":
        answer = input("Do you want to continue? (y/n): ")
        if answer == "n":
            exit(0)


def load_datasamples(args: Args) -> dict[str, tuple[list[DataSample], list[DataSample]]]:
    data_samples: dict[str, tuple[list[DataSample], list[DataSample]]] = {}
    for label_name in args.sample_label:
        label_path: str = os.path.join(args.dataset_dir, label_name)
        samples: list[DataSample] = []
        counter_examples: list[DataSample] = []

        for label_kind in ["valid", "counter_examples"]:
            # print(os.listdir(label_path))
            try:
                label_kind_path: str = os.path.join(label_path, label_kind)
                print(f"Loading samples from {label_kind_path}")
                samples_kind = os.listdir(label_kind_path)
                if len(samples_kind) == 0:
                    print(f"Warning: {label_kind} folder is empty in {label_path}")
                    continue
                for sample in samples_kind:
                    sample_path = os.path.join(label_kind_path, sample)
                    try:
                        sample_data: DataSample = DataSample.fromJsonFile(
                            sample_path)
                        sample_data.label = label_name
                        if len(sample_data.gestures) > args.memory_frame:
                            sample_data.reframe(args.memory_frame)
                        if label_kind == "valid":
                            sample_data.invalid = False
                            samples.append(sample_data)
                        else:
                            sample_data.invalid = True
                            counter_examples.append(sample_data)
                    except Exception as e:
                        print(f"Error: {sample} in {label_kind}/ folder is not a valid json file. {e}")
            except Exception as e:
                print(f"Error: Failed to handle {label_kind}/ folder {label_path}: {e}")
                continue

        data_samples[label_name] = (samples, counter_examples)
    return data_samples


def main():

    args: Args = parse_args()
    summary_checker(args.name, args.null_label, args.sample_label,
                    args.data_augmentation, args.memory_frame, args.name, args.one_sided, args.active_points)

    print("Loading samples into memory...", end=" ")
    # dict[Label, tuple[list[valid samples], list[counter examples]]]
    data_samples: dict[str, tuple[list[DataSample], list[DataSample]]] = load_datasamples(args)
    print("[DONE]")

    final_labels: list[str] = args.sample_label
    _tmp_info = DataSamplesInfo(
        final_labels, args.memory_frame, args.active_points, one_side=args.one_sided)
    if args.sign_detector:
        final_labels = [args.null_label, VALID_LABEL]
    train_data: DataSamples = DataSamples(DataSamplesInfo(
        final_labels, args.memory_frame, args.active_points, one_side=args.one_sided))
    if args.null_label is not None:
        train_data.info.null_sample_id = train_data.info.label_map[args.null_label]

    progress_log: ProgressionLogger = ProgressionLogger()

    progress_log.total_cycle = sum([len(samples[IDX_VALID_SAMPLE]) + len(samples[IDX_INVALID_SAMPLE])
                           for samples in data_samples.values()]) * args.data_augmentation
    progress_log.total_data_augmentation_iterations = args.data_augmentation
    progress_log.dataset_labels = args.sample_label

    progress_log.setStartTime()
    initial_start_time: float = progress_log.start_time
    for label, samples in data_samples.items():

        progress_log.treated_label_samples = 0
        progress_log.label_id = _tmp_info.label_map[label]
        progress_log.to_treat_label_samples = len(
            samples[IDX_VALID_SAMPLE]) + len(samples[IDX_INVALID_SAMPLE])

        progress_log.print()

        for sample in samples[IDX_VALID_SAMPLE]:
            sample.label = label if not args.sign_detector else VALID_LABEL
            train_data.addDataSample(sample)

            progress_log.data_augmentation_iteration = 0
            while progress_log.data_augmentation_iteration < args.data_augmentation:

                progress_log.print()

                train_data.addDataSamples(
                    create_subset(sample, args.memory_frame, args.null_label, args.active_points))
                progress_log.completed_cycle += 1
                progress_log.data_augmentation_iteration += 1

            progress_log.treated_label_samples += 1
            progress_log.print()

        for sample in samples[IDX_INVALID_SAMPLE]:
            sample.label = label if not args.sign_detector else VALID_LABEL
            train_data.addDataSample(sample)
            progress_log.data_augmentation_iteration = 0
            while progress_log.data_augmentation_iteration < args.data_augmentation:

                progress_log.print()

                train_data.addDataSamples(
                    create_subset(sample, args.memory_frame, None, args.active_points), False)
                progress_log.completed_cycle += 1
                progress_log.data_augmentation_iteration += 1

            progress_log.treated_label_samples += 1
            progress_log.print()

        progress_log.print()

    progress_log.print()

    print()
    if args.balance_samples:
        def pick_samples(label: str, data_samples: dict[str, tuple[list[DataSample], list[DataSample]]]) -> list[DataSample]:
            if args.sign_detector and label == VALID_LABEL:
                samples: list[DataSample] = []
                for key_label, sample_pair in data_samples.items():
                    if key_label != args.null_label:
                        samples.extend(sample_pair[IDX_VALID_SAMPLE])
                return samples
            return data_samples[label][IDX_VALID_SAMPLE]

        print("Base generation duration: ", getFormatedTime(progress_log.getElapsedTime()))
        progress_log.setStartTime()
        print("Balancing dataset...")
        biggest_label_count: int = max(
            [train_data.getNumberOfSamplesOfLabel(label_id) for label_id in train_data.info.label_map.values()])
        # print("Biggest label count: ", biggest_label_count)

        label_id = 0
        progress_log.completed_cycle = 0
        progress_log.total_cycle = (biggest_label_count * len(train_data.info.labels)
                       ) - train_data.getNumberOfSamples()
        while label_id < len(train_data.samples):
            # current_data_samples: list[DataSample] = data_samples[train_data.info.labels[label_id]][IDX_VALID_SAMPLE]
            current_data_samples: list[DataSample] = pick_samples(train_data.info.labels[label_id], data_samples)

            if len(current_data_samples) == 0:
                print(f"Warning: {train_data.info.labels[label_id]} is empty")
                continue

            data_sample_len = len(current_data_samples)
            sample_idx: int = 0
            while train_data.getNumberOfSamplesOfLabel(label_id) < biggest_label_count:
                sample: DataSample = current_data_samples[sample_idx]
                sample.label = train_data.info.labels[label_id]
                generated_subset: list[DataSample] = create_subset(
                    sample, args.memory_frame, None, args.active_points)
                train_data.addDataSamples(generated_subset)

                progress_log.completed_cycle += len(generated_subset)
                sample_idx = (sample_idx + 1) % data_sample_len
                progress_log.label_id = label_id
                progress_log.print()

            label_id += 1

        print("Balance generation duration: ", getFormatedTime(progress_log.getElapsedTime()))

    train_data.getNumberOfSamples()
    print()
    print("Generation duration: ", getFormatedTime(time.time() - initial_start_time))
    print("Total unique sample created: ", train_data.getNumberOfSamples())
    print("Saving dataset...")
    train_data.toCborFile(f"./{args.name}.cbor")
    # train_data.toJsonFile(f"./{dataset_name}.json", indent=4)

# import cProfile


if __name__ == "__main__":
    # cProfile.run("main()", sort="cumtime")
    main()
