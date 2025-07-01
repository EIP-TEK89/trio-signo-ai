import os
import sys
import time
import copy
import random
from collections import deque

from src.gen_traindata.gen_static_data import gen_static_data
from src.gen_traindata.gen_dynamic_data import gen_dynamic_data
from src.gen_traindata.tools import rand_gesture

from src.gesture import DataGestures, ActiveGestures, ALL_GESTURES, ACTIVATED_GESTURES_PRESETS
from src.datasample import DataSample
from src.datasamples import IDX_VALID_SAMPLE, DataSamples, DataSamplesInfo, IDX_INVALID_SAMPLE

DATASETS_DIR = "datasets"


def print_help():
    a_param_description: str = ""
    for key, value in ACTIVATED_GESTURES_PRESETS.items():
        a_param_description += f"\n\t\t{key}: {value[1]}"

    print(f"""USAGE:
\t{sys.argv[0]} [OPTIONS] [DATASET1 DATASET2 ...]
OPTIONS:
\t-h: Display this help message
\t-s: (-s [integer]) Number of subset to generate for each sample
\t-n: (-n [string]) Name of the dataset
\t-f: (-f [integer]) Number of frame in the past in the training set
\t-x: (-s [string (label)]) NULL dataset: Define the null labeled output for the model further training data.
\t-b: (-b (enables)) Balance the number of element between label in the training dataset
\t-o: (-o (enables)) One sides all the sign making left and right hand the same
\t-a: (-a [string]) Active point: Let you define which point to activate in the training dataset
\t    (e.g: only the right hand points can be set to active) (Default: all points are active):{a_param_description}
\t[DATASET1 DATASET2 ...]: List of dataset to use to generate the training dataset, the program will take the corresponding folder in the \"datasets\" directory.
""")


def print_progression(dataset_labels: list[str], label_id: int,
                      treated_sample: int, label_total_samples: int,
                      subset: int, total_subset: int,
                      created_sample: int,
                      start_time: float, completed_cycle: int, total_cycle: int):
    elapsed_time = time.time() - start_time
    one_cycle_time = 1
    if completed_cycle != 0:
        one_cycle_time = elapsed_time / completed_cycle
    remaining_time = one_cycle_time * (total_cycle - completed_cycle)
    remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
    dataset_labels_len = len(dataset_labels)

    string: str = f"\r\033[KCreating dataset: " + \
          f"[Label ({dataset_labels[label_id]}): {str(label_id).zfill(
              len(str(dataset_labels_len)))}/{dataset_labels_len}] " + \
          f"[Datasample: {str(treated_sample).zfill(
              len(str(label_total_samples)))}/{label_total_samples}] " + \
          f"[Subset Generation: {str(subset).zfill(
              len(str(total_subset)))}/{total_subset}] " + \
          f"[Sample generated: {created_sample}] " + \
          f"Remain time: {remaining_time_str} {str(completed_cycle).zfill(len(str(total_cycle)))}/{total_cycle}"
    print(string, end="\r", flush=True)


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
        for _ in range(2):
            tmp_sample: DataSample = DataSample(null_set, [])

            target_nb_frame: int = random.randint(1, nb_frame)
            while len(tmp_sample.gestures) < target_nb_frame:
                if random.randint(0, 5) == 0:
                    tmp_sample.gestures.insert(-1, DataGestures())
                else:
                    tmp_sample.gestures.insert(-1, rand_gesture())
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


def load_datasamples(dataset_labels: list[str],
                     memory_frame: int,
                     null_label: str | None = None,
                     ) -> dict[str, tuple[list[DataSample], list[DataSample]]]:
    data_samples: dict[str, tuple[list[DataSample], list[DataSample]]] = {}
    for label_name in dataset_labels:
        label_path: str = f"{DATASETS_DIR}/{label_name}"
        dataset_samples = os.listdir(label_path)
        samples: list[DataSample] = []
        counter_examples: list[DataSample] = []

        if len(dataset_samples) == 0:
            print(f"Warning: {label_name} is empty")
            continue
        for dataset_sample in dataset_samples:
            try:
                sample: DataSample = DataSample.fromJsonFile(
                    f"{label_path}/{dataset_sample}")
                sample.label = label_name
                if len(sample.gestures) > memory_frame:
                    sample.reframe(memory_frame)
                samples.append(sample)
            except Exception as e:
                if null_label is not None and dataset_sample == "counter_example" \
                        and os.path.isdir(f"{label_path}/{dataset_sample}"):
                    file_names = os.listdir(f"{label_path}/{dataset_sample}")
                    for file_name in file_names:
                        try:
                            sample = DataSample.fromJsonFile(
                                f"{label_path}/{dataset_sample}/{file_name}")
                            sample.invalid = True
                            if len(sample.gestures) > memory_frame:
                                sample.reframe(memory_frame)
                            counter_examples.append(sample)
                        except Exception as e:
                            print(f"Error: {
                                  dataset_sample}/{file_name} is not a valid json file. {e}")
                else:
                    print(f"Error: {
                          dataset_sample} is not a valid json file. {e}")
        data_samples[label_name] = (samples, counter_examples)
    return data_samples


def main():
    i = 1
    dataset_labels: list[str] = []
    total_subsets: int = 1
    dataset_name: str | None = None
    nb_frame = 15
    null_set: str | None = None
    active_gesture: ActiveGestures | None = None
    requested_active_gesture: list[ActiveGestures] = []
    balance: bool = False
    one_side: bool = False
    while i < len(sys.argv):
        args = sys.argv[i]
        # print(args)
        if args.startswith("-"):
            match args[1]:
                case "h":
                    print_help()
                    exit()
                case "s":
                    i += 1
                    total_subsets = int(sys.argv[i])
                case "n":
                    i += 1
                    dataset_name = sys.argv[i]
                case "f":
                    i += 1
                    nb_frame = int(sys.argv[i])
                case "x":
                    i += 1
                    null_set = sys.argv[i]
                    dataset_labels.append(null_set)
                case "a":
                    i += 1
                    tmp: tuple[ActiveGestures, str] | None = ACTIVATED_GESTURES_PRESETS.get(
                        sys.argv[i])
                    if tmp is None:
                        print("Invalid active gesture preset")
                        exit(1)
                    requested_active_gesture.append(tmp[0])
                case "b":
                    balance = True
                case "o":
                    one_side = True
                case _:
                    print(f"Invalid argument: {args}")
                    exit(1)
        else:
            dataset_labels.append(args)
        i += 1

    if len(requested_active_gesture):
        active_gesture = ActiveGestures.buildWithPreset(
            requested_active_gesture)
    else:
        active_gesture = ALL_GESTURES

    folders = os.listdir(DATASETS_DIR)

    valid = True
    for dataset in dataset_labels:
        if dataset not in folders:
            print(f"Dataset\"{dataset}\" not found in {DATASETS_DIR}")
            valid = False
    if not valid:
        exit(1)

    if dataset_name is None:
        timestamp = time.time()
        # Convert the timestamp to local time (struct_time object)
        local_time = time.localtime(timestamp)
        # Format the local time as a string
        formatted_date = time.strftime("%d-%m-%Y_%H-%M-%S", local_time)
        dataset_name = f"trainset_{formatted_date}"

    summary_checker(dataset_name, null_set, dataset_labels,
                    total_subsets, nb_frame, dataset_name, one_side, active_gesture)

    print("Loading samples into memory...", end=" ")
    data_samples: dict[str, tuple[list[DataSample], list[DataSample]]] = load_datasamples(
        dataset_labels, memory_frame=nb_frame, null_label=null_set)
    print("[DONE]")

    train_data: DataSamples = DataSamples(DataSamplesInfo(
        dataset_labels, nb_frame, active_gesture, one_side=one_side))
    if null_set is not None:
        train_data.info.null_sample_id = train_data.info.label_map[null_set]
    print(train_data.info.label_map)
    total_cycle: int = sum([len(samples[IDX_VALID_SAMPLE]) + len(samples[IDX_INVALID_SAMPLE])
                           for samples in data_samples.values()]) * total_subsets
    completed_cycle = 0

    subset: int = 0
    start_time = time.time()
    label_id: int = 0
    label_total_samples: int = 0
    treated_sample: int = 0
    for label, samples in data_samples.items():

        treated_sample = 0
        label_id = train_data.info.label_map[label]
        label_total_samples = len(
            samples[IDX_VALID_SAMPLE]) + len(samples[IDX_INVALID_SAMPLE])

        print_progression(train_data.info.labels, label_id, treated_sample,
                          label_total_samples, subset, total_subsets,
                          train_data.sample_count, start_time, completed_cycle,
                          total_cycle)

        for sample in samples[IDX_VALID_SAMPLE]:
            sample.label = label
            train_data.addDataSample(sample)

            subset = 0
            while subset < total_subsets:

                print_progression(train_data.info.labels, label_id, treated_sample,
                                  label_total_samples, subset, total_subsets,
                                  train_data.sample_count, start_time, completed_cycle,
                                  total_cycle)

                train_data.addDataSamples(
                    create_subset(sample, nb_frame, null_set, active_gesture))
                completed_cycle += 1
                subset += 1

            treated_sample += 1

            print_progression(train_data.info.labels, label_id, treated_sample,
                              label_total_samples, subset, total_subsets,
                              train_data.sample_count, start_time, completed_cycle,
                              total_cycle)

        for sample in samples[IDX_INVALID_SAMPLE]:
            sample.label = label
            train_data.addDataSample(sample)
            subset = 0
            while subset < total_subsets:

                print_progression(train_data.info.labels, label_id, treated_sample,
                                  label_total_samples, subset, total_subsets,
                                  train_data.sample_count, start_time, completed_cycle,
                                  total_cycle)

                train_data.addDataSamples(
                    create_subset(sample, nb_frame, None, active_gesture), False)
                completed_cycle += 1
                subset += 1

            print_progression(train_data.info.labels, label_id, treated_sample,
                              label_total_samples, subset, total_subsets,
                              train_data.sample_count, start_time, completed_cycle,
                              total_cycle)

        print_progression(train_data.info.labels, label_id, treated_sample,
                          label_total_samples, subset, total_subsets,
                          train_data.sample_count, start_time, completed_cycle,
                          total_cycle)

    print_progression(train_data.info.labels, label_id, treated_sample,
                      label_total_samples, subset, total_subsets,
                      train_data.sample_count, start_time, completed_cycle,
                      total_cycle)

    print()
    if balance:
        print("Base generation duration: ", time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - start_time)))
        print("Balancing dataset...")
        biggest_label_count: int = max(
            [train_data.getNumberOfSamplesOfLabel(label_id) for label_id in train_data.info.label_map.values()])
        start_time2 = time.time()
        print("Biggest label count: ", biggest_label_count)

        label_id = 0
        completed_cycle: int = 0
        total_cycle = (biggest_label_count * len(train_data.info.labels)
                       ) - train_data.getNumberOfSamples()
        while label_id < len(train_data.samples):
            current_data_samples: list[DataSample] = data_samples[train_data.info.labels[label_id]][IDX_VALID_SAMPLE]

            if len(current_data_samples) == 0:
                print(f"Warning: {train_data.info.labels[label_id]} is empty")
                continue

            data_sample_len = len(current_data_samples)
            sample_idx: int = 0
            while train_data.getNumberOfSamplesOfLabel(label_id) < biggest_label_count:
                sample: DataSample = current_data_samples[sample_idx]
                sample.label = train_data.info.labels[label_id]
                generated_subset: list[DataSample] = create_subset(
                    sample, nb_frame, None, active_gesture)
                completed_cycle += len(generated_subset)
                train_data.addDataSamples(generated_subset)
                sample_idx = (sample_idx + 1) % data_sample_len
                print_progression(dataset_labels, label_id, sample_idx, data_sample_len,
                                  len(train_data.samples[label_id]
                                      ), biggest_label_count, train_data.sample_count,
                                  start_time2, completed_cycle, total_cycle)
            label_id += 1

        print("Balance generation duration: ", time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - start_time2)))

    train_data.getNumberOfSamples()
    print()
    print("Generation duration: ", time.strftime(
        "%H:%M:%S", time.gmtime(time.time() - start_time)))
    print("Total unique sample created: ", train_data.getNumberOfSamples())
    print("Saving dataset...")
    train_data.toCborFile(f"./{dataset_name}.cbor")
    # train_data.toJsonFile(f"./{dataset_name}.json", indent=4)

# import cProfile


if __name__ == "__main__":
    # cProfile.run("main()", sort="cumtime")
    main()
