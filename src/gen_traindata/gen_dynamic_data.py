import random
from collections import deque

from src.datasample import DataSample2
from src.gen_traindata.tools import make_new_sample_variation
from src.tools import rand_fix_interval
from src.gesture import HANDS_POSITION


def is_a_hand_position_set(sample: DataSample2) -> bool:
    return sample.gestures[0].l_hand_position is not None or sample.gestures[0].r_hand_position is not None


def set_hand_position_if_not_set(sample: DataSample2) -> None:
    if sample.gestures[0].l_hand_position is None \
            and sample.gestures[0].r_wrist is not None:  # Ensure the left hand itself is set
        sample.set_sample_gestures_point_to("l_hand_position", [
            rand_fix_interval(1),
            rand_fix_interval(1),
            rand_fix_interval(1),
        ])
    if sample.gestures[0].r_hand_position is None \
            and sample.gestures[0].r_wrist is not None:  # Ensure the right hand itself is set
        sample.set_sample_gestures_point_to("r_hand_position", [
            rand_fix_interval(1),
            rand_fix_interval(1),
            rand_fix_interval(1),
        ])


def decompose_into_static_sample(sample: DataSample2, label: str) -> deque[DataSample2]:
    """Create len(sample.gestures) new samples
    containing each a different frame of sample repeated len(sample.gestures) times.
    Args:
        sample (DataSample2): Sample to decompose
        label (str): The label of the new sub samples

    Returns:
        deque[DataSample2]: New sub samples
    """
    new_sub_samples: deque[DataSample2] = deque()

    for frame in sample.gestures:
        new_sample: DataSample2 = DataSample2(label=label, gestures=[frame])
        while len(new_sample.gestures) < len(sample.gestures):
            new_sample.gestures.append(frame)
        new_sub_samples.append(new_sample)

    return new_sub_samples


def gen_dynamic_data(sample: DataSample2,
                     nb_frame: int,
                     null_set: str | None = None,
                     ) -> deque[DataSample2]:
    sub_samples: deque[DataSample2] = deque()
    hands_positions: list[str] = HANDS_POSITION.getActiveFields()

    tmp_sample: DataSample2

    for i in range(2):
        # Create variation that is translated and reframed
        tmp_sample = make_new_sample_variation(sample).reframe(
            random.randint(2, nb_frame)
        ).translate_sample(
            rand_fix_interval(1),
            rand_fix_interval(1),
            rand_fix_interval(1),
            hands_positions)
        set_hand_position_if_not_set(tmp_sample)
        k: int = 1
        while i == 1 and k < len(tmp_sample.gestures) - 1:
            # Create variation that is translated and reframed but with holes
            if random.randint(0, 10) == 0:
                tmp_sample.gestures[k].setAllPointsToZero()
            k += 1
        sub_samples.append(tmp_sample.noise_sample())

    # Create sample for each frame to make the model understand the movement
    if null_set is not None and null_set != sample.label:
        sub_samples.extend(
            decompose_into_static_sample(sample, sample.label))

    if null_set is not None and null_set != sample.label:
        # Create variations with with randomized filled frames but animation backward so its not correct
        if is_a_hand_position_set(sample):
            tmp_sample = make_new_sample_variation(sample).reframe(
                random.randint(2, nb_frame)
            ).translate_sample(
                rand_fix_interval(1),
                rand_fix_interval(1),
                rand_fix_interval(1),
                hands_positions)
            tmp_sample.gestures.reverse()
            sub_samples.append(tmp_sample.noise_sample())

    return sub_samples
