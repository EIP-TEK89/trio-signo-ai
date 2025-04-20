import random
from collections import deque

from src.datasample import DataSample2
from src.gen_traindata.tools import make_new_sample_variation, zero_gesture, rand_gesture


def gen_static_data(sample: DataSample2,
                    nb_frame: int,
                    null_set: str | None = None,
                    ) -> deque[DataSample2]:
    sub_sample: deque[DataSample2] = deque()

    tmp_sample: DataSample2
    iterations: int = nb_frame * 1

    for _ in range(1):
        # Create a single framed variation
        sub_sample.append(make_new_sample_variation(sample).noise_sample())

        if nb_frame > 1:

            # Create variations with randomized length
            tmp_sample = make_new_sample_variation(sample)
            target_nb_frame: int = random.randint(2, nb_frame)
            while len(tmp_sample.gestures) < target_nb_frame:
                tmp_sample.gestures.append(tmp_sample.gestures[0])
            sub_sample.append(tmp_sample.noise_sample())

            # Create variations with with randomized filled frames
            # The idea is to make the model understand that for static gestures, only the first frame is important.
            tmp_sample = make_new_sample_variation(sample)
            target_nb_frame = random.randint(2, nb_frame)
            while len(tmp_sample.gestures) < target_nb_frame:
                val: int = random.randint(0, nb_frame // 2)
                # Since these sample need to be valid, we don't insert at position 0 as we should be.
                # This way we can ensure that the first frame is always the sign.

                if val == 0:
                    tmp_sample.gestures.insert(-1, zero_gesture())
                elif val == 1:
                    tmp_sample.gestures.insert(-1, rand_gesture())
                else:
                    tmp_sample.gestures.append(tmp_sample.gestures[0])

            sub_sample.append(tmp_sample.noise_sample())

        tmp_sample = make_new_sample_variation(sample)
        while len(tmp_sample.gestures) < nb_frame:
            tmp_sample.gestures.append(tmp_sample.gestures[0])
        sub_sample.append(tmp_sample.noise_sample())

        # # If null_set is set.
        # # Creates variations where the first frame is not the sign
        # if null_set is not None:
        #     tmp_sample = make_new_sample_variation(sample)
        #     tmp_sample.label = null_set

        #     target_nb_frame: int = random.randint(1, nb_frame - 1)
        #     while len(tmp_sample.gestures) < target_nb_frame:
        #         if random.randint(0, 5) == 0:
        #             tmp_sample.gestures.insert(0, zero_gesture())
        #         else:
        #             tmp_sample.gestures.insert(0, rand_gesture())

        #     # Removing extra frame
        #     while len(tmp_sample.gestures) > nb_frame:
        #         tmp_sample.gestures.pop(-1)
        #     sub_sample.append(tmp_sample.noise_sample())

        # # Generate coherent image succession for each sub_sample
        # src_sample: DataSample2 = make_new_sample_variation(sample)
        # tmp_sample = DataSample2(sample.label, [])
        # step: int = 0

        # while step < nb_frame * 1.5:
        #     # Copying the first frame and noise it, then add it in succession to the sample
        #     tmp_sample.gestures.insert(0, copy.deepcopy(src_sample.gestures[0]).noise())

        #     # Removing extra frame
        #     while len(tmp_sample.gestures) > nb_frame:
        #         tmp_sample.gestures.pop(-1)

        #     # Do not add noise to tmp_sample here. We already noise each DataGesture individually.
        #     # The reason we noise to the DataGesture is to make the noise coherent between frames.
        #     sub_sample.append(tmp_sample)
        #     step += 1

    # for sample in sub_sample:
    #     sample.set_sample_gestures_point_to("r_hand_position", [0, 0, 0])
    #     sample.set_sample_gestures_point_to("l_hand_position", [0, 0, 0])

    return sub_sample
