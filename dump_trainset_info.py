import sys
from src.datasamples import DataSamplesTensors, IDX_VALID_SAMPLE, IDX_INVALID_SAMPLE

info, samples = DataSamplesTensors.dumpInfo(sys.argv[1])

print(f"Labels ({len(info.labels)}):")
sample_count: int = 0
null_sample_count: int = 0
for key, val in info.label_map.items():
    print(f"\tLabel: \"{key}\" ID: {val} Sample count: {
          sum(samples[val])} {samples[val]}")
    sample_count += sum(samples[val])
    if info.null_sample_id is not None:
        if val == info.null_sample_id:
            null_sample_count += samples[val][IDX_VALID_SAMPLE]
        else:
            null_sample_count += samples[val][IDX_INVALID_SAMPLE]
print(f"Null sample count (include counter examples): {null_sample_count}")
print(f"Total number of samples: {sample_count}\n")

print(f"Active gestures:")
for field in info.active_gestures.getActiveFields():
    print(f"\t{field}")
