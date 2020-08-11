import numpy as np


def nano_baby_overlap(nano_event, baby_event):
    n_overlap = np.sum(np.in1d(nano_event, baby_event))
    n_nano = len(nano_event)
    n_baby = len(baby_event)

    print()
    print("Overlapping events:", n_overlap)
    print("Events only in nano:", n_nano - n_overlap)
    print("Events only in baby:", n_baby - n_overlap)
    print()

    _, nano_idx, baby_idx = np.intersect1d(nano_event, baby_event, return_indices=True)

    return nano_idx, baby_idx
