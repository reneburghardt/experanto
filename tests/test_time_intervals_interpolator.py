import numpy as np
import pytest

from experanto.interpolators import Interpolator, TimeIntervalInterpolator

from .create_time_intervals_data import create_time_intervals_data


def assert_true_and_false(signal, true_indices, label_idx):
    mask = np.zeros(signal.shape[0], dtype=bool)
    for start, end in true_indices:
        mask[start:end] = True
        assert signal[
            start:end, label_idx
        ].all(), f"Expected True in {start}:{end} for label {label_idx}"
    assert (
        ~signal[~mask, label_idx]
    ).all(), f"Expected False outside {true_indices} for label {label_idx}"


def test_time_interval_interpolation():
    with create_time_intervals_data() as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        test_indices = [[0, 200]]
        assert_true_and_false(signal, test_indices, 0)
        train_indices = [[200, 400], [600, 800]]
        assert_true_and_false(signal, train_indices, 1)
        validation_indices = [[400, 600], [800, 1000]]
        assert_true_and_false(signal, validation_indices, 2)


def test_time_interval_interpolation_overlap2():
    with create_time_intervals_data(type="overlap2") as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        test_indices = [[0, 250]]
        assert_true_and_false(signal, test_indices, 0)
        train_indices = [[200, 400], [600, 800]]
        assert_true_and_false(signal, train_indices, 1)
        validation_indices = [[400, 600], [800, 1000]]
        assert_true_and_false(signal, validation_indices, 2)


def test_time_interval_interpolation_overlap3():
    with create_time_intervals_data(type="overlap3") as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        test_indices = [[0, 250]]
        assert_true_and_false(signal, test_indices, 0)
        train_indices = [[178, 403], [557, 823]]
        assert_true_and_false(signal, train_indices, 1)
        validation_indices = [[375, 601], [789, 1000]]
        assert_true_and_false(signal, validation_indices, 2)


def test_time_interval_interpolation_gap():
    with create_time_intervals_data(type="gap") as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        test_indices = [[0, 180]]
        assert_true_and_false(signal, test_indices, 0)
        train_indices = [[200, 400], [632, 800]]
        assert_true_and_false(signal, train_indices, 1)
        validation_indices = [[420, 600], [827, 1000]]
        assert_true_and_false(signal, validation_indices, 2)


def test_time_interval_interpolation_gap_and_overlap():
    with create_time_intervals_data(type="gap_and_overlap") as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        test_indices = [[0, 250]]
        assert_true_and_false(signal, test_indices, 0)
        train_indices = [[200, 390], [600, 800]]
        assert_true_and_false(signal, train_indices, 1)
        validation_indices = [[400, 600], [800, 1000]]
        assert_true_and_false(signal, validation_indices, 2)


def test_time_interval_interpolation_nans():
    with create_time_intervals_data(type="nans") as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        test_indices = [[0, 100]]
        assert_true_and_false(signal, test_indices, 0)
        train_indices = [[100, 250], [450, 650]]
        assert_true_and_false(signal, train_indices, 1)
        validation_indices = [[250, 450], [650, 850]]
        assert_true_and_false(signal, validation_indices, 2)


# New test for zero-length interval (start == end)
def test_time_interval_interpolation_zero_length():
    with create_time_intervals_data(type="zero_length") as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        # test label has a zero-length interval at 100, so should be all False
        assert_true_and_false(signal, [], 0)
        # train and validation as usual
        train_indices = [[200, 400], [600, 800]]
        assert_true_and_false(signal, train_indices, 1)
        validation_indices = [[400, 600], [800, 1000]]
        assert_true_and_false(signal, validation_indices, 2)


def test_time_interval_interpolation_multi_zero_length():
    with create_time_intervals_data(type="multi_zero_length") as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        test_indices = [[500, 600]]
        assert_true_and_false(signal, test_indices, 0)
        assert_true_and_false(signal, [], 1)
        validation_indices = [[800, 1000]]
        assert_true_and_false(signal, validation_indices, 2)


def test_time_interval_interpolation_full_range():
    with create_time_intervals_data(type="full_range") as timestamps:
        interp_obj = Interpolator.create("tests/time_interval_data")
        assert isinstance(interp_obj, TimeIntervalInterpolator)

        signal = interp_obj.interpolate(timestamps)

        # test label covers the entire range
        full_range = [[0, 1000]]
        assert_true_and_false(signal, full_range, 0)
        # train and validation as usual
        train_indices = [[200, 400], [600, 800]]
        validation_indices = [[400, 600], [800, 1000]]
        assert_true_and_false(signal, train_indices, 1)
        assert_true_and_false(signal, validation_indices, 2)
