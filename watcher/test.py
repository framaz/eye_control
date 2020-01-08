if __name__ == "__main__":
    import calibrator
    timed_list = [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [2, 2, 2, 2]
    ]
    calibrator.smooth_n_cut(timed_list, 2)