import argparse
import time
import sys

def try_catch_sleep_example(t=2):
    """
    simple example to test try, catch, sys logs, and communicate function
    :param time: seconds to sleep
    :return: exit code, finished sleeping?
    """

    try:
        time.sleep(t)
        assert(1 == 2)
        print('successfully finished')
        return 0, "successfully finished", 2, 3, 4, 5
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print('failed to finished')
        raise AssertionError("1 != 2")
        return 1, "did not finish sleeping", exc_type, exc_value, exc_traceback


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the arguments for the Model')
    parser.add_argument("-t", "--time", type=int)
    args = vars(parser.parse_args())
    args["time"] = 2
    try_catch_sleep_example(args["time"])
