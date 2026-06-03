import time


def track_time(func):
    """
    Decorator Pattern:
    Measures training execution time.
    """

    def wrapper(*args, **kwargs):

        start = time.time()

        result = func(*args, **kwargs)

        end = time.time()

        print(f"{func.__name__} executed in {end - start:.2f} seconds")

        return result

    return wrapper