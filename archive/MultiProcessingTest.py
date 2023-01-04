import multiprocessing
import threading

list1 = []
list2 = []


def test():
    global list1, list2

    num_parallels = 1000000

    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    for i in range(num_parallels):
        pool.apply_async(async_func, callback=callback_func)

    pool.close()
    pool.join()
    if list1 == list2:
        print("list1 == list2")
    else:
        print("list1 != list2")
    print(list1)
    print(list2)
    pass


def async_func():
    curr_thread = threading.currentThread()
    return curr_thread.ident


def callback_func(result):
    global list1, list2
    list1.append(result)
    list2.append(result)


if __name__ == "__main__":
    test()
