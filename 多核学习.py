from multiprocessing import Process, Queue
import os, time, random


def writer(queue):
    print("Writing process with PID {}".format(os.getpid()))
    for i in range(5):
        print("Putting {} into the queue".format(i + 1))
        queue.put(i + 1)
        time.sleep(random.random())


def reader(queue):
    print("Reading process with PID {}".format(os.getpid()))
    while True:
        get_value = queue.get()
        print("Getting {} from the queue".format(get_value))


if __name__ == "__main__":
    queue = Queue()
    writer_process = Process(target=writer(queue))
    reader_process = Process(target=reader(queue))

    reader_process.start()
    writer_process.start()

    writer_process.join()
    reader_process.terminate()  # while True, 因此不能等到其结束，只能使用terminate