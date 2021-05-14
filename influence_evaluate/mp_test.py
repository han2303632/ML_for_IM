import time
from multiprocess import Queue
from multiprocess import current_process, Process, freeze_support
import snap

def func(idx):
    time.sleep(5)
    return 1

def worker(input, output):
    while not input.empty():
        func, args = input.get()
        output.put(func(args))


if __name__ == "__main__":
    '''
    pool = mp.Pool(processes = 4)
    def func1(i):
        time.sleep(20)
        return 1

    result = []

    for i in range(100):
        result.append(pool.apply_async(func1, (1)))

        print(len(result))

    pool.close()
    pool.join()
    '''
    freeze_support()
    NUMBER_OF_PROCESSES = 4
    TASK1 = [(func,i) for i in range(10)]

    task_queue = Queue()
    done_queue = Queue()

    list(map(task_queue.put, TASK1))
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=worker, args=(task_queue, done_queue)).start()

    print("submit complete")

    for i in range(len(TASK1)):
        print(done_queue.get())


