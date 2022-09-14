import concurrent.futures

def foo(bar, second):
    print('hello {} {}'.format(bar, second))
    for i in range(100):
        print('proc: {} value {}'.format(bar, i))

    return 'foo' + str(bar)

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(foo, 1, "something")
        future2 = executor.submit(foo, 2, "else")
        print("both started")
        return_value1 = future1.result()
        return_value2 = future2.result()
        print("both finished")
        print(return_value1, return_value2)

