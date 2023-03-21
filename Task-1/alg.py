import argparse

def multiplicate(A: list[int]) -> list[int]:
    n = len(A)
    num_of_zeros = 0
    for elem in A:
        if elem == 0:
            num_of_zeros += 1
    if num_of_zeros > 1:
        return [0] * n
    elif num_of_zeros == 1:
        prod = 1
        for i, elem in enumerate(A):
            if elem != 0:
                prod *= elem
                A[i] = 0
            else:
                k = i
        A[k] = prod
        return A
    else:
        prod = 1
        for elem in A:
            prod *= elem
        for i, elem in enumerate(A):
            A[i] = prod // elem
        return A




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--integers', type=int, nargs='+',
                        help='list of integers')
    args = parser.parse_args()
    print(multiplicate(args.integers))

#Временная сложность в худшем случае O(n)
#Пространственная сложность в худшем случае O(n)

