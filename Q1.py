def count_pairs(lst):
    c = 0
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            if lst[i] + lst[j] == 10:
                c += 1
    return c

lst = [2, 7, 4, 1, 3, 6]
print(cnt_pairs(lst))
