def matrix(A, m):
    n = len(A)
    result = [[int(i == j) for j in range(n)] for i in range(n)] 
    for _ in range(m):
        result = multiply_matrix(result, A)
    return result

def multiply_matrix(A, B):
    n = len(A)
    res = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                res[i][j] += A[i][k] * B[k][j]
    return res

A = [
    [1, 2],
    [3, 4]
]
m = 2
result = matrix(A, m)
print("Ans is:")
for row in result:
    print(row)
