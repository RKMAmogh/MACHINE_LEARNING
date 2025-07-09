nums = [3, 7, 1, 9, 4, 2, 8, 6, 5, 10, 2, 3, 7, 6, 5, 1, 4, 2, 9, 8, 10, 3, 1, 6, 7]

mean = sum(nums) / len(nums)

nums.sort()
n = len(nums)
if n % 2 == 0:
    median = (nums[n//2 - 1] + nums[n//2]) / 2
else:
    median = nums[n//2]

f = {}
for i in nums:
    if i in f:
        f[i] += 1
    else:
        f[i] = 1

m = max(f.values())
for k in f:
    if f[k] == m:
        mode = k
        break

print("nums:", nums)
print("mean:", round(mean, 2))
print("median:", median)
print("mode:", mode)
