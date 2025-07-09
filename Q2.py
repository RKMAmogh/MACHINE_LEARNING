def range(nums):
    if len(nums) < 3:
        return "No Range determination"
    return max(nums) - min(nums)

nums = [5, 3, 8, 1, 0, 4]
print(get_range(nums))
