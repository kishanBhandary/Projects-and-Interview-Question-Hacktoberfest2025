# Question:
# You are given an infinite supply of coins in n different denominations, represented by the array coins = [C0, C1, C2, ..., Cn-1].
# Your task is to determine the total number of distinct combinations to make up a given total sum, using any number of coins from the available denominations.
#
# Each coin denomination can be used any number of times.
# Note: The order of coins does not matter. That is, combinations like [1, 2] and [2, 1] are considered the same and counted only once.
#
# Input:
# Implement the function count(coins, N, target) which takes:
# - coins: List of integers representing coin denominations
# - N: Size of the coins array
# - target: The target total sum
#
# Output:
# Return the number of ways to get the target sum.
#
# Constraints:
# 1 <= N <= 12
# 1 <= target <= 1000
# 1 <= coins[i] <= 10^6
#
# Example:
# Input:
# coins = [1,2,3], N=3, target=4
# Output: 4
# Explanation: The total number of ways are 4: [1,1,1,1], [1,1,2], [1,3], [2,2]

# Solution:

def count(coins, N, target):
    n = len(coins)
    dp = [[-1 for _ in range(n)] for _ in range(target + 1)]
    
    def helper(value, idx):
        if value == 0:
            return 1
        if idx >= n or value < 0:
            return 0
        if dp[value][idx] != -1:
            return dp[value][idx]
        
        pick = 0
        if coins[idx] <= value:
            pick = helper(value - coins[idx], idx)
        notpick = helper(value, idx + 1)
        dp[value][idx] = pick + notpick
        return dp[value][idx]
    
    return helper(target, 0)

# Test cases
if __name__ == "__main__":
    test_cases = [
        ([1, 2, 3], 3, 4),       # Expected output: 4
        ([2, 5, 3, 6], 4, 10),   # Expected output: 5
        ([1, 2], 2, 3),          # Expected output: 2
        ([3, 5, 7], 3, 12),      # Expected output: 4
        ([1, 2, 5], 3, 5)        # Expected output: 4
    ]
    
    for coins, N, target in test_cases:
        print(f"coins={coins}, target={target} => ways={count(coins, N, target)}")
