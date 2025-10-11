#include <stdio.h>

int countWays(int coins[], int n, int sum) {
    int table[sum + 1];
    for (int i = 0; i <= sum; i++)
        table[i] = 0;

    table[0] = 1; // Base case

    for (int i = 0; i < n; i++)
        for (int j = coins[i]; j <= sum; j++)
            table[j] += table[j - coins[i]];

    return table[sum];
}

int main() {
    int coins[] = {1, 2, 3};
    int n = sizeof(coins) / sizeof(coins[0]);
    int sum = 4;
    printf("Number of ways to make change for %d is %d\n", sum, countWays(coins, n, sum));
    return 0;
}
