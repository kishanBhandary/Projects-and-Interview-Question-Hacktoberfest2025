#include <stdio.h>

int LIS(int arr[], int n) {
    int lis[n];
    for (int i = 0; i < n; i++)
        lis[i] = 1;

    for (int i = 1; i < n; i++)
        for (int j = 0; j < i; j++)
            if (arr[i] > arr[j] && lis[i] < lis[j] + 1)
                lis[i] = lis[j] + 1;

    int max = lis[0];
    for (int i = 1; i < n; i++)
        if (lis[i] > max)
            max = lis[i];
    return max;
}

int main() {
    int arr[] = {10, 22, 9, 33, 21, 50, 41, 60};
    int n = sizeof(arr) / sizeof(arr[0]);
    printf("Length of Longest Increasing Subsequence is %d\n", LIS(arr, n));
    return 0;
}
