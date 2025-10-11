#include <stdio.h>

#define MAX 100


int indexSequentialSearch(int arr[], int n, int blockSize, int key) {
    int index[MAX], k = 0;

    
    for (int i = 0; i < n; i += blockSize)
        index[k++] = i;

    
    int block = -1;
    for (int i = 0; i < k; i++) {
        if (arr[index[i]] <= key && (i == k - 1 || arr[index[i + 1]] > key)) {
            block = i;
            break;
        }
    }

    
    if (block == -1)
        return -1;

    
    int start = index[block];
    int end = (start + blockSize < n) ? (start + blockSize) : n;

    for (int i = start; i < end; i++) {
        if (arr[i] == key)
            return i;
    }

    return -1;
}

int main() {
    int arr[] = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
    int n = sizeof(arr) / sizeof(arr[0]);
    int key = 35;
    int blockSize = 3;

    printf("Array: ");
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");

    int result = indexSequentialSearch(arr, n, blockSize, key);

    if (result != -1)
        printf("Element %d found at index %d\n", key, result);
    else
        printf("Element %d not found\n", key);

    return 0;
}
