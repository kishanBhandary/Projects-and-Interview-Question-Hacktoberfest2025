#include <iostream>
#include <vector>
using namespace std;

int maxSumSubarray(vector<int>& arr, int k) {
    int n = arr.size();
    int max_sum = 0, window_sum = 0;

    for (int i = 0; i < k; i++) {
        window_sum += arr[i];
    }
    max_sum = window_sum;

    for (int i = k; i < n; i++) {
        window_sum += arr[i] - arr[i - k]; // Slide the window
        max_sum = max(max_sum, window_sum);
    }
    return max_sum;
}

int main() {
    int n, k;
    cout << "Enter array size and window size: ";
    cin >> n >> k;
    vector<int> arr(n);
    cout << "Enter array elements: ";
    for (int i = 0; i < n; i++) cin >> arr[i];

    cout << "Maximum sum of subarray of size " << k << " is " << maxSumSubarray(arr, k) << endl;
    return 0;
}
