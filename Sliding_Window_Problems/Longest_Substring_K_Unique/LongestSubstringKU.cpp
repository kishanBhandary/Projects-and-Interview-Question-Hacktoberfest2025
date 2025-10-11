#include <iostream>
#include <string>
#include <unordered_map>
using namespace std;

int longestSubstringKUnique(string s, int k) {
    unordered_map<char, int> freq;
    int left = 0, max_len = 0;
    for (int right = 0; right < s.size(); right++) {
        freq[s[right]]++;
        while (freq.size() > k) {
            freq[s[left]]--;
            if (freq[s[left]] == 0) freq.erase(s[left]);
            left++;
        }
        max_len = max(max_len, right - left + 1);
    }
    return max_len;
}

int main() {
    string s;
    int k;
    cout << "Enter string: ";
    cin >> s;
    cout << "Enter K: ";
    cin >> k;

    cout << "Length of longest substring with " << k << " unique characters is " << longestSubstringKUnique(s, k) << endl;
    return 0;
}
