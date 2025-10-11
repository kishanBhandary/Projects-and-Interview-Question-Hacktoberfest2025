#include <iostream>
#include <string>
#include <unordered_map>
#include <climits>
using namespace std;

string minWindow(string s, string t) {
    unordered_map<char,int> need, window;
    for(char c: t) need[c]++;
    int left=0, right=0, valid=0, start=0, len=INT_MAX;
    while(right < s.size()){
        char c = s[right];
        right++;
        if(need.count(c)){
            window[c]++;
            if(window[c]==need[c]) valid++;
        }
        while(valid == need.size()){
            if(right-left < len){
                start = left;
                len = right-left;
            }
            char d = s[left];
            left++;
            if(need.count(d)){
                if(window[d]==need[d]) valid--;
                window[d]--;
            }
        }
    }
    return len==INT_MAX ? "" : s.substr(start,len);
}

int main(){
    string s, t;
    cout << "Enter string S: "; cin >> s;
    cout << "Enter string T: "; cin >> t;

    string res = minWindow(s, t);
    if(res.empty()) cout << "No valid window found." << endl;
    else cout << "Minimum window substring: " << res << endl;
    return 0;
}
