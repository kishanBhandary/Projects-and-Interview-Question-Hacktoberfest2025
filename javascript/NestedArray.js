

var nestedArray = [
    [5, 6, 7, -2],
    [-5, -6, -7],
    [8, 9, 10],
    [3, 5, 2, 1, 4],
    [-3, 5, 2, 1,],
    [4, 2, 6, 8],
]

//1. Find Maximum: Write a function to find the maximum number in a nested array of integers.
function findMaximum(nestedArray) {
    let maximum = nestedArray[0][0]
    nestedArray.forEach(element => {
        element.forEach(item => maximum = item > maximum ? item : maximum)
    });
    return maximum
}

console.log('Maximum:', findMaximum(nestedArray));

//2. Calculate Average: Create a function to calculate the average of all numbers in a nested array.
function average(nestedArray) {
    let sum=0,count=0
    nestedArray.forEach(element => {
        element.forEach(item => {
            sum+=item
            count++
        })
    })
    return (sum/count).toFixed(2)
}
console.log('Average:',average(nestedArray));

//3. Count Negative Numbers: Implement a function that counts the number of negative numbers in a nested array.
function countNegative(nestedArray) {
    let count=0
    for (const element of nestedArray) {
        for (const item of element) {
            if (item<0) {
                count++
            }
        }
    }
    return count
}
console.log('Negative numbers:',countNegative(nestedArray));

//4. Subarray Sums: Write a function that returns an array of sums of each subarray within the nested array.
function sumOfSubArray(nestedArray) {
    return nestedArray.map(subArray => {
        return subArray.reduce((sum,currentNumber) => {
            return sum+currentNumber
        },0)
    })
}
console.log("The sums of each subarray are:", sumOfSubArray(nestedArray))

//5. Sort Subarrays: Implement a function that sorts each subarray in a nested array of numbers.
function sortSubArray(nestedArray) {
    return nestedArray.map(subArray=> {
        return [...subArray].sort((a,b)=>a-b)
    })
}
console.log('Sort Subarrays:');

console.log(sortSubArray(nestedArray));

//6. Flatten Nested Array: Write a function to flatten a nested array to a single-level array.
function flattenNestedArray(nestedArray) {
    return nestedArray.flat()
}
console.log('Flatten Nested Array',flattenNestedArray(nestedArray));

//7. Remove Duplicates: Create a function that removes duplicate elements from the nested array.

//8. Reverse Subarrays: Implement a function to reverse each subarray within the nested array.
function reverseSubArray(nestedArray) {
    return nestedArray.map(subArray => {
        for (let index=0;index < (subArray.length/2); index++) {
                let temp = subArray[index];
                subArray[index] = subArray[subArray.length-1-index]
                subArray[subArray.length-1-index]=temp
        }
        return subArray
    })
}
console.log('Reverse Subarrays');
console.log(reverseSubArray(nestedArray));

//9. Filter Even Numbers: Write a function to filter out all even numbers from the nested array.
function evenArray(nestedArray) {
    return nestedArray.map(subArray=>{
        return subArray.filter(item=>item%2==0)
    })
}
console.log('Even numbers array:',evenArray(nestedArray));

//10. Find Longest Subarray: Create a function that returns the longest subarray within the nested array.
function longestSubArray(nestedArray){
    return nestedArray.reduce((longest,current) => {
        return longest.length >= current.length ? longest : current
    })
}
console.log("The longest subarray is:", longestSubArray(nestedArray))