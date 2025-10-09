import java.util.Arrays;
import java.util.Scanner;

public class BinarySearchRecursive {
    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            System.out.print("Enter number of elements: ");
            int n = sc.nextInt();
            int[] arr = new int[n];
            System.out.println("Enter elements (unsorted):");
            for (int i = 0; i < n; i++) arr[i] = sc.nextInt();

            Arrays.sort(arr); // Binary search requires sorted array
            System.out.println("Sorted array: " + Arrays.toString(arr));

            System.out.print("Enter element to search: ");
            int key = sc.nextInt();

            int index = binarySearch(arr, 0, arr.length - 1, key);
            if (index != -1)
                System.out.println(key + " found at index " + index);
            else
                System.out.println(key + " not found");
        }
    }

    public static int binarySearch(int[] arr, int low, int high, int key) {
        if (low > high) return -1;
        int mid = low + (high - low) / 2;
        if (arr[mid] == key) return mid;
        else if (arr[mid] < key) return binarySearch(arr, mid + 1, high, key);
        else return binarySearch(arr, low, mid - 1, key);
    }
}
