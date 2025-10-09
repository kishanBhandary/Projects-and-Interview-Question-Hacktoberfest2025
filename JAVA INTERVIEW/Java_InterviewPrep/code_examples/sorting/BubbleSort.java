import java.util.Scanner;

public class BubbleSort {
    public static void main(String[] args) {
        // try-with-resources automatically closes Scanner
        try (Scanner sc = new Scanner(System.in)) {
            System.out.print("Enter number of elements: ");
            int n = sc.nextInt();
            int[] arr = new int[n];
            System.out.println("Enter elements:");
            for (int i = 0; i < n; i++) arr[i] = sc.nextInt();

            bubbleSort(arr);

            System.out.println("Sorted array:");
            for (int num : arr) System.out.print(num + " ");
        } // no need to call sc.close()
    }

    public static void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n-1; i++) {
            for (int j = 0; j < n-i-1; j++) {
                if (arr[j] > arr[j+1]) {
                    int temp = arr[j]; arr[j] = arr[j+1]; arr[j+1] = temp;
                }
            }
        }
    }
}
