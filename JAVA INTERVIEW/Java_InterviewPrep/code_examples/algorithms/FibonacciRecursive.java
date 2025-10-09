import java.util.Scanner;

public class FibonacciRecursive {
    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            System.out.print("Enter number of terms: ");
            int n = sc.nextInt();

            System.out.print("Fibonacci series: ");
            for (int i = 0; i < n; i++) {
                System.out.print(fib(i) + " ");
            }
        }
    }

    public static int fib(int n) {
        if (n <= 1) return n;
        return fib(n - 1) + fib(n - 2);
    }
}
