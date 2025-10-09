import java.util.Scanner;

public class PrimesInRange {
    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            System.out.print("Enter start of range: ");
            int start = sc.nextInt();
            System.out.print("Enter end of range: ");
            int end = sc.nextInt();

            System.out.println("Prime numbers between " + start + " and " + end + ":");
            for (int i = start; i <= end; i++) {
                if (isPrime(i)) System.out.print(i + " ");
            }
        }
    }

    public static boolean isPrime(int n) {
        if (n <= 1) return false;
        for (int i = 2; i <= Math.sqrt(n); i++) {
            if (n % i == 0) return false;
        }
        return true;
    }
}
