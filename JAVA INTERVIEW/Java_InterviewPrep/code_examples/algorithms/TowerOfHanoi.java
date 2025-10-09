import java.util.Scanner;

public class TowerOfHanoi {
    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            System.out.print("Enter number of disks: ");
            int n = sc.nextInt();
            System.out.println("The sequence of moves:");
            solveHanoi(n, 'A', 'C', 'B'); // A = source, C = destination, B = auxiliary
        }
    }

    // Recursive method to solve Tower of Hanoi
    public static void solveHanoi(int n, char source, char destination, char auxiliary) {
        if (n == 1) {
            System.out.println("Move disk 1 from " + source + " to " + destination);
            return;
        }
        solveHanoi(n - 1, source, auxiliary, destination);
        System.out.println("Move disk " + n + " from " + source + " to " + destination);
        solveHanoi(n - 1, auxiliary, destination, source);
    }
}
