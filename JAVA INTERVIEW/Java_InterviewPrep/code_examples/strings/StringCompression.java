import java.util.Scanner;

public class StringCompression {
    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            System.out.print("Enter a string: ");
            String str = sc.nextLine();

            StringBuilder sb = new StringBuilder();
            int count = 1;
            for (int i = 1; i <= str.length(); i++) {
                if (i < str.length() && str.charAt(i) == str.charAt(i - 1)) {
                    count++;
                } else {
                    sb.append(str.charAt(i - 1));
                    if (count > 1) sb.append(count);
                    count = 1;
                }
            }

            System.out.println("Compressed string: " + sb.toString());
        }
    }
}
