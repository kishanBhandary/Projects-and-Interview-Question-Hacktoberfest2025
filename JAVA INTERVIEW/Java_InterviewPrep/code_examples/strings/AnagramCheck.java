import java.util.Arrays;
import java.util.Scanner;

public class AnagramCheck {
    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            System.out.print("Enter first string: ");
            String str1 = sc.nextLine().replaceAll("\\s+", "").toLowerCase();
            System.out.print("Enter second string: ");
            String str2 = sc.nextLine().replaceAll("\\s+", "").toLowerCase();

            if (isAnagram(str1, str2))
                System.out.println("Strings are anagrams.");
            else
                System.out.println("Strings are not anagrams.");
        }
    }

    public static boolean isAnagram(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        char[] a1 = s1.toCharArray();
        char[] a2 = s2.toCharArray();
        Arrays.sort(a1);
        Arrays.sort(a2);
        return Arrays.equals(a1, a2);
    }
}
