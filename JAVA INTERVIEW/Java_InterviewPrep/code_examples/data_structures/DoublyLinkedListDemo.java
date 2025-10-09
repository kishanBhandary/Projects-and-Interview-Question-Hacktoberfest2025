import java.util.Scanner;

class DLLNode {
    int data;
    DLLNode prev, next;

    DLLNode(int data) {
        this.data = data;
        prev = next = null;
    }
}

public class DoublyLinkedListDemo {
    DLLNode head;

    // Insert at end
    public void insert(int data) {
        DLLNode newNode = new DLLNode(data);
        if (head == null) {
            head = newNode;
            return;
        }
        DLLNode temp = head;
        while (temp.next != null) temp = temp.next;
        temp.next = newNode;
        newNode.prev = temp;
    }

    // Display forward
    public void displayForward() {
        DLLNode temp = head;
        System.out.print("Forward: ");
        while (temp != null) {
            System.out.print(temp.data + " ");
            temp = temp.next;
        }
        System.out.println();
    }

    // Display backward
    public void displayBackward() {
        DLLNode temp = head;
        if (temp == null) return;
        while (temp.next != null) temp = temp.next; // go to last
        System.out.print("Backward: ");
        while (temp != null) {
            System.out.print(temp.data + " ");
            temp = temp.prev;
        }
        System.out.println();
    }

    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            DoublyLinkedListDemo dll = new DoublyLinkedListDemo();
            System.out.print("Enter number of elements to insert: ");
            int n = sc.nextInt();
            System.out.println("Enter elements:");
            for (int i = 0; i < n; i++) dll.insert(sc.nextInt());

            dll.displayForward();
            dll.displayBackward();
        }
    }
}
