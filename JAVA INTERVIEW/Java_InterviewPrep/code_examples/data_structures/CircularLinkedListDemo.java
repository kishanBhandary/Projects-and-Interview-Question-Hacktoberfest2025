import java.util.Scanner;

class CLLNode {
    int data;
    CLLNode next;

    CLLNode(int data) {
        this.data = data;
        next = null;
    }
}

public class CircularLinkedListDemo {
    CLLNode head;

    // Insert at end
    public void insert(int data) {
        CLLNode newNode = new CLLNode(data);
        if (head == null) {
            head = newNode;
            newNode.next = head;
            return;
        }
        CLLNode temp = head;
        while (temp.next != head) temp = temp.next;
        temp.next = newNode;
        newNode.next = head;
    }

    // Display
    public void display() {
        if (head == null) return;
        CLLNode temp = head;
        System.out.print("Circular Linked List: ");
        do {
            System.out.print(temp.data + " ");
            temp = temp.next;
        } while (temp != head);
        System.out.println();
    }

    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            CircularLinkedListDemo cll = new CircularLinkedListDemo();
            System.out.print("Enter number of elements: ");
            int n = sc.nextInt();
            System.out.println("Enter elements:");
            for (int i = 0; i < n; i++) cll.insert(sc.nextInt());
            cll.display();
        }
    }
}
