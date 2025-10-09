import java.util.Scanner;

class Node {
    int data;
    Node next;

    Node(int data) {
        this.data = data;
        this.next = null;
    }
}

public class LinkedListDemo {
    Node head;

    // Insert at end
    public void insert(int data) {
        Node newNode = new Node(data);
        if (head == null) {
            head = newNode;
            return;
        }
        Node temp = head;
        while (temp.next != null) temp = temp.next;
        temp.next = newNode;
    }

    // Delete first occurrence of value
    public void delete(int key) {
        if (head == null) return;

        if (head.data == key) {
            head = head.next;
            return;
        }

        Node temp = head;
        while (temp.next != null && temp.next.data != key) {
            temp = temp.next;
        }

        if (temp.next != null) temp.next = temp.next.next;
    }

    // Search for a value
    public boolean search(int key) {
        Node temp = head;
        while (temp != null) {
            if (temp.data == key) return true;
            temp = temp.next;
        }
        return false;
    }

    // Display list
    public void display() {
        Node temp = head;
        while (temp != null) {
            System.out.print(temp.data + " -> ");
            temp = temp.next;
        }
        System.out.println("null");
    }

    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            LinkedListDemo list = new LinkedListDemo();
            System.out.print("Enter number of elements to insert: ");
            int n = sc.nextInt();
            System.out.println("Enter elements:");
            for (int i = 0; i < n; i++) {
                list.insert(sc.nextInt());
            }

            System.out.println("Current Linked List:");
            list.display();

            System.out.print("Enter element to delete: ");
            int del = sc.nextInt();
            list.delete(del);
            System.out.println("Linked List after deletion:");
            list.display();

            System.out.print("Enter element to search: ");
            int key = sc.nextInt();
            System.out.println(list.search(key) ? key + " found!" : key + " not found.");
        }
    }
}
