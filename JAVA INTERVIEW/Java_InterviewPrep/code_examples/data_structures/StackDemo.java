import java.util.Scanner;

class StackNode {
    int data;
    StackNode next;

    StackNode(int data) {
        this.data = data;
        this.next = null;
    }
}

public class StackDemo {
    private StackNode top;

    // Push element onto stack
    public void push(int data) {
        StackNode newNode = new StackNode(data);
        newNode.next = top;
        top = newNode;
    }

    // Pop element from stack
    public int pop() {
        if (top == null) {
            System.out.println("Stack Underflow!");
            return -1;
        }
        int data = top.data;
        top = top.next;
        return data;
    }

    // Peek top element
    public int peek() {
        if (top == null) {
            System.out.println("Stack is empty!");
            return -1;
        }
        return top.data;
    }

    // Display stack
    public void display() {
        StackNode temp = top;
        System.out.print("Stack (top -> bottom): ");
        while (temp != null) {
            System.out.print(temp.data + " ");
            temp = temp.next;
        }
        System.out.println();
    }

    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            StackDemo stack = new StackDemo();
            System.out.print("Enter number of elements to push: ");
            int n = sc.nextInt();

            System.out.println("Enter elements:");
            for (int i = 0; i < n; i++) {
                stack.push(sc.nextInt());
            }

            stack.display();
            System.out.println("Top element: " + stack.peek());

            System.out.print("Enter number of elements to pop: ");
            int pops = sc.nextInt();
            for (int i = 0; i < pops; i++) {
                System.out.println("Popped: " + stack.pop());
            }

            stack.display();
        }
    }
}
