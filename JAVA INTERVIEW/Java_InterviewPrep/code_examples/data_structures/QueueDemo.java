import java.util.Scanner;

class QueueNode {
    int data;
    QueueNode next;

    QueueNode(int data) {
        this.data = data;
        this.next = null;
    }
}

public class QueueDemo {
    private QueueNode front, rear;

    // Enqueue element
    public void enqueue(int data) {
        QueueNode newNode = new QueueNode(data);
        if (rear == null) {
            front = rear = newNode;
            return;
        }
        rear.next = newNode;
        rear = newNode;
    }

    // Dequeue element
    public int dequeue() {
        if (front == null) {
            System.out.println("Queue Underflow!");
            return -1;
        }
        int data = front.data;
        front = front.next;
        if (front == null) rear = null;
        return data;
    }

    // Peek front element
    public int peek() {
        if (front == null) {
            System.out.println("Queue is empty!");
            return -1;
        }
        return front.data;
    }

    // Display queue
    public void display() {
        QueueNode temp = front;
        System.out.print("Queue (front -> rear): ");
        while (temp != null) {
            System.out.print(temp.data + " ");
            temp = temp.next;
        }
        System.out.println();
    }

    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            QueueDemo queue = new QueueDemo();
            System.out.print("Enter number of elements to enqueue: ");
            int n = sc.nextInt();

            System.out.println("Enter elements:");
            for (int i = 0; i < n; i++) {
                queue.enqueue(sc.nextInt());
            }

            queue.display();
            System.out.println("Front element: " + queue.peek());

            System.out.print("Enter number of elements to dequeue: ");
            int dq = sc.nextInt();
            for (int i = 0; i < dq; i++) {
                System.out.println("Dequeued: " + queue.dequeue());
            }

            queue.display();
        }
    }
}
