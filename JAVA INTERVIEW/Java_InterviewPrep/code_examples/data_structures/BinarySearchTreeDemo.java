import java.util.Scanner;

class BSTNode {
    int data;
    BSTNode left, right;

    BSTNode(int data) {
        this.data = data;
        left = right = null;
    }
}

public class BinarySearchTreeDemo {
    BSTNode root;

    public void insert(int data) {
        root = insertRec(root, data);
    }

    private BSTNode insertRec(BSTNode node, int data) {
        if (node == null) return new BSTNode(data);
        if (data < node.data) node.left = insertRec(node.left, data);
        else node.right = insertRec(node.right, data);
        return node;
    }

    public void inorder() {
        inorderRec(root);
        System.out.println();
    }

    private void inorderRec(BSTNode node) {
        if (node != null) {
            inorderRec(node.left);
            System.out.print(node.data + " ");
            inorderRec(node.right);
        }
    }

    public static void main(String[] args) {
        try (Scanner sc = new Scanner(System.in)) {
            BinarySearchTreeDemo bst = new BinarySearchTreeDemo();
            System.out.print("Enter number of elements to insert: ");
            int n = sc.nextInt();
            System.out.println("Enter elements:");
            for (int i = 0; i < n; i++) bst.insert(sc.nextInt());

            System.out.print("Inorder traversal: ");
            bst.inorder();
        }
    }
}
