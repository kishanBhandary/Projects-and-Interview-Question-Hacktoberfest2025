#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
using namespace std;

struct Book {
    int id;
    string title;
    string author;
    bool issued;
};

vector<Book> books;

void saveBooks() {
    ofstream file("library.txt");
    for (auto &b : books)
        file << b.id << " " << b.title << " " << b.author << " " << b.issued << "\n";
    file.close();
}

void loadBooks() {
    ifstream file("library.txt");
    Book b;
    while (file >> b.id >> b.title >> b.author >> b.issued)
        books.push_back(b);
    file.close();
}

void addBook() {
    Book b;
    cout << "Enter Book ID: ";
    cin >> b.id;
    cout << "Enter Title: ";
    cin >> b.title;
    cout << "Enter Author: ";
    cin >> b.author;
    b.issued = false;
    books.push_back(b);
    saveBooks();
    cout << "✅ Book added successfully!\n";
}

void issueBook() {
    int id;
    cout << "Enter Book ID to issue: ";
    cin >> id;
    for (auto &b : books) {
        if (b.id == id && !b.issued) {
            b.issued = true;
            saveBooks();
            cout << "📚 Book issued successfully!\n";
            return;
        } else if (b.id == id && b.issued) {
            cout << "❌ Book already issued!\n";
            return;
        }
    }
    cout << "❌ Book not found!\n";
}

void viewBooks() {
    cout << "\nID   Title       Author      Status\n";
    cout << "-----------------------------------\n";
    for (auto &b : books)
        cout << setw(4) << b.id << setw(12) << b.title << setw(12) << b.author
             << setw(10) << (b.issued ? "Issued" : "Available") << "\n";
}

int main() {
    loadBooks();
    int choice;
    do {
        cout << "\n--- Library Management System ---\n";
        cout << "1. Add Book\n2. View Books\n3. Issue Book\n4. Exit\n";
        cout << "Enter choice: ";
        cin >> choice;
        switch (choice) {
            case 1: addBook(); break;
            case 2: viewBooks(); break;
            case 3: issueBook(); break;
            case 4: cout << "Exiting...\n"; break;
            default: cout << "Invalid option!\n";
        }
    } while (choice != 4);
    return 0;
}