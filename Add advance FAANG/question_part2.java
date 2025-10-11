/**
 * AdvancedJavaPart2.java
 *
 * Advanced Java concepts for FAANG/Product-level interviews.
 * Detailed line-by-line explanations included in comments.
 *
 * Topics:
 * 1. Advanced Collections
 * 2. Concurrent Collections
 * 3. Design Patterns: Singleton, Factory
 * 4. AtomicInteger concurrency
 * 5. LRU Cache using LinkedHashMap
 */

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class AdvancedJavaPart2 {

    // 1️⃣ PriorityQueue example (min-heap)
    public static void priorityQueueExample() {
        PriorityQueue<Integer> pq = new PriorityQueue<>(); // min-heap
        pq.add(30);
        pq.add(10);
        pq.add(20);

        System.out.println("PriorityQueue elements (poll order - min first):");
        while (!pq.isEmpty()) {
            System.out.println(pq.poll()); // retrieves smallest element first
        }

        // Explanation:
        // Internal structure: Binary heap
        // Each parent node <= child nodes (min-heap)
    }

    // 2️⃣ ConcurrentHashMap vs synchronizedMap
    public static void concurrentMapExample() {
        Map<String, Integer> syncMap = Collections.synchronizedMap(new HashMap<>()); // synchronized wrapper
        ConcurrentHashMap<String, Integer> concurrentMap = new ConcurrentHashMap<>(); // concurrent map

        // Thread-safe operations
        concurrentMap.put("a", 1);
        concurrentMap.put("b", 2);
        System.out.println("ConcurrentHashMap: " + concurrentMap);

        // Explanation:
        // synchronizedMap locks entire map → low concurrency
        // ConcurrentHashMap locks at bucket/segment level → higher concurrency
    }

    // 3️⃣ Singleton Pattern (Thread-safe)
    public static class Singleton {
        private static volatile Singleton instance; // volatile ensures visibility

        private Singleton() {} // private constructor

        public static Singleton getInstance() {
            if (instance == null) {
                synchronized (Singleton.class) { // double-checked locking
                    if (instance == null) {
                        instance = new Singleton();
                    }
                }
            }
            return instance;
        }
    }

    // 4️⃣ Factory Pattern Example
    interface Shape { void draw(); }

    static class Circle implements Shape { 
        public void draw() { System.out.println("Drawing Circle"); } 
    }

    static class Square implements Shape { 
        public void draw() { System.out.println("Drawing Square"); } 
    }

    static class ShapeFactory {
        public static Shape getShape(String type) {
            if ("circle".equalsIgnoreCase(type)) return new Circle();
            else if ("square".equalsIgnoreCase(type)) return new Square();
            else return null;
        }
    }

    public static void designPatternExample() {
        Shape s1 = ShapeFactory.getShape("circle");
        s1.draw(); // Drawing Circle

        Shape s2 = ShapeFactory.getShape("square");
        s2.draw(); // Drawing Square
    }

    // 5️⃣ AtomicInteger Example for thread-safe counter
    public static void atomicExample() throws InterruptedException {
        AtomicInteger atomicCount = new AtomicInteger(0);
        int threads = 1000;
        Thread[] tArr = new Thread[threads];

        for (int i = 0; i < threads; i++) {
            tArr[i] = new Thread(atomicCount::incrementAndGet); // atomic increment
            tArr[i].start();
        }
        for (Thread t : tArr) t.join();
        System.out.println("AtomicInteger final count: " + atomicCount.get());

        // Explanation:
        // No need for synchronized
        // AtomicInteger ensures thread-safe increment operations
    }

    // 6️⃣ LRU Cache using LinkedHashMap
    static class LRUCache<K, V> extends LinkedHashMap<K, V> {
        private final int capacity;

        public LRUCache(int capacity) {
            super(capacity, 0.75f, true); // access-order mode
            this.capacity = capacity;
        }

        @Override
        protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
            return size() > capacity; // remove least recently used
        }
    }

    public static void lruCacheExample() {
        LRUCache<Integer, String> cache = new LRUCache<>(3);
        cache.put(1, "A");
        cache.put(2, "B");
        cache.put(3, "C");
        cache.get(1); // access 1 → makes it recently used
        cache.put(4, "D"); // evicts key 2 (least recently used)

        System.out.println("LRU Cache state: " + cache);

        // Explanation:
        // LinkedHashMap with accessOrder=true maintains order based on access
        // removeEldestEntry handles eviction automatically
    }

    public static void main(String[] args) throws InterruptedException {
        System.out.println("==== PriorityQueue Example ====");
        priorityQueueExample();

        System.out.println("\n==== ConcurrentHashMap Example ====");
        concurrentMapExample();

        System.out.println("\n==== Design Pattern Example ====");
        designPatternExample();

        System.out.println("\n==== AtomicInteger Example ====");
        atomicExample();

        System.out.println("\n==== LRU Cache Example ====");
        lruCacheExample();
    }
}
