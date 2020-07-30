import java.util.HashMap;

public class LRUCache {

    public LRUCache(int capacity) {
        this.capacity=capacity;
        index=new HashMap();
        now=0;
        head=new Node(0,0);
        tail=new Node(0,0);
        head.previous=null;
        head.next=tail;
        tail.previous=head;
        tail.next=null;
    }

    void addToHead(Node node){
        node.next=head.next;
        node.previous=head;
        head.next.previous=node;
        head.next=node;
        index.put(node.key,node);
        now++;
    }

    void removeNode(Node node){
        node.previous.next=node.next;
        node.next.previous=node.previous;
        index.remove(node.key,node);
        now--;
    }

    public int get(int key) {
        if(!index.containsKey(key)){
            return -1;
        }
        else {
            Node node=index.get(key);
            removeNode(node);
            addToHead(node);
            return node.value;
        }
    }

    public void put(int key, int value) {
        if(index.containsKey(key)){
            Node node=index.get(key);
            node.value=value;
            removeNode(node);
            addToHead(node);
        }
        else {
            if (now<capacity){
                addToHead(new Node(key,value));
            }
            else {
                removeNode(tail.previous);
                addToHead(new Node(key,value));
            }
        }
    }
    Node head;
    Node tail;
    int capacity;
    int now;
    HashMap<Integer,Node> index;

    class Node{
        Node previous;
        Node next;
        int key;
        int value;
        Node(int key,int val){
            this.key=key;
            this.value=val;
        }
        Node(){
        }
    }
}

