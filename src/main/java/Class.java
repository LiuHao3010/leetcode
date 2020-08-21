import java.util.*;

public class Class {
}
class LRUCache {

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

class MedianFinder {
    PriorityQueue<Integer> low=new PriorityQueue();
    PriorityQueue<Integer> high=new PriorityQueue(new Comparator<Integer>() {
        @Override
        public int compare(Integer o1, Integer o2) {
            return o2-o1;
        }
    });
    public MedianFinder() {
    }

    public void addNum(int num) {
        low.offer(num);
        high.add(low.poll());
        if(high.size()>low.size()){
            low.add(high.poll());
        }
    }

    public double findMedian() {
        if(low.size()>high.size()){
            return low.peek();
        }
        else {
            return ((double)(low.peek()+high.peek()))/2;
        }
    }
}

class MinStack {

    /** initialize your data structure here. */

    Stack<Integer> data;
    Stack<Integer> minData;
    public MinStack() {
        data=new Stack();
        minData=new Stack();
    }

    public void push(int x) {
        data.push(x);
        if(minData.isEmpty()||minData.peek()>x){
            minData.push(x);
        }
        else {
            minData.push(minData.peek());
        }
    }

    public void pop() {
        data.pop();
        minData.pop();
    }

    public int top() {
        return data.peek();
    }

    public int getMin() {
        return minData.peek();
    }
}

class QueueWithMax {
    class innerData{
        int val;
        int index;
        innerData(int val,int index){
            this.val=val;
            this.index=index;
        }
    }
    ArrayDeque<innerData> data=new ArrayDeque();
    ArrayDeque<innerData> max=new ArrayDeque();
    int curindex=0;
    void offer(int num){
        innerData innerData=new innerData(num,curindex);
        data.addLast(innerData);
        while (!max.isEmpty()&&max.getLast().val<num)
            max.removeLast();
        max.addLast(innerData);
        curindex++;
    }

    void remove(){
        innerData innerData=data.removeFirst();
        if(innerData.index==max.getFirst().index)
            max.removeFirst();
    }

    int getMax(){
        return max.getFirst().val;
    }

}

class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        return preorder(root)+"="+inorder(root);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] order=data.split("=");
        if(order.length==0){
            return null;
        }
        else {
            String[] preorder=order[0].split(",");
            String[] inorder=order[1].split(",");
            int[] per=new int[preorder.length];
            int[] in=new int[preorder.length];
            for(int i=0;i<per.length;i++){
                per[i]=Integer.parseInt(preorder[i]);
                in[i]=Integer.parseInt(inorder[i]);
            }
            return help(per,0,per.length-1,in,0,in.length-1);
        }
    }

    String inorder(TreeNode root){
        if(root==null)
            return "";
        StringBuilder res=new StringBuilder();
        Stack<TreeNode> stack=new Stack();
        TreeNode tmp=root;
        while (!stack.isEmpty()||tmp!=null){
            if(tmp!=null){
                stack.push(tmp);
                tmp=tmp.left;
            }
            else {
                tmp=stack.pop();
                res.append(tmp.val);
                res.append(",");
                tmp=tmp.right;
            }
        }
        return res.substring(0,res.length()-1);
    }

    String preorder(TreeNode root){
        if(root==null)
            return "";
        StringBuilder res=new StringBuilder();
        Stack<TreeNode> stack=new Stack();
        TreeNode tmp=root;
        while (!stack.isEmpty()||tmp!=null){
            if(tmp!=null){
                res.append(tmp.val);
                res.append(",");
                stack.push(tmp);
                tmp=tmp.left;
            }
            else {
                tmp=stack.pop();
                tmp=tmp.right;
            }
        }
        return res.substring(0,res.length()-1);
    }

    TreeNode help(int[] pre,int preStart,int preEnd,int[] in,int inStart,int inEnd){
        if(preStart>preEnd || inStart > inEnd){ //到达边界条件时返回null
            return null;
        }
        TreeNode  treeNode =new TreeNode(pre[preStart]);   //新建一个TreeNode
        int i=inStart;
        for(;i<inEnd;i++){
            if(in[i]==pre[preStart])
                break;
        }
        treeNode.left=help(pre,preStart+1,preStart+i-inStart,in,inStart,i-1);
        treeNode.right=help(pre,preStart+i-inStart+1,preEnd,in,i+1,inEnd);
        return treeNode;
    }
}