
import java.util.Stack;

public class MinStack {

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
