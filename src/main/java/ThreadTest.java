import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

public class ThreadTest {
    public static void main1(String[] s) throws Exception{

        Thread thread1=new Thread(()->{
            for(int i=0;i<10;i++){
                try {
                    System.out.println(String.format("线程%s运行：%d", Thread.currentThread().getName(), i));
                    Thread.sleep(1000);
                }
                catch (Exception e){}
            }
        });
        myThread myThread=new myThread(thread1);
        myThread.start();
        synchronized (myThread){
            myThread.wait();
        }
        myThread.join();
//        thread1.join(3000);
        System.out.println("主线程开始执行");
        for(int i=0;i<3;i++){
            try {
                System.out.println(String.format("线程%s运行：%d", Thread.currentThread().getName(), i));
                Thread.sleep(1000);
            }
            catch (Exception e){}
        }
    }

    public static void main(String[] args) throws InterruptedException {
        Thread myThread = new Thread(new Runnable() {
            @Override
            public void run() {
                for (int i = 0;i < 10; ++i) {
                    System.out.println("work");
                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        });
        synchronized (myThread) {
            myThread.wait();
        }
        myThread.join();
    }
}
class myThread extends Thread{
    Thread thread;
    public myThread(Thread thread){
        this.thread=thread;
    }
    public void run(){
        for(int i=0;i<3;i++){
            try {
//                synchronized (thread) {
                    System.out.println(String.format("线程%s运行：%d", Thread.currentThread().getName(), i));
                    Thread.sleep(1000);
//                }
            }
            catch (Exception e){}
        }
    }
}