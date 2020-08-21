
import org.apache.spark.sql.sources.In;
import scala.Char;
import scala.Int;
import sun.reflect.generics.tree.Tree;

import java.util.*;


public class Solution {
    int maxprofit=Integer.MIN_VALUE;
    int minPath=Integer.MAX_VALUE;
    int[] mx={1,-1,0,0};
    int[] my={0,0,1,-1};
    int max=0;
    boolean res=false;
    int res1=0;
    int res2=Integer.MIN_VALUE;
    public static void main(String[] a) throws Exception{
        Solution solution=new Solution();
       int[][] people={{9,0},{7,0},{1,9},{3,0},{2,7},{5,3},{6,0},{3,4},{6,2},{5,2}};
       int[] num={1,5,11,5};
       TreeNode t1=new TreeNode(10);
        TreeNode t2=new TreeNode(5);
        TreeNode t3=new TreeNode(-3);
        TreeNode t4=new TreeNode(3);
        TreeNode t5=new TreeNode(2);
        TreeNode t6=new TreeNode(11);
        TreeNode t7=new TreeNode(3);
        TreeNode t8=new TreeNode(-2);
        TreeNode t9=new TreeNode(1);
        t1.left=t2;
        t1.right=t3;
        t2.left=t4;
        t2.right=t5;
        t3.right=t6;
        t4.left=t7;
        t4.right=t8;
        t5.right=t9;
        int i=solution.pathSum(t1,8);
       solution.canPartition(num);
        int[][] res=solution.reconstructQueue(people);
    }
    public boolean possibleBipartition(int N, int[][] dislikes) {
        int[] visited=new int[N+1];
        ArrayList<Integer>[] g=new ArrayList[N+1];
        for(int i=1;i<=N;i++){
            g[i]=new ArrayList<>();
        }
        for(int[] i:dislikes){
            g[i[0]].add(i[1]);
            g[i[1]].add(i[0]);
        }
        for (int i=1;i<=N;i++){
            if(visited[i]==0&&!dfs(g,visited,i,1)) {
                return false;
            }
        }
        return true;
    }
    public boolean dfs(ArrayList<Integer>[] g,int[] visited,int n,int p){
        if(visited[n]==p) {
            return true;
        }
        if(visited[n]==-p) {
            return false;
        }
        visited[n]=p;
        for(int t:g[n]){
            if(!dfs(g, visited, t, 0-p)) {
                return false;
            }
        }
        return true;
    }

    public int maxProfit(int[] prices) {
        int[] s0=new int[prices.length];
        int[] s1=new int[prices.length];
        int[] s2=new int[prices.length];
        s0[0]=0;
        s1[0]=-prices[0];
        s2[0]=Integer.MIN_VALUE;
        for(int i=1;i<prices.length;i++){
            s0[i]=Math.max(s0[i-1],s2[i-1]);
            s1[i]=Math.max(s1[i-1],s0[i-1]-prices[i]);
            s2[i]=s1[i-1]+prices[i];
        }
        return Math.max(s0[prices.length-1],s2[prices.length-1]);
    }
    public void help(int[] prices,boolean flag,boolean hold,int profit,int day){
        if(profit>maxprofit){
            maxprofit=profit;
        }
        if(day<prices.length){
            if(!flag){
                help(prices,true,hold,profit,day+1);
            }
            else {
                help(prices,true,hold,profit,day+1);
                if(hold) {
                    help(prices, false, false, profit + prices[day], day + 1);
                }
                else {
                    help(prices,true,true,profit-prices[day],day+1);
                }
            }
        }
    }

    public List<String> letterCombinations(String digits) {
        List<String>  res=new ArrayList();
        Map<Character,char[]> map=new HashMap();
        char[] tmp2={'a','b','c'};
        char[] tmp3={'d','e','f'};
        char[] tmp4={'g','h','i'};
        char[] tmp5={'j','k','l'};
        char[] tmp6={'m','n','o'};
        char[] tmp7={'p','q','r','s'};
        char[] tmp8={'t','u','v'};
        char[] tmp9={'w','x','y','z'};
        map.put('2',tmp2);
        map.put('3',tmp3);
        map.put('4',tmp4);
        map.put('5',tmp5);
        map.put('6',tmp6);
        map.put('7',tmp7);
        map.put('8',tmp8);
        map.put('9',tmp9);
        help(0,digits,"",res,map);
        return res;
    }
    void help(int i,String digits,String str,List res,Map<Character,char[]> map){
        if(i==digits.length()){
            res.add(str);
            return;
        }
        else {
            char[] tmp=map.get(digits.charAt(i));
            for(char c:tmp){
                String s=str+c;
                help(i+1,digits,s,res,map);
            }
        }
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode res=new ListNode();
        res.next=head;
        ListNode n1=res;
        ListNode n2=res;
        for(int i=0;i<=n;i++){
            n2=n2.next;
        }
        while (n2.next!=null){
            n1=n1.next;
            n2=n2.next;
        }
        n1.next=n1.next.next;
        return res.next;
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode res=new ListNode();
        ListNode tmp=res;
        while (l1!=null&&l2!=null) {
            if (l1.val < l2.val) {
                tmp.next = l1;
                l1 = l1.next;
                tmp=tmp.next;
            }
            else {
                tmp.next=l2;
                l2=l2.next;
                tmp=tmp.next;
            }
        }
        if(l1==null){
            tmp.next=l2;
        }
        else {
            tmp.next=l1;
        }
        return res.next;
    }

    public List<String> generateParenthesis(int n) {
        List<String> res=new ArrayList();
        dfs(0,0,n,"",res);
        return res;
    }
    void dfs(int l,int r,int n,String tmp,List<String> res){
        if(l==n&&r==n){
            res.add(tmp);
            return;
        }
        else {
            if(l>r){
                if(l<n) {
                    dfs(l + 1, r, n, tmp + "(", res);
                }
                dfs(l,r+1,n,tmp+")",res);
            }
            else if(l==r&&l<n) {
                dfs(l+1,r,n,tmp+"(",res);
            }
        }
    }

    public int longestValidParentheses(String s) {
        Stack<Character> stack=new Stack();

        int max=0;
        int[] dp=new int[s.length()];
        for(int i=1;i<s.length();i++){
            if(s.charAt(i)==')'){
                if(s.charAt(i-1)=='('){
                    dp[i]=i>1?dp[i-2]+2:2;
                }
                else {
                    if((i-dp[i-1]-1)>=0&&s.charAt(i-dp[i-1]-1)=='('){
                        dp[i]=(i-dp[i-1]-2)>0?dp[i-1]+2+dp[i-dp[i-1]-2]:dp[i-1]+2;
                    }
                    else {
                        dp[i]=0;
                    }
                }
            }
        }
        for(int i:dp){
            max=max>i?max:i;
        }
        return max;
    }

    public int search(int[] nums, int target) {
        if(nums.length==0) {
            return -1;
        }
        int l=0;
        int r=nums.length-1;
        int m;
        while(l<=r){
            m=l+(r-l)/2;
            if(nums[m]==target){
                return m;
            }
            if(nums[m]>=nums[l]){
                if(nums[m]>target&&nums[l]<=target){
                    r=m-1;
                }
                else{
                    l=m+1;
                }
            }
            else{
                if(nums[m]<target&&nums[r]>=target){
                    l=m+1;
                }
                else{
                    r=m-1;
                }
            }
        }
        return -1;
    }

    public int[] searchRange(int[] nums, int target) {
        int l=findleft(nums,target);
        if(l==-1){
            int[]  res={-1};
            return res;
        }
        else {
            int r=findright(nums,target);
            if(l==r){
                int[] res={l,r};
                return res;
            }
            int[] res=new int[(r-l+1)];
            for (int n=0,i=l;i<=r;i++,n++){
                res[n]=i;
            }
            return res;
        }
    }
   int findleft(int[] nums,int target){

        int l=0,r=nums.length-1,m;
        while (l <= r) {
           m=l+(r-l)/2;
           if(nums[m]==target&&(m==0||nums[m-1]<target)) {
               return m;
           }
           else {
               if(nums[m]<target){
                   l=m+1;
               }
               else {
                   r=m-1;
               }
           }
        }
        return -1;
   }
   int findright(int[] nums,int target){

        int l=0,r=nums.length-1,m;
        while (l <= r) {
            m=l+(r-l)/2;
            if(nums[m]==target&&(m==nums.length-1||nums[m+1]>target)) {
                return m;
            }
            else {
                if(nums[m]>target){
                    r=m-1;
                }
                else {
                    l=m+1;
                }
            }
        }
        return -1;
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        ArrayList<List<Integer>> res=new ArrayList();
        Arrays.sort(candidates);
        dfs(0,target,0,candidates,res,new ArrayList<Integer>());
        return res;
    }
    void dfs(int sum,int target,int n,int[] candidates,ArrayList<List<Integer>> res,ArrayList<Integer> tmp){
        if(target<sum) {
            return;
        }
        if(target==sum){
            res.add(new ArrayList(tmp));
        }
        else {
            for(int i=n;i<candidates.length;i++){
                tmp.add(candidates[i]);
                dfs(sum + candidates[i], target, i, candidates, res, tmp);
                tmp.remove(tmp.size()-1);
            }
        }
    }

    public void nextPermutation(int[] nums) {
        int i=nums.length-1;
        while (i>0){
            if(nums[i]>nums[i-1]){
                break;
            }
            i--;
        }
        if (i == 0) {
            reverse(nums,0,nums.length-1);
        }
        else {
            i--;
            int j=nums.length-1;
            while (j>i){
                if(nums[j]>nums[i]) {
                    break;
                }
                j--;
            }
            swap(nums,i,j);
            reverse(nums,i+1,nums.length-1);
        }
    }
    void reverse(int[] nums,int s,int e){
        while (s<e){
            int tmp=nums[s];
            nums[s]=nums[e];
            nums[e]=tmp;
            s++;
            e--;
        }
    }
    void swap(int[] nums,int x,int y){
        int tmp=nums[x];
        nums[x]=nums[y];
        nums[y]=tmp;
    }

    public int firstMissingPositive(int[] nums) {
        int l=nums.length;
        for(int i=0;i<nums.length;i++){
            if(nums[i]>l||nums[i]<1){
                nums[i]=l+1;
            }
        }
        for(int i=0;i<nums.length;i++){
            while (nums[i]!=l+1&&nums[i]!=i+1){
                if(nums[i]==nums[nums[i]-1]){
                    nums[i]=l+1;
                    break;
                }
                swap(nums,i,nums[i]-1);
            }
        }
        int i=0;
        for(;i<l;i++){
            if(nums[i]==l+1){
                break;
            }
        }
        return i+1;
    }

    public int trap(int[] height) {
        if(height.length<3) {
            return 0;
        }
        int res=0;
        int l=height.length;
        for(int i=0;i<l;i++){
            int maxl=0;
            int maxr=0;
            int tmp=i;
            while (tmp<l){
                maxr=maxr>height[tmp]?maxr:height[tmp];
                tmp++;
            }
            tmp=i;
            while (tmp>=0){
                maxl=maxl>height[tmp]?maxl:height[tmp];
                tmp--;
            }
            res+=Math.min(maxl,maxr)-height[i];
        }
        return res;
    }
    int trap_dp(int[] height){
        if(height.length<3) {
            return 0;
        }
        int res=0;
        int l=height.length;
        int[] dpl=new int[l];
        int[] dpr=new int[l];
        dpl[0]=height[0];
        dpr[l-1]=height[l-1];
        for(int i=1;i<l;i++){
            dpl[i]=dpl[i-1]>height[i]?dpl[i-1]:height[i];
        }
        for(int i=l-2;i>=0;i--){
            dpr[i]=dpr[i+1]>height[i]?dpr[i+1]:height[i];
        }
        for(int i=0;i<l;i++){
            res+=Math.min(dpl[i],dpr[i])-height[i];
        }
        return res;
    }
    int trap_two_point(int[] height){
        int l=0,r=height.length-1;
        int res=0;
        int maxl=0;
        int maxr=0;
        while (l<r){
            if(height[l]<height[r]){
                if(height[l]<maxl){
                    res+=maxl-height[l];
                }
                else {
                    maxl=height[l];
                }
                l++;
            }
            else {
                if(height[r]<maxr){
                    res+=maxr-height[r];
                }
                else {
                    maxr=height[r];
                }
                r--;
            }
        }
        return res;
    }

    public int jump(int[] nums) {
        long t1=System.currentTimeMillis();
        int[] visited=new int[nums.length];
        dfs(0,visited,0,nums,nums.length-1);
        System.out.println(System.currentTimeMillis()-t1);
        return minPath;
    }
    public void dfs(int now,int[] visited,int path,int[] nums,int end){
        if(now==end){
            minPath=minPath<path?minPath:path;
            return;
        }
        if(visited[now]!=0&&visited[now]<path) {
            return;
        }
        for(int i=1;i<=nums[now]&&now+i<=end;i++){
            visited[now]=path;
            dfs(now+i,visited,path+1,nums,end);
        }

    }
    public int jump2(int[] nums){
        int ce=0,cf=0;
        int jump=0;
        for(int i=0;i<nums.length;i++){
            cf=Math.max(cf,i+nums[i]);
            if(i==ce){
                jump++;
                ce=cf;
            }
        }
        return jump;
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res=new ArrayList();
        dfs(nums,new HashSet<Integer>(),new ArrayList<Integer>(),res);
        return res;
    }
    void dfs(int[] nums,HashSet<Integer> visited,List<Integer> tmp,List<List<Integer>> res){
        if(tmp.size()==nums.length){
            res.add(new ArrayList<>(tmp));
            return;
        }
        for(int i=0;i<nums.length;i++){
            if(visited.contains(nums[i])){
                continue;
            }
            tmp.add(nums[i]);
            visited.add(nums[i]);
            dfs(nums,visited,tmp,res);
            tmp.remove(tmp.size()-1);
            visited.remove(nums[i]);
        }
    }

    public void rotate(int[][] matrix) {
        int l=matrix.length;
        int mid=l/2;
        int tmp1,tmp2,tmpi,tmpj,t;
        for(int i=0;i<mid;i++){
            for(int j=0;j<mid;j++){
                tmpi=j;
                tmpj=l-i-1;
                tmp1=matrix[i][j];
                for(int x=0;x<4;x++){
                    tmp2=matrix[tmpi][tmpj];
                    matrix[tmpi][tmpj]=tmp1;
                    tmp1=tmp2;
                    t=tmpi;
                    tmpi=tmpj;
                    tmpj=l-t-1;

                }
            }
        }
        if(l%2!=0){
            int i=mid;
            for(int j=0;j<mid;j++){
                tmpi=j;
                tmpj=l-i-1;
                tmp1=matrix[i][j];
                for(int x=0;x<4;x++){
                    tmp2=matrix[tmpi][tmpj];
                    matrix[tmpi][tmpj]=tmp1;
                    tmp1=tmp2;
                    t=tmpi;
                    tmpi=tmpj;
                    tmpj=l-t-1;

                }
            }
        }
    }

    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> res=new ArrayList();
        Map<String,List<String>> map=new HashMap();
        int[] cs;
        for(int i=0;i<strs.length;i++){
            String tmp="";
            cs=new int[26];
            for(Character c:strs[i].toCharArray()){
                cs[c-'a']++;
            }
            for(int j=0;j<26;j++){
                if(cs[j]!=0){
                    tmp+=cs[j]+"*"+j+",";
                }
            }
            if(!map.containsKey(tmp)){
                map.put(tmp,new ArrayList<String>());
            }
            map.get(tmp).add(strs[i]);
        }
        for(String key:map.keySet()){
            res.add(map.get(key));
        }
        return res;
    }

    public int maxSubArray(int[] nums) {
        int max=0;
        int min=nums[0];
        int tmp=0;
        for(int i=0;i<nums.length;i++){
            min=Math.max(min,nums[i]);
            if(tmp+nums[i]<0) {
                tmp=0;
            } else {
                tmp+=nums[i];
                max=max>tmp?max:tmp;
            }
        }
        if(max==0) {
            max=min;
        }
        return max;
    }

    public boolean canJump(int[] nums) {
        int max=0;
        for(int i=0;i<nums.length;i++){
            max=Math.max(max,i+nums[i]);
            if(i==max){
                break;
            }
        }
        return max>=nums.length-1;
    }

    public int minPathSum(int[][] grid) {
        int m=grid.length;
        int n=grid[0].length;
        dfs(grid,0,0,grid[0][0],m,n);
        return minPath;
    }
    void dfs(int[][] path,int x,int y,int current,int m,int n){
        if(x==m&&y==n) {
            minPath=Math.min(minPath,current);
            return;
        }
        if(x<m){
            dfs(path,x+1,y,current+path[x+1][y],m,n);
        }
        if(y<n){
            dfs(path,x,y+1,current+path[x][y+1],m,n);
        }
    }

    public int minPathSumWithDP(int[][] grid){
        int[][] dp=new int[grid.length][grid[0].length];
        dp[0][0]=grid[0][0];
        for(int i=1;i<grid.length;i++){
            dp[i][0]=dp[i-1][0]+grid[i][0];
        }
        for(int i=1;i<grid[0].length;i++){
            dp[0][i]=dp[0][i-1]+grid[0][i];
        }
        for(int i=1;i<grid.length;i++) {
            for(int j=1;j<grid[0].length;j++){
                dp[i][j]=Math.min(dp[i-1][j],dp[i][j-1])+grid[i][j];
            }
        }
        return dp[grid.length-1][grid[0].length-1];
    }

    public int climbStairs(int n) {
        int[] dp=new int[n];
        dp[0]=1;
        dp[1]=2;
        for(int i=2;i<n;i++){
            dp[i]=dp[i-1]+dp[i-2];
        }
        return dp[n-1];
    }

    public int minDistance(String word1, String word2) {
        int m=word1.length();
        int n=word2.length();
        int[][] dp=new int[m][n];
        dp[0][0]=word1.charAt(0)==word2.charAt(0)?0:1;
        for(int i=1;i<m;i++){
            if(word1.charAt(i)==word2.charAt(0)){
                dp[i][0]=i;
            }
            else {
                dp[i][0]=dp[i-1][0]+1;
            }
        }
        for(int i=1;i<n;i++){
            if(word1.charAt(0)==word2.charAt(i)){
                dp[0][i]=i;
            }
            else {
                dp[0][i]=dp[0][i-1]+1;
            }
        }

        for(int i=1;i<m;i++) {
            for (int j=1;j<n;j++){
                if(word1.charAt(i)==word2.charAt(j)){
                    dp[i][j]=dp[i-1][j-1];
                }
                else {
                    dp[i][j]=Math.min(dp[i-1][j-1]+1,Math.min(dp[i-1][j]+1,dp[i][j-1]+1));
                }
            }
        }
        return dp[m-1][n-1];
    }

    public void sortColors(int[] nums) {
        int a=0;
        int b=nums.length-1;
        for(int i=0;i<b;i++){
            while (nums[i]==2&&i<=b){
                swap(nums,b,i);
                b--;
            }
            while (nums[i]==0&&i>a){
                swap(nums,a,i);
                a++;
            }
        }
    }

    public String minWindow(String s, String t) {
        int a=0,b=0;
        int ma=0,mb=0;
        Map<Character,LinkedList<Integer>> map=new HashMap();
        int min=99999999;
        int t_min=-1;
        HashMap<Character,Integer> tmap=new HashMap();
        int count;
        int count1=0;
        for(Character c:t.toCharArray()){
            count1++;
            count=tmap.getOrDefault(c,0);
            tmap.put(c,count+1);
        }
        while (b<s.length()){
            if(tmap.keySet().contains(s.charAt(b))){
                LinkedList list=map.getOrDefault(s.charAt(b),new LinkedList<>());
                list.add(b);
                if(list.size()>tmap.get(s.charAt(b))){
                    list.remove(0);
                }
                map.put(s.charAt(b),list);
                int c1=0;
                for(Character c:map.keySet()){
                    c1+=map.get(c).size();
                }
                if(count1==c1) {
                    int tmp=9999999;
                    for (Character c : tmap.keySet()) {
                        tmp = Math.min(tmp, map.get(c).get(0));
                    }
                    t_min=tmp;
                }
            }
            while (t_min>=a){
                if((b-a)<min){
                    min=(b-a);
                    ma=a;
                    mb=b;
                }
                a++;
            }
            b++;
        }
        if(min!=99999999) {
            return s.substring(ma,mb+1);
        } else {
            return "";
        }
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res=new ArrayList();
        dfs(0,nums,res,new ArrayList<>());
        return res;
    }
    public void dfs(int i,int[] nums,List<List<Integer>> res,List<Integer> tmp){
        if(i==nums.length){
            res.add(new ArrayList<>(tmp));
            return;
        }
        tmp.add(nums[i]);
        dfs(i+1,nums,res,tmp);
        tmp.remove(tmp.size()-1);
        dfs(i+1,nums,res,tmp);
    }

    public boolean exist(char[][] board, String word) {
        boolean[][] visited=new boolean[board.length][board[0].length];
        visited[0][0]=true;
        for(int i=0;i<board.length&&!res;i++) {
            for(int j=0;j<board[0].length&&!res;j++){
                visited[i][j]=true;
                dfs(board,visited,i,j,word,board[i][j]+"",0);
                visited[i][j]=false;
            }
        }
        return res;
    }
    void dfs(char[][] board,boolean[][] visited,int x,int y,String word,String tmp,int index){
        if(!res&&tmp.length()<word.length()&&tmp.charAt(index)==word.charAt(index)){
            for(int i=0;i<4;i++){
                int xx=x+mx[i];
                int yy=y+my[i];
                if(xx>=0&&xx<board.length&&yy>=0&&yy<board[0].length&&!visited[xx][yy]){
                    visited[xx][yy]=true;
                    dfs(board,visited,xx,yy,word,tmp+board[xx][yy],index+1);
                    visited[xx][yy]=false;
                }
            }
        }
        if(tmp.equals(word)){
            res=true;
            return;
        }
    }

    public int largestRectangleArea(int[] heights) {
        int[] minl=new int[heights.length];
        int[] minr=new int[heights.length];
        int max=0;

        for(int i=0;i<minr.length;i++){
            minl[i]=-1;
            minr[i]=heights.length;
        }
        for (int i = 1; i < heights.length; i++) {
            int p = i - 1;

            while (p >= 0 && heights[p] >= heights[i]) {
                p = minl[p];
            }
            minl[i] = p;
        }

        for (int i = heights.length - 2; i >= 0; i--) {
            int p = i + 1;

            while (p < heights.length && heights[p] >= heights[i]) {
                p = minr[p];
            }
            minr[i] = p;
        }
        for(int i=0;i<heights.length;i++){
            int tmp=(minr[i]-minl[i]-1)*heights[i];
            max=Math.max(max,tmp);
        }
        return max;
    }

    public int maximalRectangle(char[][] matrix) {
        int[][] left=new int[matrix.length][matrix[0].length];
        int[][] up=new int[matrix.length][matrix[0].length];
        int max=0;
        for(int i=0;i<matrix.length;i++) {
            for (int j=0;j<matrix[0].length;j++){
                if(matrix[i][j]=='0'){
                    left[i][j]=0;
                    up[i][j]=0;
                }
                else {
                    if(j>0) {
                        left[i][j] = left[i][j - 1] + 1;
                    } else {
                        left[i][j] = matrix[i][j] == '1' ? 1 : 0;
                    }
                    if(i>0) {
                        up[i][j] = up[i - 1][j] + 1;
                    } else {
                        up[i][j] = matrix[i][j] == '1' ? 1 : 0;
                    }
                }
            }
        }
        for(int i=0;i<matrix.length;i++) {
            for(int j=0;j<matrix[0].length;j++){
                if(left[i][j]==0||up[i][j]==0) {
                    continue;
                }
                int ii=j;
                int y=up[i][j];
                while (ii>=0&&left[i][ii]>0){
                    y=Math.min(y,up[i][ii]);
                    max=Math.max(max,(j-ii+1)*y);
                    ii--;
                }
            }
        }
        return max;
    }
    public int maximalRectangleWithDP(char[][] matrix){
        int res=0;
        int m=matrix.length;
        int n=matrix[0].length;
        int[] l=new int[n];
        int[] r=new int[n];
        int[] h=new int[n];
        for(int i=0;i<n;i++){
            r[i]=n;
        }
        for(int i=0;i<m;i++){
            int cur_left=0, cur_right=n;
            for(int j=0;j<n;j++){
                if(matrix[i][j]=='0'){
                    h[j]=0;
                }
                else {
                    h[j]++;
                }
            }
            for(int j=0;j<n;j++){
                if(matrix[i][j]=='1'){
                    l[j]=Math.max(cur_left,l[j]);
                }
                else {
                    l[j]=0;
                    cur_left=j+1;
                }
            }
            for(int j=n-1;j>=0;j--){
                if(matrix[i][j]=='1'){
                    r[j]=Math.min(cur_right,r[j]);
                }
                else {
                    r[j]=n;
                    cur_right=j;
                }
            }
            for(int j=0;j<n;j++){
                res=Math.max(res,(r[j]-l[j])*h[j]);
            }
        }
        return res;
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res=new ArrayList();
        if(root==null) {
            return res;
        }
        Stack<TreeNode> stack=new Stack();
        TreeNode tmp=root;
        do {
            while (tmp != null) {
                stack.push(tmp);
                tmp = tmp.left;
            }
            tmp=stack.pop();
            res.add(tmp.val);
            tmp=tmp.right;
        } while (!stack.isEmpty());
        return res;
    }

    public int numTrees(int n) {
        int[]dp=new int[n+1];
        dp[0]=1;
        dp[1]=1;
        for(int i=2;i<=n;i++){
            for(int j=0;j<i;j++){
                dp[i]+=dp[j]*dp[i-j-1];
            }
        }
        return dp[n];
    }

    public boolean isValidBST(TreeNode root) {
        if(root==null) {
            return true;
        }
        Stack<TreeNode> stack=new Stack();
        TreeNode tmp=root;
        int order=0;
        boolean flag=false;
        while (!stack.isEmpty()||tmp!=null){
            if(tmp!=null) {
                stack.push(tmp);
                tmp = tmp.left;
            }
            else {
                tmp=stack.pop();
                if(flag&&tmp.val<=order){
                    return false;
                }
                order=tmp.val;
                flag=true;
                tmp=tmp.right;
            }
        }
        return true;
    }

    public boolean isSymmetric(TreeNode root) {
        if(root==null) {
            return true;
        }
        visit(root.left,root.right);
        return res;
    }
    void visit(TreeNode node1,TreeNode node2){
        if(node1!=null&&node2!=null){
            if (node1.val!=node2.val) {
                res = false;
                return;
            }
            else {
                visit(node1.left,node2.right);
                visit(node1.right,node2.left);
            }
        }
        else if(node1==null&&node2!=null){
            res = false;
            return;
        }
        else if(node1!=null&&node2==null){
            res = false;
            return;
        }
    }

    public int maxDepth(TreeNode root) {
        dfs(root,0);
        return max;
    }
    void dfs(TreeNode node,int path){
        if(node==null){
            max=Math.max(max,path);
            return;
        }
        else {
            dfs(node.left,path+1);
            dfs(node.right,path+1);
        }
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return help(0,0,inorder.length-1,preorder,inorder);
    }

    public TreeNode help(int preStart, int inStart, int inEnd, int[] preorder, int[] inorder) {
        if (preStart > preorder.length - 1 || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[preStart]);
        int inIndex = 0; // Index of current root in inorder
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == root.val) {
                inIndex = i;
            }
        }
        root.left = help(preStart + 1, inStart, inIndex - 1, preorder, inorder);
        root.right = help(preStart + inIndex - inStart + 1, inIndex + 1, inEnd, preorder, inorder);
        return root;
    }

    public String restoreString(String s, int[] indices) {
        char[] ss=new char[s.length()];
        for(int i=0;i<indices.length;i++){
            ss[indices[i]]=s.charAt(i);
        }
        return new String(ss);
    }

    public int minFlips(String target) {
        int res=0;
        char now='0';
        for(int i=0;i<target.length();i++){
            if(now!=target.charAt(i)){
                res++;
                now=target.charAt(i);
            }
        }
        return res;
    }

    public int countPairs(TreeNode root, int distance) {
        dfs1(root,distance);
        return res1;
    }
    List<Integer> dfs1(TreeNode node,int distance){
        List<Integer> tmp=new ArrayList();
        if(node==null) {
            return tmp;
        }
        List<Integer> left=dfs1(node.left,distance);
        List<Integer> right=dfs1(node.right,distance);
        if(left.size()==0&&right.size()==0){
            tmp.add(1);
            return tmp;
        }
        for(int i=0;i<left.size();i++){
            for(int j=0;j<right.size();j++){
                if(left.get(i)+right.get(j)<=distance){
                    res1++;
                }
            }
        }
        for(int i:left){
            tmp.add(i+1);
        }
        for(int i:right){
            tmp.add(i+1);
        }
        return tmp;
    }
    int[] dfs2(TreeNode node,int distance){
        int[] tmp=new int[11];
        if(node==null){
            return tmp;
        }
        if(node.left==null&&node.right==null){
            tmp[1]=1;
            return tmp;
        }
        int[] left=dfs2(node.left,distance);
        int[] right=dfs2(node.right,distance);
        for(int i=0;i<11;i++) {
            for(int j=0;j<11;j++){
                if(i+j<=distance){
                    res1+=left[i]*right[j];
                }
            }
        }
        for(int i=1;i<11;i++){
            tmp[i]=left[i-1]+right[i-1];
        }
        return tmp;
    }

    public void flatten(TreeNode root) {
        Stack<TreeNode> stack=new Stack();
        TreeNode tmp;
        TreeNode res=new TreeNode();
        TreeNode previous=res;
        stack.push(root);
        while (!stack.isEmpty()){
            tmp=stack.pop();
            previous.right=tmp;
            previous.left=null;
            previous=tmp;
            if(tmp.right!=null) {
                stack.push(tmp.right);
            }
            if(tmp.left!=null) {
                stack.push(tmp.left);
            }
        }
    }

    public int maxPathSum(TreeNode root) {
        help(root);
        return res2;
    }
    int help(TreeNode node){
        if(node==null) {
            return 0;
        }
        int res=node.val;
        int left=help(node.left);
        int right=help(node.right);
        if(left<0&&right<0){
            res2=Math.max(res,res2);
        }
        else{
            res2=Math.max(res2,res+left+right);
            res+=Math.max(left,right);
            res2=Math.max(res,res2);
        }
        return res;
    }

    public int longestConsecutive(int[] nums) {
        int res=0;
        Set<Integer> set=new HashSet();
        for(Integer n:nums){
            set.add(n);
        }
        for(Integer n:set){
            if(!set.contains(n-1)){
                int currentres=1;
                int current=n+1;
                while (set.contains(current)){
                    currentres++;
                    current++;
                }
                res=Math.max(currentres,res);
            }
        }
        return res;
    }

    public Node copyRandomList(Node head) {
        Node res;
        if(head==null) {
            return null;
        }
        Node node=head;
        while (node!=null){
            Node tmp=new Node(node.val);
            tmp.next=node.next;
            tmp.random=node.random;
            node.next=tmp;
            node=tmp.next;
        }
        res=head.next;
        node=head;
        Node next;
        while(node!=null){
            next=node.next;
            if(node.random!=null){
                next.random=node.random.next;
            }
            node=node.next.next;
        }
        node=head;
        while (node!=null){
            next=node.next;
            if(node.next.next!=null){
                node.next=next.next;
                node=next.next;
                next.next=node.next;
            }
            else{
                node.next=null;
                node=null;
            }
        }
        return res;
    }

    public boolean wordBreak(String s, List<String> wordDict) {
        HashSet set=new HashSet(wordDict);
        boolean[] dp=new boolean[s.length()+1];
        dp[0]=true;
        for(int i=1;i<=s.length();i++){
            for(int j=0;j<i;j++){
                if(dp[j]&&set.contains(s.substring(j,i))){
                    dp[i]=true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }
    void dfs(String s,String tmp,List<String> wordDict){
        if(!s.substring(0,tmp.length()).equals(tmp)){
            return;
        }
        else if(!res){
            for(int i=0;i<wordDict.size();i++){
                dfs(s,tmp+wordDict.get(i),wordDict);
            }
        }
    }

    public boolean hasCycle(ListNode head) {
        if(head==null||head.next==null) {
            return false;
        }
        ListNode slow=head.next;
        ListNode fast=head.next.next;
        while (fast!=null){
            if(slow==fast) {
                return true;
            }
            if(fast.next==null) {
                return false;
            }
            slow=slow.next;
            fast=fast.next.next;
        }
        return false;
    }

    public int maxProduct(int[] nums) {
        int res=Integer.MIN_VALUE;
        int[] dp1=new int[nums.length];
        int[] dp2=new int[nums.length];
        dp1[0]=nums[0];
        dp2[0]=nums[0];
        for(int i=1;i<nums.length;i++){
            if(nums[i]==0){
                dp1[i]=0;
                dp2[i]=0;
            }
            else if(nums[i]>0){
                dp1[i]=(dp1[i-1]>0?dp1[i-1]:1)*nums[i];
                dp2[i]=(dp2[i-1]<=0?dp2[i-1]:1)*nums[i];
            }
            else {
                dp1[i]=(dp2[i-1]<=0?dp2[i-1]:1)*nums[i];
                dp2[i]=(dp1[i-1]>0?dp1[i-1]:1)*nums[i];
            }

        }
        return res;
    }

    public int numIslands(char[][] grid) {
        int m=grid.length;
        int n=grid[0].length;
        boolean[][] visited=new boolean[m][n];
        for(int i=0;i<m;i++) {
            for (int j=0;j<n;j++){
                if(grid[i][j]=='1'&&!visited[i][j]){
                    dfs(grid,visited,i,j,m,n,false);
                }
            }
        }
        return res1;
    }
    void dfs(char[][] grid,boolean[][] visited,int x,int y,int m,int n,boolean checked){
        if(!checked){
            res1++;
            checked=true;
        }
        for(int i=0;i<4;i++){
            int xx=x+mx[i];
            int yy=y+my[i];
            if(xx>=0&&xx<m&&yy>=0&&yy<n&&!visited[xx][yy]&&grid[xx][yy]=='1'){
                visited[xx][yy]=true;
                dfs(grid,visited,xx,yy,m,n,true);
            }
        }
    }

    public ListNode reverseList(ListNode head) {
        if(head==null) {
            return head;
        }
        ListNode pervious=head;
        ListNode res=head.next;
        if(res==null) {
            return head;
        }
        head.next=null;
        ListNode next=res.next;
        while (next!=null){
            res.next=pervious;
            pervious=res;
            res=next;
            next=next.next;
        }
        res.next=pervious;
        return res;
    }

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int[] indegree=new int[numCourses];
        ArrayList<Integer>[] g=new ArrayList[numCourses];
        Queue<Integer> queue=new LinkedList();
        for(int i=0;i<numCourses;i++) {
            g[i]=new ArrayList();
        }
            for(int i=0;i<prerequisites.length;i++){
                g[prerequisites[i][0]].add(prerequisites[i][1]);
                indegree[prerequisites[i][1]]++;
        }
        for(int i=0;i<numCourses;i++){
            if(indegree[i]==0) {
                queue.add(i);
            }
        }
        while (!queue.isEmpty()){
            int now=queue.remove();
            for(int i:g[now]){
                indegree[i]--;
                if(indegree[i]==0){
                    queue.add(i);
                }
            }
        }
        for(int i=0;i<numCourses;i++){
            if(indegree[i]>0) {
                return false;
            }
        }
        return true;
    }
    boolean check(int n,ArrayList<Integer>[] g){
        boolean res=true;
        for(int i=0;i<g.length;i++){
            if(g[i]==null) {
                continue;
            }
            for(int ii:g[i]){
                if(ii==n) {
                    return false;
                }
            }
        }
        return res;
    }

    public int findKthLargest(int[] nums, int k) {
//        ArrayList<Integer> list=new ArrayList(k);
//        for (int i:nums){
//            if(list.size()<k) {
//                list.add(i);
//                if(list.size()==k)
//                    list.sort((a,b)->a-b);
//            }
//            else {
//                if(i>list.get(0)){
//                    list.remove(0);
//                    list.add(i);
//                    list.sort((a,b)->a-b);
//                }
//            }
//        }
//        return list.get(0);
        return help(0,nums.length-1,nums,k);
    }

    int help(int m,int n,int[] nums,int k){
        int tmp=nums[m];
        int mm=m;
        int nn=n;
        while (m<n){
            while (m<n&&nums[n]<=tmp) {
                n--;
            }
            nums[m]=nums[n];
            while (m<n&&nums[m]>=tmp) {
                m++;
            }
            nums[n]=nums[m];
        }
        nums[m]=tmp;
        if(m==k-1) {
            return nums[m];
        }
        if(m<k) {
            return help(m + 1, nn, nums, k);
        }
        else {
            return help(mm,n-1,nums,k);
        }
    }

    public int test(List<int[]> pairs){
        pairs.sort((a,b)->{
            if(a[0]!=b[0]) {
                return a[0]-b[0];
            } else {
                return b[1]-a[1];
            }
        });
        Stack<Integer> stack=new Stack();
        int res=0;
        for (int[] pair:pairs){
            if(stack.isEmpty()){
                stack.push(pair[1]);
            }
            else {
                if(pair[0]<stack.peek()){
                    stack.push(pair[1]);
                }
                else {
                    res=Math.max(res,stack.size());
                    while (!stack.isEmpty()&&stack.peek()<=pair[0]) {
                        stack.pop();
                    }
                    stack.push(pair[1]);
                }
            }
        }
        res=Math.max(res,stack.size());
        return res;
    }

    public TreeNode invertTree(TreeNode root) {
        if(root==null) {
            return null;
        }
        TreeNode left=invertTree(root.left);
        TreeNode right=invertTree(root.right);
        root.left=right;
        root.right=left;
        return root;
    }

    public int kthSmallest(TreeNode root, int k) {
        Stack<TreeNode> stack=new Stack();
        TreeNode node=root;
        int index=0;
        int res=-1;
        while (!stack.isEmpty()||node!=null){
            if(node!=null){
                stack.push(node);
                node=node.left;
            }
            else {
                index++;
                node=stack.pop();
                if(index==k){
                    res=node.val;
                    break;
                }
                node=node.right;
            }
        }

        return res;
    }

    public boolean isPalindrome(ListNode head) {
        int num = 0;
        int index = 0;
        ListNode node = head;
        while (node != null) {
            node = node.next;
            num++;
        }
        node = head;
       int half=num/2;
       while (index<half){
           node=node.next;
           index++;
       }
       if(num%2==1){
           node=node.next;
       }
        ListNode mid =node;
       mid=reverseList(mid);
       node=head;
       index=0;
       while (index<half){
           if (node.val!=mid.val) {
               return false;
           }
           index++;
       }
        return true;
    }

    public int[] productExceptSelf(int[] nums) {
        int n=1;
        int count=0;
        int[] res=new int[nums.length];
        for(int i=0;i<nums.length;i++){
            if(nums[i]==0){
                count++;
                continue;
            }
            n*=nums[i];
        }
        for(int i=0;i<res.length;i++){
            if (nums[i]==0) {
                res[i]=count==1?n:0;
            } else {
                res[i]=count==0?n/nums[i]:0;
            }
        }
        return res;
    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        int[] res=new int[nums.length-k+1];
        QueueWithMax queue=new QueueWithMax();
        for(int i=0;i<k;i++){
            queue.offer(nums[i]);
        }
        for(int i=k;i<nums.length;i++){
            res[i-k]=queue.getMax();
            queue.remove();
            queue.offer(nums[i]);
        }
        res[nums.length-k]=queue.getMax();
        return res;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        int m=matrix.length;
        int n=matrix[0].length;
        int x=m-1;
        int y=0;
        while (x>=0&&y<n){
            if(matrix[x][y]==target) {
                return true;
            } else if(matrix[x][y]>target) {
                x--;
            } else {
                y++;
            }
        }
        return false;
    }

    public int minMeetingRooms(Interval[] intervals) {
        Arrays.sort(intervals, new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                if(o1.start!=o2.start){
                    return o1.start-o2.start;
                }
                else {
                    return o2.end-o1.end;
                }
            }
        });
        int max=0;
        Stack<Integer> stack=new Stack();
        for(Interval interval:intervals){
            if(stack.isEmpty()){
                stack.push(interval.end);
            }
            else {
                if(stack.peek()<interval.start) {
                    max=Math.max(max,stack.size());
                    while (!stack.isEmpty()&&stack.peek() < interval.start) {
                        stack.pop();
                    }
                    stack.push(interval.end);
                }
                else {
                    stack.push(interval.end);
                }
            }
        }
        return Math.max(max,stack.size());
    }

    public int numSquares(int n) {
       int[] dp=new int[n+1];
       dp[0]=0;
       dp[1]=1;
       for(int i=2;i<=n;i++){
           dp[i]=dp[i-1]+1;
           for(int j=2;j*j<=i;j++){
               dp[i]=Math.min(dp[i-j*j]+1,dp[i]);
           }
       }
        return dp[n];
    }

    public void moveZeroes(int[] nums) {
        for(int i=1;i<nums.length;i++){
            if(nums[i]==0) {
                continue;
            }
            ;
            int j=i-1;
            while (j>=0&&nums[j]==0) {
                j--;
            }
            if(j!=i-1){
                nums[j+1]=nums[i];
                nums[i]=0;
            }
        }
    }

    public int findDuplicate(int[] nums) {
        if(nums.length==0) {
            return -1;
        }
        int slow=nums[0];
        int fast=nums[0];
        do{
            slow=nums[slow];
            fast=nums[nums[fast]];
        }while (slow!=fast);
        slow=nums[0];
        while (slow!=fast){
            slow=nums[slow];
            fast=nums[fast];
        }
        return slow;
    }

    public int lengthOfLIS(int[] nums) {
        if(nums.length==0) {
            return 0;
        }
        int res=1;
        int[] dp=new int[nums.length];
        Arrays.fill(dp,1);
        for(int i=nums.length-1;i>=0;i--){
            int j=i+1;
            while (j<nums.length){
                if(nums[i]<nums[j]) {
                    dp[i]=Math.max(dp[i],dp[j]+1);
                }
                j++;
            }
        }
        for(int i:dp) {
            res=Math.max(res,i);
        }
        return res;
    }
    public int lengthOfLIS_2(int[] nums){
        List<Integer> list=new ArrayList();
        for(int i=0;i<nums.length;i++){
            if(list.isEmpty()||nums[i]>list.get(list.size()-1)){
                list.add(nums[i]);
            }
            else {
                int j=list.size()-2;
                while (j>=0&&list.get(j)>=nums[i]) {
                    j--;
                }
                list.remove(j+1);
                list.add(j+1,nums[i]);
            }
        }
        return list.size();
    }

    public boolean increasingTriplet(int[] nums) {
        boolean res=false;
        int a=Integer.MAX_VALUE;
        int b=Integer.MAX_VALUE;
        for (int i=0;i<nums.length;i++){
            if (nums[i]<=a) {
                a=nums[i];
            } else if(nums[i]<=b) {
                b=nums[i];
            } else {
                return true;
            }
        }
        return res;
    }

    public List<String> removeInvalidParentheses(String s) {
        HashSet visited=new HashSet();
        Queue<String> queue=new LinkedList();
        List<String> res=new ArrayList();
        queue.offer(s);
        visited.add(s);
        int l=-1;
        while (!queue.isEmpty()){
            String tmp=queue.poll();
            if(l>=0&&tmp.length()<l) {
                return res;
            }
            else {
                if (check(tmp)) {
                    l = tmp.length();
                    res.add(tmp);
                }
                for(int i=0;i<tmp.length();i++){
                    String ss=tmp.substring(0,i)+tmp.substring(i+1,tmp.length());
                    if(!visited.contains(ss)) {
                        visited.add(ss);
                        queue.offer(tmp.substring(0, i) + tmp.substring(i + 1, tmp.length()));
                    }
                }
            }
        }
        return res;
    }
    boolean check(String s){
//        Stack<Character> stack=new Stack();
//        for(Character c:s.toCharArray()){
//            if(c=='('||c=='['){
//                stack.push(c);
//            }
//            else {
//                if(!stack.isEmpty()&&c==')'&&stack.peek()=='('){
//                    stack.pop();
//                }
//                else if(!stack.isEmpty()&&c==']'&&stack.peek()=='['){
//                    stack.pop();
//                }
//                else if(c!=')'&&c!=']')
//                    continue;
//                else return false;
//            }
//        }
//        return stack.isEmpty();
        int count1=0,count2=0;
        for(Character c:s.toCharArray()){
            if(c=='('){
                count1++;
            }
            else if(c=='['){
                count2++;
            }
            else if(c==')'){
                count1--;
                if(count1<0) {
                    return false;
                }
            }
            else if(c==']'){
                count2--;
                if(count2<0) {
                    return false;
                }
            }
        }
        return count1==0&&count2==0;
    }

    public int coinChange(int[] coins, int amount) {
        return coinChange(coins,amount,new int[amount]);
    }
    void dfs(int[] coins,int tmp,int tmpcount,int amount){
        if(tmpcount==amount){
            res1=Math.max(res1,tmp);
        }
        else if(tmpcount>amount) {
            return;
        } else {
            for(int coin:coins){
                dfs(coins,tmp+1,tmpcount+coin,amount);
            }
        }
    }
    int coinChange(int[] coins,int amount,int[] dp){
        if(amount==0) {
            return 0;
        } else if(amount<0) {
            return -1;
        } else if(dp[amount-1]!=0) {
            return dp[amount-1];
        }
        int min=Integer.MAX_VALUE;
        for(int coin:coins){
            int res=coinChange(coins,amount-coin,dp);
            min=res>=0?Math.min(min,res+1):min;
        }

        dp[amount-1]=min!=Integer.MAX_VALUE?min:-1;
        return dp[amount-1];
    }
    public int coinChange_2(int[] coins, int amount){
        Arrays.sort(coins);
        int[] dp=new int[amount+1];
        for(int i=1;i<dp.length;i++){
            int min=Integer.MAX_VALUE;
            for(int j=0;j<coins.length;j++){
                if(coins[j]>i) {
                    break;
                }
                else {
                    int res=dp[j-coins[i]];
                    min=res>=0?Math.min(min,res+1):min;
                }
            }
            dp[i]=min==Integer.MAX_VALUE?-1:min;
        }
        return dp[amount];
    }

    public int rob(TreeNode root) {
        return rob(root,false);
    }
    static HashMap<TreeNode,Integer> dp=new HashMap();
    static {
        dp.put(null,0);
    }
    public int rob_1(TreeNode root) {

        if(root==null) {
            return 0;
        } else {
            int left=0;
            int right=0;
            if(root.left!=null){
                if(!dp.containsKey(root.left)){
                    dp.put(root.left,rob_1(root.left));
                }
                if(!dp.containsKey(root.left.left)){
                    dp.put(root.left.left,rob_1(root.left.left));
                }
                if(!dp.containsKey(root.left.right)){
                    dp.put(root.left.right,rob_1(root.left.right));
                }
                left=dp.get(root.left.left)+dp.get(root.left.right);
            }
            if(root.right!=null){
                if(!dp.containsKey(root.right)){
                    dp.put(root.right,rob_1(root.right));
                }
                if(!dp.containsKey(root.right.left)){
                    dp.put(root.right.left,rob_1(root.right.left));
                }
                if(!dp.containsKey(root.right.right)){
                    dp.put(root.right.left,rob_1(root.right.right));
                }
                right=dp.get(root.right.left)+dp.get(root.right.right);
            }
            return Math.max(left+right+root.val,dp.get(root.left)+dp.get(root.right));
        }
    }
    int rob(TreeNode node,boolean robed){
       if(node==null) {
           return 0;
       } else {
           if(robed){
               return rob(node.left,false)+rob(node.right,false);
           }
           else {
               int r1=rob(node.left,false)+rob(node.right,false);
               int r2=rob(node.left,true)+rob(node.right,true)+node.val;
               return Math.max(r1,r2);
           }
       }
    }
    int[] robSub(TreeNode root){
        if(root==null){
            return new int[2];
        }
        else {
            int[] res=new int[2];
            int[] left=robSub(root.left);
            int[] right=robSub(root.right);
            res[0]=Math.max(left[0],left[1])+Math.max(right[0],right[1]);
            res[1]=left[0]+right[0]+root.val;
            return res;
        }
    }

    public int[] countBits(int num) {
        int[] res=new int[num+1];
        for(int i=1;i<=num;i++){
            res[i]=res[i>>1]+i&1;
        }
        return res;
    }

    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer,Integer> map=new HashMap();
        for(int i=0;i<nums.length;i++){
            int now=map.getOrDefault(nums[i],0);
            map.put(nums[i],now+1);
        }
        PriorityQueue<Tuple> queue=new PriorityQueue(new Comparator<Tuple>() {
            @Override
            public int compare(Tuple o1, Tuple o2) {
                return o1.b-o2.b;
            }
        });
        for(Integer i:map.keySet()){
            if(queue.size()<k) {
                queue.offer(new Tuple(i,map.get(i)));
            } else {
                if(queue.peek().b<map.get(i)) {
                    queue.offer(new Tuple(i,map.get(i)));
                    queue.poll();
                }
            }
        }
        int[] res=new int[k];
        for(int i=0;i<k;i++){
            res[i]=queue.peek().a;
            queue.poll();
        }
        return res;
    }
    static class Tuple{
        int a;
        int b;
        public Tuple(int a,int b){
            this.a=a;
            this.b=b;
        }
    }
    public int[] topKFrequent_1(int[] nums, int k) {
        Map<Integer,Integer> map=new HashMap();
        for(int i=0;i<nums.length;i++){
            int now=map.getOrDefault(nums[i],0);
            map.put(nums[i],now+1);
        }
        int[] f=new int[map.keySet().size()];
        int index=0;
        for(int i:map.keySet()){
            f[index]=i;
            index++;
        }
        help(f,0,f.length-1,k,map);
        int[] res=new int[k];
        for(int i=0;i<k;i++){
            res[i]=f[i];
        }
        return res;
    }
    int quickSelect(int[] nums,int l,int r,int k,Map<Integer,Integer> map){
        if(l==k) {
            return l;
        }
        int tmp=nums[l];
        while (l<r){
            while (l<r&&map.get(nums[r])<=map.get(tmp)) {
                r--;
            }
            nums[l]=nums[r];
            while (l<r&&map.get(nums[l])>=map.get(tmp)) {
                l++;
            }
            nums[r]=nums[l];
        }
        nums[l]=tmp;
        return l;
    }
    void help(int[] nums,int l,int r,int k,Map<Integer,Integer> map){
        int mid=quickSelect(nums,l,r,k,map);
        if(mid==k) {
            return;
        } else {
            if(mid<k){
                help(nums,mid+1,r,k,map);
            }
            else {
                help(nums,l,mid-1,k,map);

            }
        }
    }

    public String decodeString(String s) {
        Stack<Character> stack=new Stack();
        StringBuilder res=new StringBuilder();
        StringBuilder tmps=new StringBuilder();
        StringBuilder tmpi=new StringBuilder();
        for(Character c:s.toCharArray()){
            if(c==']'){
                tmps=new StringBuilder();
                tmpi=new StringBuilder();
                while (stack.peek()!='['){
                    tmps.append(stack.pop());
                }
                stack.pop();
                while (!stack.isEmpty()&&stack.peek()>='0'&&stack.peek()<='9'){
                    tmpi.append(stack.pop());
                }
                int n=Integer.parseInt(tmpi.reverse().toString());
                String substring=tmps.reverse().toString();
                if(stack.isEmpty()){
                    for(int i=0;i<n;i++) {
                        res.append(substring);
                    }
                }
                else {
                    for(int i=0;i<n;i++) {
                        for (Character tmpc : substring.toCharArray()) {
                            stack.push(tmpc);
                        }
                    }
                }
            }
            else if(c=='['||(c>='0'&&c<='9')){
                stack.push(c);
            }
            else {
                if(stack.isEmpty()){
                    res.append(c);
                }
                else {
                    stack.push(c);
                }
            }
        }
        return res.toString();
    }

    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if(o2[0]!=o1[0]) {
                    return o2[0]-o1[0];
                } else {
                    return o1[1]-o2[1];
                }
            }
        });
        List<int[]> list=new LinkedList();
        for(int i=0;i<people.length;i++){
            list.add(people[i][1],people[i]);
        }
        int[][] res=list.toArray(new int[list.size()][2]);
        return res;
    }

    public boolean canPartition(int[] nums) {
        int total=0;
        for(int n:nums){
            total+=n;
        }
        if(total%2==1){
            return false;
        }
        int half=total/2;
       boolean[] dp=new boolean[half+1];
       dp[0] = true;
       for(int i=1;i<=nums.length;i++){
           for(int j = half; j >= nums[i-1]; j--){
               dp[j]=dp[j]||dp[j-nums[i-1]];
           }
       }
       return dp[half];
    }
    public boolean canPartition_1(int[] nums){
        int total=0;
        for(int n:nums){
            total+=n;
        }
        if(total%2==1){
            return false;
        }
        int half=total/2;
        boolean[][] dp=new boolean[nums.length+1][half+1];
        for(int i=0;i<nums.length+1;i++){
            dp[i][0]=true;
        }
        for(int i=1;i<=nums.length;i++){
            for(int j=1;j<=half;j++){
                dp[i][j]=dp[i-1][j];
                if(j>=nums[i]){
                    dp[i][j]=dp[i][j]||dp[i-1][j-nums[i]];
                }
            }
        }
        return dp[nums.length][half];
    }

    public int pathSum(TreeNode root, int sum) {
        return res1;
    }

    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        int l1=s.length();
        int l2=p.length();
        if(l1<l2){
            return res;
        }
        int[] ss=new int[26];
        int[] pp=new int[26];
        for(Character c:p.toCharArray()){
            pp[c-'a']++;
        }
        int l=0,r=0;
        while (r<l1&&l<l1-l2){
            if((r-l)==l2){
                res.add(l);
                l++;
            }
            else {

            }
        }
        return res;
    }

}
