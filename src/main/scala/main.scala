import java.io.{BufferedWriter, File, FileInputStream, FileWriter}
import java.util.Random

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

object main extends App {
  val config=new SparkConf
  config.setAppName("test")
  config.setMaster("local[4]")
  var count=0;
  val sc=new SparkContext(config)
  val rdd=sc.textFile("D:\\\\test.txt",10)
  val reduced=rdd.map(c=>{
    val s=c.split(",")
    ((s(0),s(1)),1)
  })
  val grouped=reduced.reduceByKey(_+_).groupBy(_._1._1)
  val subjects=reduced.map(_._1._1).distinct().collect()
//  val sorted=grouped.mapPartitions(c=>{
//    c.toList.sortBy(_._2).reverse.take(2).toIterator
//  })
//  println(grouped.getNumPartitions)
//  for(sub <- subjects){
//    println(grouped.filter(_._1._1==sub).sortBy(_._2,false).take(2).toBuffer)
//  }
  val sorted=grouped.mapValues(_.toList.sortBy(_._2).reverse.take(5))
  println(sorted.collect.toBuffer)
//  val file=new File("D:\\test.txt")
//  try {
//    val random=new Random
//    var num1:Int=0
//    var num2:Int=0
//    val bw = new BufferedWriter(new FileWriter(file))
//    for(i<- 0 to 1000){
//      num1=random.nextInt(10)
//      num2=random.nextInt(100)
//      bw.write(s"subject${num1},teacher${num2}\n")
//    }
//    bw.flush()
//  }
//  catch {
//    case e:Exception =>{}
//  }
}
