/**
  * Created by hadoop on 11/30/16.
  */

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.graphx._
import org.apache.spark.graphx.{Graph, VertexRDD}
import org.apache.spark.graphx.util.GraphGenerators

import scala.util.Random
import scala.collection.mutable.HashMap
import scala.io._
import java.io._

import scala.io.Source
import org.apache.spark.rdd.RDD
import java.io.StringReader

import com.opencsv.CSVReader

import scala.collection.mutable.ArrayBuffer

object knnPredict {
  val sc = new SparkContext(new SparkConf().setMaster("local").set("spark.driver.allowMultipleContexts", "true").setAppName("knnPredict"))
  def main(args: Array[String]) {

    val input =  sc.textFile("./trainning.csv")
    val tstdata =  sc.textFile("./testing.csv")
    val	result = input.map{ line =>
      val reader = new CSVReader(new	StringReader(line));
      reader.readNext();
    }

    val ss=result.collect().map(_.toList).mkString(",")
    val sss= ss.split(",")   // array of every list
    //val ssss= sss(6).split(";")  // array of every element
    //println(ssss(0))
    //println(sss.size)
    val trainSize = 440
    val a = (0 to trainSize).map(i => {

      val ssss= sss(i).split(";")
      val arr = ssss.slice(1, ssss.length - 1).map(_.toDouble)
      if (ssss(0) == "List(None") {
        knnVertex(None, arr)
      }
      else if (ssss(0) == "List(1")
        knnVertex(Some(1), arr)
      else if (ssss(0) == "List(2")
        knnVertex(Some(2), arr)
      else if (ssss(0) == "List(3")
        knnVertex(Some(3), arr)
      else
        knnVertex(None, arr)
    }) //end map


    //val g = knnGraph(a, 10)  // brute force method
    val start = System.currentTimeMillis()
    val g = knnGraphApprox(a, 2)  // approximation method
    val gs = semiSupervisedLabelPropagation(g)
    val end = System.currentTimeMillis()
    //  testing
   // val sc1 = new SparkContext(new SparkConf().setMaster("local").set("spark.driver.allowMultipleContexts", "true").setAppName("knnPredict"))


    val	tstresult = tstdata.map{ line =>
      val reader = new CSVReader(new	StringReader(line));
      reader.readNext();
    }
   print("testing_data")


    val tt=tstresult.collect().map(_.toList).mkString(",")
    val ttt= tt.split(",")   // array of every list
    val tstSize = 43
    val b = (0 to tstSize).map(i => {
      val tttt= ttt(i).split(";")
      knnPredict(gs, tttt.slice(1, tttt.length - 1).map(_.toDouble))})
    println(b)
    writeToFile("out.txt",b.toString)

    //error rate  29 is the testing data #
    var errorcounter = 0

    val c = (0 to tstSize).map(i => {
      val correctclass= ttt(i).split(";")
      if ( correctclass(0) == "List(1")
        if (b(i) != 1)
          errorcounter +=1
      if ( correctclass(0) == "List(2")
        if (b(i) != 2)
          errorcounter +=1
      if ( correctclass(0) == "List(3")
        if (b(i) != 3)
          errorcounter +=1
    })
    println(errorcounter.toDouble/tstSize)

  }  //end main


  def writeToFile(p: String, s: String): Unit = {
    val pw = new PrintWriter(new File(p))
    try pw.write(s) finally pw.close()
  }

  case class knnVertex(classNum:Option[Int],
                       pos:Array[Double]) extends Serializable {
    def dist(that:knnVertex): Double = {
//      val zp = pos.zip(that.pos)
//      val m = zp.map(x => (x._1-x._2)*(x._1-x._2))
//      val r = m.reduce(_+_)
      val res = math.sqrt(
        pos.zip(that.pos).map(x => (x._1-x._2)*(x._1-x._2)).reduce(_ + _))
      return res
    }
  }

  def knnGraph(a:Seq[knnVertex], k:Int) = {
    //val sc = new SparkContext(new SparkConf().setMaster("local").set("spark.driver.allowMultipleContexts", "true").setAppName("knnPredict"))
    val a2 = a.zipWithIndex.map(x => (x._2.toLong, x._1)).toArray
    val v = sc.makeRDD(a2)
    //val edge = v.map(v1 => (v1._1, a2.map(v2 => (v2._1, v1._2.dist(v2._2)))))
    val e = v.map(v1 => (v1._1, a2.map(v2 => (v2._1, v1._2.dist(v2._2)))
      .sortWith((e,f) => e._2 < f._2)
      .slice(1,k+1)
      .map(_._1)))
      .flatMap(x => x._2.map(vid2 =>
        Edge(x._1, vid2,
          1 / (1+a2(vid2.toInt)._2.dist(a2(x._1.toInt)._2)))))
    Graph(v,e)
  }

  def knnGraphApprox(a:Seq[knnVertex], k:Int) = {
    //val sc = new SparkContext(new SparkConf().setMaster("local").set("spark.driver.allowMultipleContexts", "true").setAppName("knnPredict"))
    val a2 = a.zipWithIndex.map(x => (x._2.toLong, x._1)).toArray
    val v = sc.makeRDD(a2)
    val n = 3
    val minMax =
      v.map(x => (x._2.pos(0), x._2.pos(0), x._2.pos(1), x._2.pos(1)))
        .reduce((a,b) => (math.min(a._1,b._1), math.max(a._2,b._2),
          math.min(a._3,b._3), math.max(a._4,b._4)))
    val xRange = minMax._2 - minMax._1
    val yRange = minMax._4 - minMax._3

    def calcEdges(offset: Double) =
      v.map(x => (math.floor((x._2.pos(0) - minMax._1)
        / xRange * (n-1) + offset) * n
        + math.floor((x._2.pos(1) - minMax._3)
        / yRange * (n-1) + offset),
        x))
        .groupByKey(n*n)
        .mapPartitions(ap => {
          val af = ap.flatMap(_._2).toList
          af.map(v1 => (v1._1, af.map(v2 => (v2._1, v1._2.dist(v2._2)))
            .toArray
            .sortWith((e,f) => e._2 < f._2)
            .slice(1,k+1)
            .map(_._1)))
            .flatMap(x => x._2.map(vid2 => Edge(x._1, vid2,
              1 / (1+a2(vid2.toInt)._2.dist(a2(x._1.toInt)._2)))))
            .iterator
        })

    val e = calcEdges(0.0).union(calcEdges(1))
      .distinct
      .map(x => (x.srcId,x))
      .groupByKey
      .map(x => x._2.toArray
        .sortWith((e,f) => e.attr > f.attr)
        .take(k))
      .flatMap(x => x)


    Graph(v,e)
  }

  def semiSupervisedLabelPropagation(g:Graph[knnVertex,Double],
                                     maxIterations:Int = 0) = {
    val maxIter = if (maxIterations == 0) g.vertices.count / 2
    else maxIterations

    var g2 = g.mapVertices((vid,vd) => (vd.classNum.isDefined, vd))
    var isChanged = true
    var i = 0

    do {
      val newV =
        g2.aggregateMessages[Tuple2[Option[Int],HashMap[Int,Double]]](
          ctx => {
            ctx.sendToSrc((ctx.srcAttr._2.classNum,
              if (ctx.dstAttr._2.classNum.isDefined)
                HashMap(ctx.dstAttr._2.classNum.get->ctx.attr)
              else
                HashMap[Int,Double]()))
            if (ctx.srcAttr._2.classNum.isDefined)
              ctx.sendToDst((None,
                HashMap(ctx.srcAttr._2.classNum.get->ctx.attr)))
          },
          (a1, a2) => {
            if (a1._1.isDefined)
              (a1._1, HashMap[Int,Double]())
            else if (a2._1.isDefined)
              (a2._1, HashMap[Int,Double]())
            else
              (None, a1._2 ++ a2._2.map{
                case (k,v) => k -> (v + a1._2.getOrElse(k,0.0)) })
          }
        )

      val newVClassVoted = newV.map(x => (x._1,
        if (x._2._1.isDefined)
          x._2._1
        else if (x._2._2.size > 0)
          Some(x._2._2.toArray.sortWith((a,b) => a._2 > b._2)(0)._1)
        else None
      ))

      isChanged = g2.vertices.join(newVClassVoted)
        .map(x => x._2._1._2.classNum != x._2._2)
        .reduce(_ || _)

      g2 = g2.joinVertices(newVClassVoted)((vid, vd1, u) =>
        (vd1._1, knnVertex(u, vd1._2.pos)))

      i += 1
    } while (i < maxIter && isChanged)

    g2.mapVertices((vid,vd) => vd._2)
  }



  def knnPredict[E](g:Graph[knnVertex,E],pos:Array[Double]) =
    g.vertices
      .filter(_._2.classNum.isDefined)
      .map(x => (x._2.classNum.get, x._2.dist(knnVertex(None,pos))))
      .min()(new Ordering[Tuple2[Int,Double]] {
        override def compare(a:Tuple2[Int,Double],
                             b:Tuple2[Int,Double]): Int =
          a._2.compare(b._2)
      })
      ._1


}
