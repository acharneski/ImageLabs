/*
 * Copyright (c) 2018 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

///*
// * Copyright (c) 2018 by Andrew Charneski.
// *
// * The author licenses this file to you under the
// * Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance
// * with the License.  You may obtain a copy
// * of the License at
// *
// *   http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing,
// * software distributed under the License is distributed on an
// * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// * KIND, either express or implied.  See the License for the
// * specific language governing permissions and limitations
// * under the License.
// */
//
//package interactive.word2vec
//
//import java.io._
//import java.net.URI
//import java.util.zip.GZIPInputStream
//
//import _root_.util._
//import com.simiacryptus.util.StreamNanoHTTPD
//import com.simiacryptus.util.io.HtmlNotebookOutput
//import org.apache.commons.io.IOUtils
//import org.apache.hadoop.fs.{Path, RemoteIterator}
//import org.apache.spark.SparkConf
//import org.apache.spark.sql.SparkSession
//
//import scala.collection.mutable.ArrayBuffer
//
//object LocalLab extends Report {
//  System.setProperty("hadoop.home.dir", "D:\\SimiaCryptus\\hadoop")
//
//  lazy val sparkConf: SparkConf = new SparkConf().setAppName(getClass.getName)
//    .setMaster("local")
//  //.setByCoord("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//  //.setByCoord("spark.kryoserializer.buffer.max", "64")
//  lazy val sqlContext = SparkSession.builder().config(sparkConf).getOrCreate()
//  lazy val sc = sqlContext.sparkContext
//
//  def main(args: Array[String]): Unit = {
//
//    report((server, out) ⇒ args match {
//      case Array() ⇒ new LocalLab(server, out).eval()
//    })
//
//  }
//
//}
//
//import interactive.word2vec.LocalLab._
//import interactive.word2vec.VectorUtils.convert
//
//
//
//class LocalLab(server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {
//
//  def synonymns(rdd: Word2VecLocalModel, a: String) = out.eval {
//    val List(vector) = findVector(rdd, a)
//    System.out.println(s"Related to $a")
//    val result = rdd.map(x => {
//      val (key, vector) = x
//      key -> (vector ^ vector)
//    }).filter(!_._2.toDouble.isNaN)
//      .sortBy(_._2).take(50).map(t => (t._1, t._2)).toList
//    result.foreach(x=>System.out.println("%s -> %.3f".format(x._1,x._2)))
//    result
//  }
//
//  def synonymns(rdd: Word2VecLocalModel, a: Seq[String], n: Int = 50) = out.eval {
//    val vectors = findVector(rdd, a: _*)
//    System.out.println(s"Related to $a")
//    val result = rdd.map(x => {
//      val (key, vector) = x
//      key -> vectors.map(_ ^ vector).map(x => x * x).sum
//    }).filter(!_._2.toDouble.isNaN)
//      .sortBy(_._2).take(n).map(t => (t._1, t._2)).toList
//    result.foreach(x=>System.out.println("%s -> %.3f".format(x._1,x._2)))
//    result
//  }
//
//  def inplane(rdd: Word2VecLocalModel, a: Seq[String], n: Int = 50) = out.eval {
//    val vectors = findVector(rdd, a: _*)
//    System.out.println(s"Inplane apply $a")
//    val result = rdd.map(x => {
//      val (key, vector) = x
//      key -> vectors.tail.foldLeft(vector - vectors.head)((x, y) => x without (y - vectors.head)).magnitude.toDouble
//    }).filter(!_._2.toDouble.isNaN)
//      .sortBy(_._2).take(n).map(t => (t._1, t._2)).toList
//    result.foreach(x=>System.out.println("%s -> %.3f".format(x._1,x._2)))
//    result
//  }
//
//  def analogy1(rdd: Word2VecLocalModel, n: Int = 50)(a: String, b: String)(c: String) = out.eval {
//    System.out.println(s"$a is to $b as $c is to...")
//    val List(va, vb, vc) = findVector(rdd, a, b, c)
//    val vd = vb - va + vc
//    val result = rdd.map(x => {
//      val (key, vector) = x
//      key -> (vd ^ vector)
//    }).filter(!_._2.toDouble.isNaN)
//      .sortBy(_._2).take(n).map(t => (t._1, t._2)).toList
//    result.foreach(x=>System.out.println("%s -> %.3f".format(x._1,x._2)))
//    result
//  }
//
//  def analogy2(rdd: Word2VecLocalModel, b: String, a: String)(c: String)(n: Int) = out.eval {
//    System.out.println(s"$a is to $b as $c is to...")
//    val List(va, vb, vc) = findVector(rdd, a, b, c)
//    val result = rdd.map(x => {
//      val (key, vector) = x
//      key -> ((vb ^ vector) - (va ^ vector) + (vc ^ vector))
//    }).filter(!_._2.toDouble.isNaN)
//      .sortBy(_._2).take(n).map(t => (t._1, t._2)).toList
//    result.foreach(x=>System.out.println("%s -> %.3f".format(x._1,x._2)))
//    result
//  }
//
//  def membership(rdd: Word2VecLocalModel, a: Seq[String]) = out.eval {
//    val vectors = a.zip(findVector(rdd, a: _*))
//    System.out.println(s"Membership analysis of group $a")
//    val result = vectors.map(x => {
//      val (key, vector) = x
//      key -> vectors.map(_._2 ^ vector).map(x => x * x).sum
//    }).filter(!_._2.toDouble.isNaN)
//      .sortBy(_._2).map(t => (t._1, t._2)).toList
//    result.foreach(x=>System.out.println("%s -> %.3f".format(x._1,x._2)))
//    result
//  }
//
//  def continuum(rdd: Word2VecLocalModel, a: String, b: String, n: Int = 50) = out.eval {
//    val List(va, vb) = findVector(rdd, a, b)
//    System.out.println(s"Spectrum from $a to $b")
//    val result = rdd.map(x => {
//      val (key, v) = x
//      val a = va ^ v
//      val b = v ^ vb
//      val c = va ^ vb
//      key -> (a / (a + b), (a + b - c) / c)
//    }).filter(!_._2._1.toDouble.isNaN).filter(!_._2._2.toDouble.isNaN)
//      .sortBy(_._2._2).take(n).sortBy(_._2._1).toList
//    result.foreach(x=>System.out.println(s"%s -> %.3f / %.3f".format(x._1,x._2._1,x._2._2)))
//    result
//  }
//
//  def writeModel(rdd: Seq[(String, Array[Float])], file:String) = {
//    val fileSystem = org.apache.hadoop.fs.FileSystem.get(new URI(file),sc.hadoopConfiguration)
//
//    {
//      val out = fileSystem.create(new Path(file+"/metadata.txt"), true)
//      IOUtils.write(rdd.map(_._1).mkString("\n"), out)
//      out.close()
//    }
//    {
//      val out = fileSystem.create(new Path(file+"/data.tsv"), true)
//      IOUtils.write(rdd.map(_._2.mkString("\t")).mkString("\n"), out)
//      out.close()
//    }
//  }
//
//  def findVector(rdd: Word2VecLocalModel, str: String*): List[Array[Float]] = {
//    val value = rdd.filter(x => str.contains(x._1)).toMap
//    val tuple: (String, Array[Float]) = value.head
//    if (null != tuple) {
//      str.map(x => value.getOrElse(x, null)).toList
//    } else {
//      val l = str.map(_.toLowerCase)
//      val value = rdd.filter(x => l.contains(x._1.toLowerCase)).toMap
//      str.map(x => value.getOrElse(x, null)).toList
//    }
//  }
//
//  def eval(awaitExit: Boolean = true): Unit = {
//    val rdd = loadLocal()
//
//    {
//      val tuples = inplane(rdd, List("happy", "sad", "hungry", "excited", "tired"), n = 2000)
//      writeModel(tuples.map(_._1).zip(findVector(rdd,tuples.map(_._1):_*)),"file:///D://SimiaCryptus/data/emotions1")
//    }
//
//    {
//      val tuples = synonymns(rdd, List("happy", "sad", "hungry", "excited", "tired"), n = 2000)
//      writeModel(tuples.map(_._1).zip(findVector(rdd,tuples.map(_._1):_*)),"file:///D://SimiaCryptus/data/emotions2")
//    }
//
//    {
//      val tuples = inplane(rdd, List("dog", "cat", "bird"), n=500)
//      writeModel(tuples.map(_._1).zip(findVector(rdd,tuples.map(_._1):_*)),"file:///D://SimiaCryptus/data/animals")
//    }
//
//    synonymns(rdd, List("dog", "cat", "bird"))
//
//    inplane(rdd, List("happy", "curious", "joyful"),n=5000)
//    synonymns(rdd, List("happy", "curious", "joyful"),n=5000)
//
//    membership(rdd, List(
//      "cat", "lion", "tiger", "animal", "rock", "this", "the"
//    ))
//    synonymns(rdd, List("Cocker_Spaniel", "Miniature_Pinscher", "Golden_Retriever"))
//    synonymns(rdd, List("France", "capitol", "city", "name"))
//    synonymns(rdd, List("New_York", "Los_Angeles", "London", "Paris", "Moscow"))
//    synonymns(rdd, "cat")
//
//    continuum(rdd, "dog", "cat")
//
//    continuum(rdd, "good", "bad")
//    continuum(rdd, "USA", "USSR")
//
//    analogy2(rdd, "girl", "boy")("man")(50)
//    analogy1(rdd)("boy", "girl")("man")
//
//    analogy2(rdd, "London", "England")("USA")(50)
//    analogy1(rdd)("England", "London")("USA")
//
//    analogy2(rdd, "Greyhound", "horse")("cow")(50)
//    analogy2(rdd, "Greyhound", "horse")("cow")(50)
//
//
//    if (awaitExit) waitForExit()
//  }
//
//  class Word2VecLocalModel(keys: List[String], flatValues: Array[Float]) {
//    def map[T](function: ((String,Array[Float])) => T) = entries.map(function(_))
//    def filter[T](function: ((String,Array[Float])) => Boolean) = entries.filter(function(_))
//    def foreach(function: ((String,Array[Float])) => Unit) = entries.foreach(function(_))
//
//    def get(key:String): Array[Float] = {
//      val i = keys.indexOf(key)
//      flatValues.slice(i*dimensions,(i+1)*dimensions)
//    }
//    def entries: Stream[(String, Array[Float])] = {
//      keys.toStream.map(x=>x->get(x))
//    }
//    def dimensions = keys.size / flatValues.length
//  }
//
//  def loadLocal(): Word2VecLocalModel = {
//    val file = "C:\\Users\\andre\\Downloads\\GoogleNews-vectors-negative300.bin.gz"
//
//    def loadStream(inputStream: DataInputStream): Stream[(String, Array[Float])] = {
//      def readUntil(inputStream: DataInputStream, term: Char, maxLength: Int = 1024 * 8): String = {
//        var char: Char = inputStream.readByte().toChar
//        val str = new StringBuilder
//        while (!char.equals(term)) {
//          str.append(char)
//          assert(str.size < maxLength)
//          char = inputStream.readByte().toChar
//        }
//        str.toString
//      }
//
//      val header = readUntil(inputStream, '\n')
//      val (records, dimensions) = header.split(" ") match {
//        case Array(records, dimensions) => (records.toInt, dimensions.toInt)
//      }
//      (0 until records).toStream.map(recordIndex => {
//        readUntil(inputStream, ' ') -> (0 until dimensions).map(dimensionIndex => {
//          java.lang.Float.intBitsToFloat(java.lang.Integer.reverseBytes(inputStream.readInt()))
//
//        }).toArray
//      })
//    }
//
//    implicit def toList[T](v: RemoteIterator[T]) = {
//      Stream.continually[Option[T]](if (v.hasNext) Option(v.next()) else None).takeWhile(_.isDefined).map(_.get)
//    }
//
//    val stream: Stream[(String, Array[Float])] = loadStream(new DataInputStream(new GZIPInputStream(new FileInputStream(file))))
//    val keys = new ArrayBuffer[String]()
//    val array = new ArrayBuffer[Float]()
//    for(item <- stream) {
//      keys += item._1
//      array ++= item._2
//    }
//    new Word2VecLocalModel(keys.toList, array.toArray)
//  }
//
//
//}
//
