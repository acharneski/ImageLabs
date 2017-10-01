/*
 * Copyright (c) 2017 by Andrew Charneski.
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

package interactive.word2vec

import java.io._
import java.util.zip.GZIPInputStream

import _root_.util._
import com.simiacryptus.util.StreamNanoHTTPD
import com.simiacryptus.util.io.HtmlNotebookOutput
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object SparkLab extends Report {
  System.setProperty("hadoop.home.dir", "D:\\SimiaCryptus\\hadoop")
  val dataFolder = "file:///H:/data_word2vec"

  lazy val sparkConf: SparkConf = new SparkConf().setAppName(getClass.getName)
    .setMaster("local[4]")
    //.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    //.set("spark.kryoserializer.buffer.max", "64")
  lazy val sc = new SparkContext(sparkConf)

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array() ⇒ new SparkLab(server, out).run()
    })

  }

}
import SparkLab._

object VectorUtils {
  implicit def convert(values:Seq[Float])=new VectorUtils(values)
}
case class VectorUtils(values:Seq[Float]) {
  def +(right:Seq[Float]) : Seq[Float] = {
    values.zip(right).map(x=>x._1+x._2)
  }
  def -(right:Seq[Float]) : Seq[Float] = {
    values.zip(right).map(x=>x._1-x._2)
  }
  def *(right:Seq[Float]) : Seq[Float] = {
    values.zip(right).map(x=>x._1*x._2)
  }
  def *(right:Float) : Seq[Float] = {
    values.map(x=>x*right)
  }
  def sq : Seq[Float] = {
    values.map(x=>x*x)
  }
  def l0 : Float = {
    values.size.toFloat
  }
  def l1 : Float = {
    values.sum
  }
  def l2 : Float = {
    Math.sqrt(sq.sum).toFloat
  }
  def mean : Float = {
    l1 / l0
  }
  def magnitude : Float = {
    l2
  }
  def withMath : VectorUtils = {
    this
  }
}
import VectorUtils.convert

class SparkLab(server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  def loadBin(file: String) = {
    def readUntil(inputStream: DataInputStream, term: Char, maxLength: Int = 1024 * 8): String = {
      var char: Char = inputStream.readByte().toChar
      val str = new StringBuilder
      while (!char.equals(term)) {
        str.append(char)
        assert(str.size < maxLength)
        char = inputStream.readByte().toChar
      }
      str.toString
    }
    val inputStream: DataInputStream = new DataInputStream(new GZIPInputStream(new FileInputStream(file)))
    try {
      val header = readUntil(inputStream, '\n')
      val (records, dimensions) = header.split(" ") match {
        case Array(records, dimensions) => (records.toInt, dimensions.toInt)
      }
      new Word2VecModel((0 until records).toArray.map(recordIndex => {
        readUntil(inputStream, ' ') -> (0 until dimensions).map(dimensionIndex => {
          inputStream.readFloat()
        }).toArray
      }).toMap)
    } finally {
      inputStream.close()
    }
  }

  implicit def +(left:Seq[Float],right:Seq[Float]): Seq[Float] = {
    left.zip(right).map(x=>x._1+x._2)
  }

  def run(awaitExit: Boolean = true): Unit = {
    val model = loadBin("C:\\Users\\andre\\Downloads\\GoogleNews-vectors-negative300.bin.gz")
    val vectors = model.getVectors.mapValues(_.toList)
    val rdd = sc.parallelize(vectors.toList, 256)
    println(vectors.keys.mkString(", "))
    for(words <- List(
      ("dog", "cat"),
      ("angry", "happy")
    )) {
      val vector1 = vectors(words._1)
      val vector2 = vectors(words._2)
      println(s"${words._1} => $vector1")
      println(s"${words._2} => $vector2")
      val c = (vector1 - vector2).magnitude
      rdd.map(x=>{
        val (key, vector) = x
        val a = (vector - vector1).magnitude
        val b = (vector - vector2).magnitude
        key -> (a/(a+b), (a+b)/c)
      }).sortBy(_._2._2).take(10).foreach(x=>{
        val (key, (pos,offaxis)) = x
        println(s"  $key : $pos / $offaxis")
      })
    }
    //rdd.take(100).foreach(println)
    if (awaitExit) waitForExit()
  }

}

