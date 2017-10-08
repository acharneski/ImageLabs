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
import java.lang
import java.net.URI
import java.util.zip.GZIPInputStream

import _root_.util._
import com.simiacryptus.util.StreamNanoHTTPD
import com.simiacryptus.util.io.HtmlNotebookOutput
import com.simiacryptus.util.lang.CodeUtil
import org.apache.commons.io.IOUtils
import org.apache.hadoop.fs.{Path, RemoteIterator}
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.{ArrayBuffer, WrappedArray}
import scala.util.Try

object SparkLab extends Report {
  System.setProperty("hadoop.home.dir", "D:\\SimiaCryptus\\hadoop")

  lazy val sparkConf: SparkConf = new SparkConf().setAppName(getClass.getName)
    .setMaster("local")
  lazy val sqlContext = SparkSession.builder().config(sparkConf).getOrCreate()
  lazy val sc = sqlContext.sparkContext

  def main(args: Array[String]): Unit = {
    report((server, out) ⇒ args match {
      case Array() ⇒ new SparkLab(server, out).run()
    })
  }

}

import interactive.word2vec.SparkLab._
import interactive.word2vec.VectorUtils._

class SparkLab(server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  def inplane(rdd: RDD[(String, Array[Float])], positiveExamples: Seq[String], n: Int = 50) = out.eval {
    val primaryBasis = findVector(rdd, positiveExamples: _*)
    val orthogonalBasis = new ArrayBuffer[Array[Float]]()
    orthogonalBasis += primaryBasis.reduce(_ + _) / primaryBasis.size
    primaryBasis.foreach(vector => {
      orthogonalBasis += orthogonalBasis.tail.foldLeft(vector - orthogonalBasis.head)((l, r) => l without r).unitV
    })
    val orthogonalBasisArray = orthogonalBasis.toArray
    System.out.println(s"Inplane with $positiveExamples")
    val result = rdd.map(x => {
      val (key, vector) = x
      (key, orthogonalBasisArray.tail.foldLeft(vector - orthogonalBasisArray.head)((l, r) => l without r).magnitude.toDouble, vector)
    }).filter(!_._2.toDouble.isNaN)
      .sortBy(_._2).take(n).toList
    System.out.println(result.map(_._1).mkString(", "))
    result
  }

  def analogy(rdd: RDD[(String, Array[Float])], b: String, a: String)(c: String)(n: Int) = out.eval {
    System.out.println(s"$a is to $b as $c is to...")
    val List(va, vb, vc) = findVector(rdd, a, b, c)
    val result = rdd.map(x => {
      val (key, vector) = x
      key -> Math.abs((vb ^ vector) - (va ^ vector) + (vc ^ vector))
    }).filter(!_._2.toDouble.isNaN)
      .sortBy(_._2).take(n).map(t => (t._1, t._2)).toList
    result.foreach(x => System.out.println("%s -> %.3f".format(x._1, x._2)))
    result
  }

  def membership(rdd: RDD[(String, Array[Float])], a: Seq[String]) = out.eval {
    val vectors = a.zip(findVector(rdd, a: _*))
    System.out.println(s"Membership analysis of group $a")
    val result = vectors.map(x => {
      val (key, vector) = x
      key -> vectors.map(_._2 ^ vector).map(x => x * x).sum
    }).filter(!_._2.toDouble.isNaN)
      .sortBy(_._2).map(t => (t._1, t._2)).toList
    result.foreach(x => System.out.println("%s -> %.3f".format(x._1, x._2)))
    result
  }

  def continuum(rdd: RDD[(String, Array[Float])], a: String, b: String, n: Int = 50) = out.eval {
    val List(va, vb) = findVector(rdd, a, b)
    System.out.println(s"Spectrum from $a to $b")
    val result = rdd.map(x => {
      val (key, v) = x
      val a = va ^ v
      val b = v ^ vb
      val c = va ^ vb
      key -> (a / (a + b), (a + b - c) / c)
    }).filter(!_._2._1.toDouble.isNaN).filter(!_._2._2.toDouble.isNaN)
      .sortBy(_._2._2).take(n).sortBy(_._2._1).toList
    result.foreach(x => System.out.println(s"%s -> %.3f / %.3f".format(x._1, x._2._1, x._2._2)))
    result
  }

  def writeModel(rdd: Seq[(String, Array[Float])], file: String) = {
    val fileSystem = org.apache.hadoop.fs.FileSystem.get(new URI(file), sc.hadoopConfiguration)

    val sortedGroups = clusterKMeans2(numClusters = 10,
      numIterations = 20,
      tuples = rdd.toList).zipWithIndex.flatMap(x => x._1.map(y => (y._1, y._2, x._2)))

    {
      val out = fileSystem.create(new Path(file + "/metadata.txt"), true)
      IOUtils.write("Label\tGroup\n", out)
      IOUtils.write(sortedGroups.map(x => List(x._1, x._3).mkString("\t")).mkString("\n"), out)
      out.close()
    }
    {
      val out = fileSystem.create(new Path(file + "/data.tsv"), true)
      IOUtils.write(sortedGroups.map(_._2.mkString("\t")).mkString("\n"), out)
      out.close()
    }
  }

  def findVector(rdd: RDD[(String, Array[Float])], str: String*): List[Array[Float]] = {
    val caseList = str.map(_.toLowerCase)
    val value: Map[String, Array[Float]] = rdd.filter(x => caseList.contains(x._1.toLowerCase)).collect().toMap
    str.map(key => value.getOrElse(key, value.map(t => t._1.toLowerCase -> t._2).getOrElse(key.toLowerCase, null))).toList
  }

  def run(awaitExit: Boolean = true): Unit = {
    CodeUtil.projectRoot = new File("../image-labs")
    out.sourceRoot = "https://github.com/acharneski/imagelabs/tree/blog-2017-10-07/"

    //val rdd = loadWord2VecRDD().mapValues(v => v.unitV).persist(StorageLevel.MEMORY_ONLY)
    val rdd = loadFastTextRDD().mapValues(v => v.unitV).persist(StorageLevel.MEMORY_ONLY)

    continuum(rdd, "hell", "heaven")
    continuum(rdd, "moist", "dry")
    continuum(rdd, "love", "hate")
    continuum(rdd, "imperfect", "perfection")
    writeModel(selectCluster_inplane(rdd = rdd,
      positiveExamples = List("holy", "bible", "god", "cult", "armageddon", "sin", "hell", "heaven")
    ).map(x => (x._1, x._3)), "file:///D://SimiaCryptus/data/wordCluster_emotes")


    out.out("Our first example is to generate a tree of words with the seed words \"happy\", \"sad\", \"laughing\", \"crying\", \"depressed\", and \"excited\"")
    writeModel(selectCluster_inplane(rdd = rdd,
      positiveExamples = List("happy", "sad", "laughing", "crying", "depressed", "excited")
    ).map(x => (x._1, x._3)), "file:///D://SimiaCryptus/data/wordCluster_emotes")

    out.out("We can also generate a continuous spectrum from one term to another")
    continuum(rdd, "happy", "sad")

    out.out("The understanding of word relationships will naturally resemble other groupings, such as geography and politics:")
    writeModel(selectCluster_inplane(rdd = rdd,
      positiveExamples = List("Earth", "London", "Chicago", "USA", "England", "USSR", "Africa", "Saturn", "Sun")
    ).map(x => (x._1, x._3)), "file:///D://SimiaCryptus/data/wordCluster_locale")
    continuum(rdd, "USA", "USSR")
    writeModel(selectCluster_inplane(rdd = rdd,
      positiveExamples = List("communist", "democrat", "republican", "conservative", "liberal")
    ).map(x => (x._1, x._3)), "file:///D://SimiaCryptus/data/wordCluster_politics")

    out.out("A couple of other examples:")
    writeModel(selectCluster_inplane(rdd = rdd,
      positiveExamples = List("math", "science", "english", "history", "art", "gym")
    ).map(x => (x._1, x._3)), "file:///D://SimiaCryptus/data/wordCluster_subjects")
    continuum(rdd, "good", "bad")

    out.out("We can also search for analogic matches")
    analogy(rdd, "boy", "girl")("man")(50)
    analogy(rdd, "wait", "waiting")("run")(50)
    analogy(rdd, "on", "off")("up")(50)
    analogy(rdd, "USA", "USSR")("England")(50)

    if (awaitExit) waitForExit()
  }

  private def selectCluster_inplane(rdd: RDD[(String, Array[Float])], positiveExamples: List[String]) = {
    val data = inplane(rdd = rdd,
      positiveExamples = positiveExamples,
      n = 1000)
    Try {
      printTree_KMeans(positiveExamples, numberOfClusters = 13, data)
    }
    data
  }

  private def printTree_KMeans(positiveExamples: List[String], numberOfClusters: Int, data: List[(String, Double, Array[Float])], levelIndent: String = "  ") = out.eval {
    val sortedGroups: List[List[(String, Double, Array[Float])]] = clusterKMeans3(numClusters = numberOfClusters,
      numIterations = 20,
      tuples = data).sortBy(_.map(_._2).sum)

    def printNode(group: List[(String, Double, Array[Float])], indent: String) = {
      val axis = group.flatMap(x => group.map(y => (x, y))).maxBy(t => t._1._3 ^ t._2._3)
      for (item <- group.sortBy(x => {
        val a = x._3 ^ axis._1._3
        val b = x._3 ^ axis._2._3
        a / (a + b)
      })) {
        System.out.println(indent + "%s".format(item._1))
      }
    }

    def printTree(group: List[(String, Double, Array[Float])], indent: String = levelIndent): Unit = {
      if (group.size < (numberOfClusters * 3)) {
        printNode(group, indent)
      } else try {
        for (group <- clusterKMeans3(group, 5, 20).sortBy(_.map(_._2).sum)) {
          System.out.println(indent + "-----------------------------")
          printTree(group, indent = indent + levelIndent)
        }
      } catch {
        case e: Throwable =>
          printNode(group, indent)
      }
    }

    for (group <- sortedGroups) {
      System.out.println("-----------------------------")
      printTree(group)
    }
    sortedGroups
  }

  private def clusterKMeans3(tuples: List[(String, Double, Array[Float])], numClusters: Int, numIterations: Int): List[List[(String, Double, Array[Float])]] = {
    val parsedData: RDD[linalg.Vector] = sc.parallelize(tuples.map(x => Vectors.dense(x._3.map(_.toDouble))))
    val clusters = KMeans.train(parsedData, numClusters, numIterations)
    tuples.groupBy(t => clusters.predict(Vectors.dense(t._3.map(_.toDouble)))).values.toList
  }

  private def clusterKMeans2(tuples: List[(String, Array[Float])], numClusters: Int, numIterations: Int): List[List[(String, Array[Float])]] = {
    val parsedData: RDD[linalg.Vector] = sc.parallelize(tuples.map(x => Vectors.dense(x._2.map(_.toDouble))))
    val clusters = KMeans.train(parsedData, numClusters, numIterations)
    tuples.groupBy(t => clusters.predict(Vectors.dense(t._2.map(_.toDouble)))).values.toList
  }

  class Word2VecLocalModel(keys: List[String], flatValues: Array[Float]) {
    def get(key: String): Array[Float] = {
      val i = keys.indexOf(key)
      flatValues.slice(i * dimensions, (i + 1) * dimensions)
    }

    def entries: Stream[(String, Array[Float])] = {
      keys.toStream.map(x => x -> get(x))
    }

    def dimensions = keys.size / flatValues.length
  }

  def readUntil(inputStream: InputStream, term: Char, maxLength: Int = 1024 * 8): String = {
    var char: Char = inputStream.read().toChar
    val str = new StringBuilder
    while (!char.equals(term)) {
      str.append(char)
      assert(str.size < maxLength)
      char = inputStream.read().toChar
    }
    str.toString
  }

  implicit def toList[T](v: RemoteIterator[T]) = {
    Stream.continually[Option[T]](if (v.hasNext) Option(v.next()) else None).takeWhile(_.isDefined).map(_.get)
  }

  def loadWord2VecRDD( parquetUrl: String = "file:///H:/data_word2vec0/final",
                       binUrl: String = "C:\\Users\\andre\\Downloads\\GoogleNews-vectors-negative300.bin.gz"
                     ): RDD[(String, Array[Float])] = {
    cache(parquetUrl) {
      val inputStream = new DataInputStream(new GZIPInputStream(new BufferedInputStream(new FileInputStream(binUrl))))
      val header = readUntil(inputStream, '\n')
      val (records, dimensions) = header.split(" ") match {
        case Array(records, dimensions) => (records.toInt, dimensions.toInt)
      }
      (0 until records).toStream.map(recordIndex => {
        readUntil(inputStream, ' ') -> (0 until dimensions).map(dimensionIndex => {
          java.lang.Float.intBitsToFloat(java.lang.Integer.reverseBytes(inputStream.readInt()))
        }).toArray
      })
    }
  }

  def loadFastTextRDD( parquetUrl: String = "file:///H:/data_wiki.en/final",
                       binUrl: String = "C:\\Users\\andre\\Downloads\\wiki.en\\wiki.en.vec"
                     ): RDD[(String, Array[Float])] = {
    cache(parquetUrl) {

      val inputStream = new BufferedReader(new InputStreamReader(new FileInputStream(binUrl)))
      val header = inputStream.readLine()
      val (records, dimensions) = header.split(' ') match {
        case Array(records, dimensions) => (records.toInt, dimensions.toInt)
      }
      (0 until records).map(recordIndex  => try {
        val str = inputStream.readLine()
        val line = str.split(' ')
        line.head -> line.tail.map(_.toFloat)
      } catch {
        case e:Throwable=>
          e.printStackTrace()
          null
      }).toStream.takeWhile(null != _)
    }
  }

  private def cache(file: String)(data: => Stream[(String, Array[Float])],
                                  tempFolder: String = file + "/../",
                                  bufferSize: Int = 100000
  ): RDD[(String, Array[Float])] = {
    val fileSystem = org.apache.hadoop.fs.FileSystem.get(new URI(file), sc.hadoopConfiguration)
    if (!fileSystem.exists(new Path(file))) {
      val cleanup = new ArrayBuffer[String]()
      data.grouped(bufferSize).zipWithIndex.map(x => {
        val tempDest = tempFolder + x._2
        cleanup += tempDest
        val rdd: RDD[(String, Array[Float])] = sc.parallelize(x._1, 1)
        val schema = StructType(List(StructField("term", StringType), StructField("vector", ArrayType(FloatType))))
        sqlContext.createDataFrame(rdd.map(x => Row(x._1, x._2)), schema).write.parquet(tempDest)
        sqlContext.read.parquet(tempDest)
      }).reduce(_.union(_)).repartition(16).write.parquet(file)
      for (file <- cleanup) fileSystem.delete(new Path(file), true)
    }
    sqlContext.read.parquet(file).rdd.map(row => row.getAs[String]("term") -> row.getAs[WrappedArray.ofRef[lang.Float]]("vector").toArray.map(_.toFloat))
  }

}

