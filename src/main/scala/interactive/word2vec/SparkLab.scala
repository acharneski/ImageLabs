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
import java.net.URI
import java.util.zip.GZIPInputStream

import _root_.util._
import com.simiacryptus.util.StreamNanoHTTPD
import com.simiacryptus.util.io.HtmlNotebookOutput
import com.simiacryptus.util.lang.CodeUtil
import org.apache.commons.io.IOUtils
import org.apache.hadoop.fs.{Path, RemoteIterator}
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{GaussianMixture, KMeans, PowerIterationClustering}
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Try

object SparkLab extends Report {
  System.setProperty("hadoop.home.dir", "D:\\SimiaCryptus\\hadoop")

  lazy val sparkConf: SparkConf = new SparkConf().setAppName(getClass.getName)
    .setMaster("local")
  //.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  //.set("spark.kryoserializer.buffer.max", "64")
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

  def synonymns(rdd: RDD[(String, Array[Float])], a: String) = out.eval {
    val List(vector) = findVector(rdd, a)
    System.out.println(s"Related to $a")
    val result = rdd.map(x => {
      val (key, vector) = x
      key -> (vector ^ vector)
    }).filter(!_._2.toDouble.isNaN)
      .sortBy(_._2).take(50).map(t => (t._1, t._2)).toList
    result.foreach(x => System.out.println("%s -> %.3f".format(x._1, x._2)))
    result
  }

  def wordCluster(rdd: RDD[(String, Array[Float])],
                  positiveExamples: Seq[String],
                  negativeExamples: Seq[String] = Seq.empty,
                  n: Int = 50,
                  power: Int = 1): List[(String, Double, Array[Float])] = out.eval {
    val allItems = (positiveExamples ++ negativeExamples).distinct.toList
    val allvectors = allItems.zip(findVector(rdd, allItems: _*)).toMap
    val posVectors = positiveExamples.map(allvectors(_))
    val negVectors = negativeExamples.map(allvectors(_))
    System.out.println(s"Related to ${positiveExamples.mkString(", ")} but not ${negativeExamples.mkString(", ")} ")
    val result = rdd.map(x => {
      val (key, vector) = x
      (key,
        if (positiveExamples.contains(key)) Double.NegativeInfinity else {
          negVectors.map(v => Math.pow(v ^ vector, power)).sum - posVectors.map(v => Math.pow(v ^ vector, power)).sum *
            java.lang.Double.compare(0, power)
        },
        vector)
    }).filter(!_._2.isNaN).sortBy(_._2).take(n).toList
    result.foreach(x => System.out.println("%s -> %.3f".format(x._1, x._2)))
    result
  }

  def inplane(rdd: RDD[(String, Array[Float])], positiveExamples: Seq[String], n: Int = 50) = out.eval {
    val primaryBasis = findVector(rdd, positiveExamples: _*)
    val orthogonalBasis = new ArrayBuffer[Array[Float]]()
    orthogonalBasis += primaryBasis.reduce(_+_) / primaryBasis.size
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

  def analogy1(rdd: RDD[(String, Array[Float])], n: Int = 50)(a: String, b: String)(c: String) = out.eval {
    System.out.println(s"$a is to $b as $c is to...")
    val List(va, vb, vc) = findVector(rdd, a, b, c)
    val vd = vb - va + vc
    val result = rdd.map(x => {
      val (key, vector) = x
      key -> (vd ^ vector)
    }).filter(!_._2.toDouble.isNaN)
      .sortBy(_._2).take(n).map(t => (t._1, t._2)).toList
    result.foreach(x => System.out.println("%s -> %.3f".format(x._1, x._2)))
    result
  }

  def analogy2(rdd: RDD[(String, Array[Float])], b: String, a: String)(c: String)(n: Int) = out.eval {
    System.out.println(s"$a is to $b as $c is to...")
    val List(va, vb, vc) = findVector(rdd, a, b, c)
    val result = rdd.map(x => {
      val (key, vector) = x
      key -> ((vb ^ vector) - (va ^ vector) + (vc ^ vector))
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

    val sortedGroups = clusterKMeans(numClusters = 10,
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

    val rdd = loadRDD().mapValues(v => v.unitV).persist(StorageLevel.MEMORY_ONLY)

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
    analogy2(rdd, "boy", "girl")("man")(50)
    analogy2(rdd, "wait", "waiting")("run")(50)
    analogy2(rdd, "on", "off")("up")(50)
    analogy2(rdd, "USA", "USSR")("England")(50)

    if (awaitExit) waitForExit()
  }

  private def selectCluster_power(rdd: RDD[(String, Array[Float])], positiveExamples: List[String], negativeExamples: List[String] = List.empty) = {
    val data = wordCluster(rdd = rdd,
      positiveExamples = positiveExamples,
      negativeExamples = negativeExamples,
      n = 1000)
    Try {
      printTree_KMeans(positiveExamples, numberOfClusters = 3, data)
    }
    //    Try { printTree_Gaussian(positiveExamples, numberOfClusters, data) }
    //    Try { printTree_PIC(positiveExamples, numberOfClusters, data) }
    data
  }

  private def selectCluster_inplane(rdd: RDD[(String, Array[Float])], positiveExamples: List[String]) = {
    val data = inplane(rdd = rdd,
      positiveExamples = positiveExamples,
      n = 1000)
    Try {
      printTree_KMeans(positiveExamples, numberOfClusters = 13, data)
    }
    //    Try { printTree_Gaussian(positiveExamples, numberOfClusters, data) }
    //    Try { printTree_PIC(positiveExamples, numberOfClusters, data) }
    data
  }

  private def printTree_KMeans(positiveExamples: List[String], numberOfClusters: Int, data: List[(String, Double, Array[Float])], levelIndent: String = "  ") = out.eval {
    val sortedGroups: List[List[(String, Double, Array[Float])]] = clusterKMeans2(numClusters = numberOfClusters,
      numIterations = 20,
      tuples = data).sortBy(_.map(_._2).sum)

    def printNode(group: List[(String, Double, Array[Float])], indent: String = levelIndent): Unit = {
      if (group.size < (numberOfClusters * 3)) {
        for (item <- group) {
          System.out.println(indent + "%s".format(item._1))
        }
      } else try {
        for (group <- clusterKMeans2(group, 5, 20).sortBy(_.map(_._2).sum)) {
          System.out.println(indent + "-----------------------------")
          printNode(group, indent = indent + levelIndent)
        }
      } catch {
        case e: Throwable =>
          for (item <- group) {
            System.out.println(indent + "%s".format(item._1))
          }
      }
    }

    for (group <- sortedGroups) {
      System.out.println("-----------------------------")
      printNode(group)
    }
    sortedGroups
  }

  private def printTree_Gaussian(positiveExamples: List[String], numberOfClusters: Int, data: List[(String, Double, Array[Float])], levelIndent: String = "  ") = out.eval {
    val sortedGroups: List[List[(String, Double, Array[Float])]] = clusterGaussian(numClusters = numberOfClusters,
      numIterations = 20,
      tuples = data).sortBy(_.map(_._2).sum)

    def printNode(group: List[(String, Double, Array[Float])], indent: String = ""): Unit = {
      if (group.size < (numberOfClusters * 3)) {
        for (item <- group) {
          System.out.println(indent + "%s - %.3f".format(item._1, item._2))
        }
      } else try {
        for (group <- clusterGaussian(group, 5, 20).sortBy(_.map(_._2).sum)) {
          System.out.println(indent + "-----------------------------")
          printNode(group, indent = indent + levelIndent)
        }
      } catch {
        case e: Throwable =>
          for (item <- group) {
            System.out.println(indent + "%s".format(item._1))
          }
      }
    }

    for (group <- sortedGroups) {
      System.out.println("-----------------------------")
      printNode(group)
    }
    sortedGroups
  }

  private def printTree_PIC(positiveExamples: List[String], numberOfClusters: Int, data: List[(String, Double, Array[Float])], levelIndent: String = "  ") = out.eval {
    val sortedGroups: List[List[(String, Double, Array[Float])]] = clusterPIC(numClusters = numberOfClusters,
      numIterations = 20,
      tuples = data).sortBy(_.map(_._2).sum)

    def printNode(group: List[(String, Double, Array[Float])], indent: String = ""): Unit = {
      if (group.size < (numberOfClusters * 3)) {
        for (item <- group) {
          System.out.println(indent + "%s - %.3f".format(item._1, item._2))
        }
      } else try {
        for (group <- clusterPIC(group, 5, 20).sortBy(_.map(_._2).sum)) {
          System.out.println(indent + "-----------------------------")
          printNode(group, indent = indent + levelIndent)
        }
      } catch {
        case e: Throwable =>
          for (item <- group) {
            System.out.println(indent + "%s".format(item._1))
          }
      }
    }

    for (group <- sortedGroups) {
      System.out.println("-----------------------------")
      printNode(group)
    }
    sortedGroups
  }

  private def clusterKMeans2(tuples: List[(String, Double, Array[Float])], numClusters: Int, numIterations: Int): List[List[(String, Double, Array[Float])]] = {
    val parsedData: RDD[linalg.Vector] = sc.parallelize(tuples.map(x => Vectors.dense(x._3.map(_.toDouble))))
    val clusters = KMeans.train(parsedData, numClusters, numIterations)
    tuples.groupBy(t => clusters.predict(Vectors.dense(t._3.map(_.toDouble)))).values.toList
  }

  private def clusterKMeans(tuples: List[(String, Array[Float])], numClusters: Int, numIterations: Int): List[List[(String, Array[Float])]] = {
    val parsedData: RDD[linalg.Vector] = sc.parallelize(tuples.map(x => Vectors.dense(x._2.map(_.toDouble))))
    val clusters = KMeans.train(parsedData, numClusters, numIterations)
    tuples.groupBy(t => clusters.predict(Vectors.dense(t._2.map(_.toDouble)))).values.toList
  }

  private def clusterGaussian(tuples: List[(String, Double, Array[Float])], numClusters: Int, numIterations: Int): List[List[(String, Double, Array[Float])]] = {
    val data: RDD[linalg.Vector] = sc.parallelize(tuples.map(x => Vectors.dense(x._3.map(_.toDouble))))
    val clusters = new GaussianMixture().setK(numClusters).setMaxIterations(20).run(data)
    tuples.groupBy(t => clusters.predict(Vectors.dense(t._3.map(_.toDouble)))).values.toList
  }

  private def clusterPIC(tuples: List[(String, Double, Array[Float])], numClusters: Int, numIterations: Int): List[List[(String, Double, Array[Float])]] = {
    val nameIndex: Map[String, Int] = tuples.map(_._1).zipWithIndex.toMap
    val reverseIndex = tuples.map(_._1).zipWithIndex.map(x => x._2 -> x._1).toMap
    val rows = tuples.map(x => x._1 -> x).toMap
    val primaryRdd = sc.parallelize(tuples)
    val distances = primaryRdd.cartesian(primaryRdd).map(x => {
      val ((k1, d1, v1), (k2, d2, v2)) = x
      (nameIndex(k1).toLong, nameIndex(k2).toLong, v1 ^ v2)
    })
    val model = new PowerIterationClustering()
      .setK(numClusters)
      .setMaxIterations(numIterations)
      .setInitializationMode("degree")
      .run(distances)
    model.assignments.collect().groupBy(_.cluster).mapValues(_.map(_.id).map(_.toInt).map(reverseIndex(_))).mapValues(_.map(rows(_)).toList).values.toList
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

  def loadRDD(): RDD[(String, Array[Float])] = {
    val dataFolder = "file:///H:/data_word2vec0"
    val file = "C:\\Users\\andre\\Downloads\\GoogleNews-vectors-negative300.bin.gz"
    val fileSystem = org.apache.hadoop.fs.FileSystem.get(new URI(dataFolder), sc.hadoopConfiguration)

    def loadStream(inputStream: DataInputStream): Stream[(String, Array[Float])] = {
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

    implicit def toList[T](v: RemoteIterator[T]) = {
      Stream.continually[Option[T]](if (v.hasNext) Option(v.next()) else None).takeWhile(_.isDefined).map(_.get)
    }

    val finalFolder = dataFolder + "/final"
    val cleanup = new ArrayBuffer[String]()
    if (!fileSystem.exists(new Path(finalFolder))) {
      loadStream(new DataInputStream(new GZIPInputStream(new FileInputStream(file)))).grouped(200000).zipWithIndex.map(x => {
        val dataFolder1 = dataFolder + "/" + x._2
        cleanup += dataFolder1
        val model = new Word2VecModel(x._1.toMap)
        println(model.getVectors.keys.mkString(", "))
        val rdd: RDD[(String, Array[Float])] = sc.parallelize(model.getVectors.toList, 1)
        val value = StructType(List(StructField("term", StringType), StructField("vector", ArrayType(FloatType))))
        val dataFrame = sqlContext.createDataFrame(rdd.map(x => Row(x._1, x._2)), value)
        dataFrame.write.parquet(dataFolder1)
        sqlContext.read.parquet(dataFolder1)
      }).reduce(_.union(_)).repartition(16).write.parquet(finalFolder)
    }
    for (file <- cleanup) fileSystem.delete(new Path(file), true)
    sqlContext.read.parquet(dataFolder + "/final").rdd.map(row => row.getAs[String]("term") -> row.getAs[mutable.WrappedArray.ofRef[java.lang.Float]]("vector").toArray.map(_.toFloat))
  }
}

