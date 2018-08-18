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
//import java.lang
//import java.net.URI
//import java.util.zip.GZIPInputStream
//
//import _root_.util._
//import com.simiacryptus.util.StreamNanoHTTPD
//import com.simiacryptus.util.io.HtmlNotebookOutput
//import com.simiacryptus.util.lang.CodeUtil
//import org.apache.commons.io.IOUtils
//import org.apache.commons.text.similarity.LevenshteinDistance
//import org.apache.hadoop.fs.{Path, RemoteIterator}
//import org.apache.spark.SparkConf
//import org.apache.spark.mllib.clustering.KMeans
//import org.apache.spark.mllib.linalg
//import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.rdd.RDD
//import org.apache.spark.sql.types._
//import org.apache.spark.sql.{Row, SparkSession}
//import org.apache.spark.storage.StorageLevel
//
//import scala.collection.mutable.{ArrayBuffer, WrappedArray}
//import scala.util.Try
//
//object SparkLab extends Report {
//  System.setProperty("hadoop.home.dir", "D:\\SimiaCryptus\\hadoop")
//
//  lazy val sparkConf: SparkConf = new SparkConf().setAppName(getClass.getName)
//    .setMaster("local")
//  lazy val sqlContext = SparkSession.builder().config(sparkConf).getOrCreate()
//  lazy val sc = sqlContext.sparkContext
//
//  def main(args: Array[String]): Unit = {
//    report((server, out) ⇒ args match {
//      case Array() ⇒ new SparkLab(server, out).run()
//    })
//  }
//
//}
//
//import interactive.word2vec.SparkLab._
//import interactive.word2vec.VectorUtils._
//
//class SparkLab(server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {
//
//  def inplane1(rdd: RDD[(String, Array[Float])], positiveExamples: Seq[String], n: Int = 50) = out.eval {
//    val primaryBasis = findVector(rdd, positiveExamples: _*)
//    System.out.println(s"Inplane apply $positiveExamples")
//    val metric: (Array[Float]) => Double = orthogonalDistance(primaryBasis:_*)
//    val result = rdd.map(x => {
//      val (key, vector) = x
//      (key, metric(vector), vector)
//    }).filter(!_._2.toDouble.isNaN)
//      .sortBy(_._2).take(n).toList
//    System.out.println(result.map(_._1).mkString(", "))
//    result
//  }
//
//  def orthogonalDistance(basis: Array[Float]*): (Array[Float]) => Double = {
//    val orthogonalBasis = new ArrayBuffer[Array[Float]]()
//    orthogonalBasis += basis.reduce(_ + _) / basis.size
//    basis.foreach(vector => {
//      orthogonalBasis += orthogonalBasis.tail.foldLeft(vector - orthogonalBasis.head)((l, r) => l without r).unitV
//    })
//    val orthogonalBasisArray = orthogonalBasis.toArray
//    vector => {
//      orthogonalBasisArray.tail.foldLeft(vector - orthogonalBasisArray.head)((l, r) => l without r).magnitude.toDouble
//    }
//  }
//
//  def inplane2(rdd: RDD[(String, Array[Float])], positiveExamples: Seq[_<:Seq[String]], n: Int = 50) = out.eval {
//    val map = positiveExamples.flatten.zip(findVector(rdd, positiveExamples.flatten: _*)).toMap
//    val orthogonalBasis: Seq[Array[Array[Float]]] = positiveExamples.map(positiveExamples=>{
//      val primaryBasis = positiveExamples.map(map.apply)
//      val orthogonalBasis = new ArrayBuffer[Array[Float]]()
//      orthogonalBasis += primaryBasis.reduce(_ + _) / primaryBasis.size
//      primaryBasis.foreach(vector => {
//        orthogonalBasis += orthogonalBasis.tail.foldLeft(vector - orthogonalBasis.head)((l, r) => l without r).unitV
//      })
//      orthogonalBasis.toArray
//    })
//    System.out.println(s"Inplane apply $positiveExamples")
//    val result = rdd.map(x => {
//      val (key, vector) = x
//      (key, orthogonalBasis.map(orthogonalBasis=>{
//        orthogonalBasis.tail.foldLeft(vector - orthogonalBasis.head)((l, r) => l without r).magnitude.toDouble
//      }).map(x=>x*x).reduceOption(_+_).map(Math.sqrt).get, vector)
////    }).reduce(_*_), toList)
//    }).filter(!_._2.toDouble.isNaN)
//      .sortBy(_._2).take(n).toList
//    System.out.println(result.map(_._1).mkString(", "))
//    result
//  }
//
//  def analogy(rdd: RDD[(String, Array[Float])], a: String, b: String)(c: String)(n: Int): Seq[(String, (Double, Double))] = out.eval {
//    System.out.println(s"$a is to $b as $c is to...")
//    val List(va, vb, vc) = findVector(rdd, a, b, c)
//    val orthoDist = orthogonalDistance(va,vb,vc)
//    val result = rdd.map(x => {
//      val (key, vector) = x
//      val xc = ((vc ^ vector) - (vb ^ va))
//      val xb = ((vb ^ vector) - (vc ^ va))
//      key -> (Math.sqrt(Math.pow(xc, 2) + Math.pow(xb, 2)), orthoDist(vector))
//    }).filter(!_._2._1.toDouble.isNaN)
//      .sortBy(_._2._1).take(n).map(t => (t._1, t._2)).toList
//    result.foreach(x => System.out.println("%s -> %.3f / %.3f".format(x._1, x._2._1, x._2._2)))
//    result
//  }
//
//  def membership(rdd: RDD[(String, Array[Float])], a: Seq[String]) = out.eval {
//    val vectors = a.zip(findVector(rdd, a: _*))
//    System.out.println(s"Membership analysis of group $a")
//    val result = vectors.map(x => {
//      val (key, vector) = x
//      key -> vectors.map(_._2 ^ vector).map(x => x * x).sum
//    }).filter(!_._2.toDouble.isNaN)
//      .sortBy(_._2).map(t => (t._1, t._2)).toList
//    result.foreach(x => System.out.println("%s -> %.3f".format(x._1, x._2)))
//    result
//  }
//
//  def continuum(rdd: RDD[(String, Array[Float])], a: String, b: String, n: Int = 50) = out.eval {
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
//    result.foreach(x => System.out.println(s"%s -> %.3f / %.3f".format(x._1, x._2._1, x._2._2)))
//    result
//  }
//
//  def writeModel(rdd: Seq[(String, Array[Float])], file: String) = {
//    val fileSystem = org.apache.hadoop.fs.FileSystem.get(new URI(file), sc.hadoopConfiguration)
//
//    val sortedGroups = clusterKMeans2(numClusters = 10,
//      numIterations = 20,
//      tuples = rdd.toList).zipWithIndex.flatMap(x => x._1.map(y => (y._1, y._2, x._2)))
//
//    val metadata_out = fileSystem.create(new Path(file + "/metadata.txt"), true)
//    IOUtils.write("Label\tGroup\n", metadata_out)
//    IOUtils.write(sortedGroups.map(x => List(x._1, x._3).mkString("\t")).mkString("\n"), metadata_out)
//    metadata_out.close()
//
//    val data_out = fileSystem.create(new Path(file + "/data.tsv"), true)
//    IOUtils.write(sortedGroups.map(_._2.mkString("\t")).mkString("\n"), data_out)
//    data_out.close()
//  }
//
//  def findVector(rdd: RDD[(String, Array[Float])], str: String*): List[Array[Float]] = {
//    val caseList = str.map(_.toLowerCase)
//    val value: Map[String, Array[Float]] = rdd.filter(x => caseList.contains(x._1.toLowerCase)).collect().toMap
//    str.map(key => value.getOrElse(key, value.map(t => t._1.toLowerCase -> t._2).getOrElse(key.toLowerCase, null))).toList.filter(_!=null)
//  }
//
//  def run(awaitExit: Boolean = true): Unit = {
//    CodeUtil.projectRoot = new File("../image-labs")
//    out.sourceRoot = "https://github.com/acharneski/imagelabs/tree/blog-2017-10-07/"
//
//    //val rdd = loadWord2VecRDD().mapValues(v => v.unitV).persist(StorageLevel.MEMORY_ONLY)
//    val rdd = loadFastTextRDD().mapValues(v => v.unitV).persist(StorageLevel.MEMORY_ONLY)
//
//    out.out("Our first example is to use our data solve analogy problems:")
//    analogy(rdd, "boy", "girl")("man")(50)
//    analogy(rdd, "wait", "waiting")("apply")(50)
//    analogy(rdd, "red", "green")("white")(50)
//    analogy(rdd, "England", "London")("USA")(50)
//
//    out.out("We can also generate word taxonomies given a few base words:")
//    writeModel(selectCluster_inplane(rdd = rdd, n = 500,
//      positiveExamples = List("happy", "sad", "laughing", "crying", "depressed", "excited")
//    ).map(x => (x._1, x._3)), "file:///D://SimiaCryptus/data/wordCluster_emote_500")
//
//    writeModel(selectCluster_inplane(rdd = rdd, n = 500,
//      positiveExamples = List("USA", "England", "Japan", "USSR")
//    ).map(x => (x._1, x._3)), "file:///D://SimiaCryptus/data/wordCluster_countries_500")
//
//    out.out("We can also generate a continuous spectrums from one term to another:")
//    continuum(rdd, "hell", "heaven")
//    continuum(rdd, "love", "hate")
//    continuum(rdd, "happy", "sad")
//
//    out.out("Here is a more involved term listing, which maps out the world of politics!")
//    writeModel(selectCluster_inplane(rdd = rdd, n = 1000,
//      positiveExamples = List("democrat", "republican", "conservative", "liberal")).map(x => (x._1, x._3)), "file:///D://SimiaCryptus/data/wordCluster_politics_1000")
//
//    if (awaitExit) waitForExit()
//  }
//
//  def aggregateWordList(words: List[String], maxEditDistance: Int = 2) = {
//    val lines = new ArrayBuffer[ArrayBuffer[String]]()
//    for(word <- words) {
//      lines.find(_.contains((x : String)=>LevenshteinDistance.getDefaultInstance.apply(x, word))).getOrElse({
//        val obj = new ArrayBuffer[String]()
//        lines += obj
//        obj
//      }) += word
//    }
//    lines.map(_.mkString(", ")).mkString("\n")
//  }
//
//  private def selectCluster_inplane(rdd: RDD[(String, Array[Float])], positiveExamples: List[String], n: Int = 2000) = {
//    val data = inplane1(rdd = rdd,
//      positiveExamples = positiveExamples,
//      n = n)
//    Try {
//      printTree_KMeans(data)
//    }
//    data
//  }
//
//  private def printTree_KMeans(data: List[(String, Double, Array[Float])], levelIndent: String = "  "): Unit = out.eval {
//    def printNode(group: List[(String, Double, Array[Float])], indent: String) = {
//      val axis = group.flatMap(x => group.map(y => (x, y))).maxBy(t => t._1._3 ^ t._2._3)
//      val words = group.sortBy(x => {
//        val a = x._3 ^ axis._1._3
//        val b = x._3 ^ axis._2._3
//        a / (a + b)
//      }).map(_._1)
//      System.out.println(indent + "-----------------------------")
//      System.out.println(indent + aggregateWordList(words).replaceAll("\n","\n"+indent))
//    }
//    def printTree(group: List[(String, Double, Array[Float])], indent: String = levelIndent): Unit = {
//      try {
//        if (group.size <= 13) {
//          printNode(group, indent)
//        } else if (group.size <= 100) {
//          for (group <- clusterKMeans3(tuples = group, numIterations = 20, numClusters = 3).sortBy(_.map(_._2).sum)) {
//            printTree(group, indent = indent + levelIndent)
//          }
//        } else {
//          for (group <- clusterKMeans3(tuples = group, numIterations = 20, numClusters = 7).sortBy(_.map(_._2).sum)) {
//            printTree(group, indent = indent + levelIndent)
//          }
//        }
//      } catch {
//        case e: Throwable =>
//          printNode(group, indent)
//      }
//    }
//    printTree(data)
//  }
//
//  private def clusterKMeans3(tuples: List[(String, Double, Array[Float])], numIterations: Int, numClusters:Int) = {
//    val parsedData: RDD[linalg.Vector] = sc.parallelize(tuples.map(x => Vectors.dense(x._3.map(_.toDouble))))
//    val clusters = KMeans.train(parsedData, numClusters, numIterations)
//    tuples.groupBy(t => clusters.predict(Vectors.dense(t._3.map(_.toDouble)))).values.toList
//  }
//
//  private def clusterKMeans2(tuples: List[(String, Array[Float])], numClusters: Int, numIterations: Int): List[List[(String, Array[Float])]] = {
//    val parsedData: RDD[linalg.Vector] = sc.parallelize(tuples.map(x => Vectors.dense(x._2.map(_.toDouble))))
//    val clusters = KMeans.train(parsedData, numClusters, numIterations)
//    tuples.groupBy(t => clusters.predict(Vectors.dense(t._2.map(_.toDouble)))).values.toList
//  }
//
//  class Word2VecLocalModel(keys: List[String], flatValues: Array[Float]) {
//    def get(key: String): Array[Float] = {
//      val i = keys.indexOf(key)
//      flatValues.slice(i * dimensions, (i + 1) * dimensions)
//    }
//
//    def entries: Stream[(String, Array[Float])] = {
//      keys.toStream.map(x => x -> get(x))
//    }
//
//    def dimensions = keys.size / flatValues.length
//  }
//
//  def readUntil(inputStream: InputStream, term: Char, maxLength: Int = 1024 * 8): String = {
//    var char: Char = inputStream.read().toChar
//    val str = new StringBuilder
//    while (!char.equals(term)) {
//      str.append(char)
//      assert(str.size < maxLength)
//      char = inputStream.read().toChar
//    }
//    str.toString
//  }
//
//  implicit def toList[T](v: RemoteIterator[T]) = {
//    Stream.continually[Option[T]](if (v.hasNext) Option(v.next()) else None).takeWhile(_.isDefined).map(_.get)
//  }
//
//  def loadWord2VecRDD( parquetUrl: String = "file:///H:/data_word2vec0/final",
//                       binUrl: String = "C:\\Users\\andre\\Downloads\\GoogleNews-vectors-negative300.bin.gz"
//                     ): RDD[(String, Array[Float])] = {
//    cache(parquetUrl) {
//      val inputStream = new DataInputStream(new GZIPInputStream(new BufferedInputStream(new FileInputStream(binUrl))))
//      val header = readUntil(inputStream, '\n')
//      val (records, dimensions) = header.split(" ") match {
//        case Array(records, dimensions) => (records.toInt, dimensions.toInt)
//      }
//      (0 until records).toStream.map(recordIndex => {
//        readUntil(inputStream, ' ') -> (0 until dimensions).map(dimensionIndex => {
//          java.lang.Float.intBitsToFloat(java.lang.Integer.reverseBytes(inputStream.readInt()))
//        }).toArray
//      })
//    }
//  }
//
//  def loadFastTextRDD( parquetUrl: String = "file:///H:/data_wiki.en/final",
//                       binUrl: String = "C:\\Users\\andre\\Downloads\\wiki.en\\wiki.en.vec"
//                     ): RDD[(String, Array[Float])] = {
//    cache(parquetUrl) {
//
//      val inputStream = new BufferedReader(new InputStreamReader(new FileInputStream(binUrl)))
//      val header = inputStream.readLine()
//      val (records, dimensions) = header.split(' ') match {
//        case Array(records, dimensions) => (records.toInt, dimensions.toInt)
//      }
//      (0 until records).map(recordIndex  => try {
//        val str = inputStream.readLine()
//        val line = str.split(' ')
//        line.head -> line.tail.map(_.toFloat)
//      } catch {
//        case e:Throwable=>
//          e.printStackTrace()
//          null
//      }).toStream.takeWhile(null != _)
//    }
//  }
//
//  private def cache(file: String)(data: => Stream[(String, Array[Float])],
//                                  tempFolder: String = file + "/../",
//                                  bufferSize: Int = 100000
//  ): RDD[(String, Array[Float])] = {
//    val fileSystem = org.apache.hadoop.fs.FileSystem.get(new URI(file), sc.hadoopConfiguration)
//    if (!fileSystem.exists(new Path(file))) {
//      val cleanup = new ArrayBuffer[String]()
//      data.grouped(bufferSize).zipWithIndex.map(x => {
//        val tempDest = tempFolder + x._2
//        cleanup += tempDest
//        val rdd: RDD[(String, Array[Float])] = sc.parallelize(x._1, 1)
//        val schema = StructType(List(StructField("term", StringType), StructField("toList", ArrayType(FloatType))))
//        sqlContext.createDataFrame(rdd.map(x => Row(x._1, x._2)), schema).write.parquet(tempDest)
//        sqlContext.read.parquet(tempDest)
//      }).reduce(_.union(_)).repartition(16).write.parquet(file)
//      for (file <- cleanup) fileSystem.delete(new Path(file), true)
//    }
//    sqlContext.read.parquet(file).rdd.map(row => row.getAs[String]("term") -> row.getAs[WrappedArray.ofRef[lang.Float]]("toList").toArray.map(_.toFloat))
//  }
//
//}
//
