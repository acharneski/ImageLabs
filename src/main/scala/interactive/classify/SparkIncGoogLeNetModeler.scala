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

package interactive.classify

import org.apache.hadoop.fs.{FileSystem, Path}
import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage
import java.awt.{Graphics2D, RenderingHints}
import java.io._
import java.net.URI
import java.util.concurrent.TimeUnit
import java.util.stream.Collectors
import javax.imageio.ImageIO

import _root_.util.Java8Util.cvt
import _root_.util._
import com.simiacryptus.mindseye.data.Tensor
import com.simiacryptus.mindseye.layers.NNLayer.NNExecutionContext
import com.simiacryptus.mindseye.layers.activation.{AbsActivationLayer, LinearActivationLayer, NthPowerActivationLayer, SoftmaxActivationLayer}
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext
import com.simiacryptus.mindseye.layers.cudnn.f32.PoolingLayer.PoolingMode
import com.simiacryptus.mindseye.layers.cudnn.f32._
import com.simiacryptus.mindseye.layers.loss.{EntropyLossLayer, MeanSqLossLayer}
import com.simiacryptus.mindseye.layers.media.ImgReshapeLayer
import com.simiacryptus.mindseye.layers.meta.{StdDevMetaLayer, WeightExtractor}
import com.simiacryptus.mindseye.layers.reducers.{AvgReducerLayer, SumInputsLayer, SumReducerLayer}
import com.simiacryptus.mindseye.layers.synapse.BiasLayer
import com.simiacryptus.mindseye.layers.util.MonitoringWrapper
import com.simiacryptus.mindseye.layers.{NNLayer, NNResult, SchemaComponent}
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.network.graph.{DAGNetwork, DAGNode, InnerNode}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line._
import com.simiacryptus.mindseye.opt.orient._
import com.simiacryptus.mindseye.opt.trainable._
import com.simiacryptus.util.StreamNanoHTTPD
import com.simiacryptus.util.io.{HtmlNotebookOutput, KryoUtil}
import com.simiacryptus.util.text.TableOutput
import interactive.classify.SparkIncGoogLeNetModeler.{artificialVariants, fuzz, sc, tileSize}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.JavaConverters._
import scala.util.Random

object SparkIncGoogLeNetModeler extends Report {
  System.setProperty("hadoop.home.dir", "D:\\SimiaCryptus\\hadoop")
  val dataFolder = "file:///H:/data"

  val sc = new SparkContext(new SparkConf().setAppName(getClass.getName))
  val modelName = System.getProperty("modelName", "googlenet_1")
  val tileSize: Int = 224
  val fuzz = 1e-4
  val artificialVariants = 10

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒ new SparkIncGoogLeNetModeler(source, server, out).run()
      case _ ⇒ new SparkIncGoogLeNetModeler("D:\\testImages\\256_ObjectCategories", server, out).run()
    })

  }

}

import interactive.classify.SparkIncGoogLeNetModeler._

class TrainingData(source: String) extends Serializable {

  def loadData(categoryDirectory: File, featureVector: Tensor): RDD[Array[Tensor]] = {
    val categoryName = categoryDirectory.getName
    val file = s"$dataFolder/category=$categoryName/"
    println(s"Processing $categoryName")
    if (FileSystem.get(new URI(file), sc.hadoopConfiguration).exists(new Path(file))) {
      val rdd: RDD[Array[Tensor]] = sc.objectFile(file)
      println(s"Loading $categoryName - ${rdd.count()} records from $file")
      rdd
    } else {
      val rdd = sc.parallelize(categoryDirectory.listFiles())
        .filter(_ != null)
        .filter(_.exists())
        .filter(_.length() > 0)
        .map(readImage(_))
        .filter(_ != null)
        .flatMap(variants(_, artificialVariants))
        .map(resize(_, tileSize))
        .map(toTenors(_, featureVector))
        .repartition(4)
        .persist(StorageLevel.MEMORY_AND_DISK_SER)
      println(s"Done Loading $categoryName - ${rdd.count()} records")
      rdd.saveAsObjectFile(file)
      println(s"Saved data for $categoryName to $file")
      rdd
    }
  }

  private def readImage(file: File): BufferedImage = {
    try {
      val image = ImageIO.read(file)
      if (null == image) {
        System.err.println(s"Error reading ${file.getAbsolutePath}: No image found")
      }
      image
    } catch {
      case e: Throwable =>
        System.err.println(s"Error reading ${file.getAbsolutePath}: $e")
        file.delete()
        null
    }
  }

  private def toTenors(originalRef:BufferedImage, expectedOutput: Tensor): Array[Tensor] = {
    try {
      val resized = originalRef
      if (null == resized) null
      else {
        Array(Tensor.fromRGB(resized), expectedOutput)
      }
    } catch {
      case e: Throwable =>
        e.printStackTrace(System.err)
        null
    }
  }

  def resize(originalRef:BufferedImage, tileSize:Int): BufferedImage = {
    try {
      val original = originalRef
      if (null == original) null
      else {
        val fromWidth = original.getWidth()
        val fromHeight = original.getHeight()
        val scale = tileSize.toDouble / Math.min(fromWidth, fromHeight)
        val toWidth = ((fromWidth * scale).toInt)
        val toHeight = ((fromHeight * scale).toInt)
        val resized = new BufferedImage(tileSize, tileSize, BufferedImage.TYPE_INT_ARGB)
        val graphics = resized.getGraphics.asInstanceOf[Graphics2D]
        graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
        if (toWidth < toHeight) {
          graphics.drawImage(original, 0, (toWidth - toHeight) / 2, toWidth, toHeight, null)
        } else {
          graphics.drawImage(original, (toHeight - toWidth) / 2, 0, toWidth, toHeight, null)
        }
        resized
      }
    } catch {
      case e: Throwable =>
        e.printStackTrace(System.err)
        null
    }
  }

  def variants(imageFn: BufferedImage, items: Int): Stream[BufferedImage] = {
    Stream.continually({
      val sy = 1.05 + Random.nextDouble() * 0.05
      val sx = 1.05 + Random.nextDouble() * 0.05
      val theta = (Random.nextDouble() - 0.5) * 0.2
      val image = imageFn
      if(null == image) return null
      val resized = new BufferedImage(image.getWidth, image.getHeight, BufferedImage.TYPE_INT_ARGB)
      val graphics = resized.getGraphics.asInstanceOf[Graphics2D]
      val transform = AffineTransform.getScaleInstance(sx,sy)
      transform.concatenate(AffineTransform.getRotateInstance(theta))
      transform.concatenate(AffineTransform.getTranslateInstance(-0.02 * image.getWidth, -0.02 * image.getHeight))
      graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
      graphics.drawImage(image, transform, null)
      resized
    }).take(items)
  }

  def toOutNDArray(max: Int, out: Int*): Tensor = {
    val ndArray = new Tensor(max)
    for (i <- 0 until max) ndArray.set(i, fuzz)
    out.foreach(out=>ndArray.set(out, 1 - (fuzz * (max - 1))))
    ndArray
  }

  def selectCategories(numCategories: Int) = {
    Random.shuffle(data.toList).take(numCategories).toMap
  }

  val categoryDirs: Stream[File] = Random.shuffle(new File(source).listFiles().toStream)
  val categoryList: Array[String] = categoryDirs.map((categoryDirectory: File) ⇒ {
    categoryDirectory.getName.split('.').last
  }).sorted.toArray
  val categoryMap: Map[String, Int] = categoryList.zipWithIndex.toMap
  val data: Map[String, RDD[Array[Tensor]]] = categoryDirs
    .map((categoryDirectory: File) ⇒ {
      val categoryName = categoryDirectory.getName.split('.').last
      categoryName -> loadData(categoryDirectory, toOutNDArray(categoryMap.size, categoryMap(categoryName)))
    }).toMap
}

class SparkIncGoogLeNetModeler(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  val data = {
    out.p("Loading data from " + source)
    val data = new TrainingData(source)
    out.eval {
      data.data.mapValues(_.count())
    }
    out.eval {
      val tuples: List[(String, RDD[Array[Tensor]])] = Random.shuffle(data.data.take(3).toList).take(3)
      TableOutput.create(tuples.flatMap(x => x._2.takeSample(false, 3).toList).par.map(x ⇒ {
        Map[String, AnyRef](
          "Image" → out.image(x(0).toRgbImage(), x(1).toString),
          "Classification" → x(1)
        ).asJava
      }).toArray: _*)
    }
    out.p("Loading data complete")
    data
  }


  def run(awaitExit: Boolean = true): Unit = {
    recordMetrics = false
    defineHeader()
    declareTestHandler()
    out.h1("Incremental GoogLeNet Builder with Adversarial Images")
    out.out("<hr/>")
    case class Parameters(initMinutes: Int, ganImages: Int, trainMinutes: Int, imagesPerIterationTrain: Int, imagesPerIterationInit: Int)
    val p = "daytime" match {
      case "smoke" =>
        new Parameters(
          initMinutes = 1,
          ganImages = 1,
          trainMinutes = 1,
          imagesPerIterationTrain = 100,
          imagesPerIterationInit = 100
        )
      case "daytime" =>
        new Parameters(
          initMinutes = 120,
          ganImages = 5,
          trainMinutes = 180,
          imagesPerIterationTrain = 1000,
          imagesPerIterationInit = 20
        )
    }
    val set1 = data.selectCategories(10).map(_._1).toSet //Set("chimp", "owl", "chess-board")
    val set2 = data.selectCategories(20).map(_._1).toSet //Set("owl", "teddy-bear", "zebra", "chess-board", "binoculars", "bonsai-101", "brain-101")
    require(set1.forall(categories.contains))
    require(set2.forall(categories.contains))
    val sourceClass = "chimp"
    val targetClass = "owl"
    out.h2("Layer Set 1")
    step_Generate()
    step_AddLayer1(trainingMin = p.initMinutes, sampleSize = p.imagesPerIterationInit)
    step_Train(trainingMin = p.trainMinutes, categories = set1, sampleSize = p.imagesPerIterationTrain, iterationsPerSample = 10, ganImages = p.ganImages)
    step_Train(trainingMin = p.trainMinutes, categories = set2, sampleSize = p.imagesPerIterationTrain, iterationsPerSample = 10, ganImages = p.ganImages)
    step_GAN(imageCount = p.ganImages,sourceCategory = sourceClass,targetCategory = targetClass)
    out.h2("Layer Set 2")
    step_AddLayer2(trainingMin = p.initMinutes, sampleSize = p.imagesPerIterationInit)
    step_Train(trainingMin = p.trainMinutes, categories = set1, sampleSize = p.imagesPerIterationTrain, iterationsPerSample = 10, ganImages = p.ganImages)
    step_Train(trainingMin = p.trainMinutes, categories = set2, sampleSize = p.imagesPerIterationTrain, iterationsPerSample = 10, ganImages = p.ganImages)
    step_GAN(imageCount = p.ganImages,sourceCategory = sourceClass,targetCategory = targetClass)
    out.h2("Layer Set 3")
    step_AddLayer3(trainingMin = p.initMinutes, sampleSize = p.imagesPerIterationInit)
    step_Train(trainingMin = p.trainMinutes, categories = set1, sampleSize = p.imagesPerIterationTrain, iterationsPerSample = 10, ganImages = p.ganImages)
    step_Train(trainingMin = p.trainMinutes, categories = set2, sampleSize = p.imagesPerIterationTrain, iterationsPerSample = 10, ganImages = p.ganImages)
    step_GAN(imageCount = p.ganImages,sourceCategory = sourceClass,targetCategory = targetClass)
    out.h2("Layer Set 4")
    step_AddLayer4(trainingMin = p.initMinutes, sampleSize = p.imagesPerIterationInit)
    step_Train(trainingMin = p.trainMinutes, categories = set1, sampleSize = p.imagesPerIterationTrain, iterationsPerSample = 10, ganImages = p.ganImages)
    step_Train(trainingMin = p.trainMinutes, categories = set2, sampleSize = p.imagesPerIterationTrain, iterationsPerSample = 10, ganImages = p.ganImages)
    step_GAN(imageCount = p.ganImages,sourceCategory = sourceClass,targetCategory = targetClass)
    out.h2("Layer Set 5")
    step_AddLayer5(trainingMin = p.initMinutes, sampleSize = p.imagesPerIterationInit)
    step_Train(trainingMin = p.trainMinutes, categories = set1, sampleSize = p.imagesPerIterationTrain, iterationsPerSample = 10, ganImages = p.ganImages)
    step_Train(trainingMin = p.trainMinutes, categories = set2, sampleSize = p.imagesPerIterationTrain, iterationsPerSample = 10, ganImages = p.ganImages)
    step_GAN(imageCount = p.ganImages,sourceCategory = sourceClass,targetCategory = targetClass)
    out.out("<hr/>")
    if (awaitExit) waitForExit()
  }

  def step_Generate() = phase({
    new PipelineNetwork(2)
  }, (_: NNLayer) ⇒ {
    // Do Nothing
  }: Unit, modelName)


  def newInceptionLayer(layerName : String, inputBands: Int, bands1x1: Int, bands3x1: Int, bands1x3: Int, bands5x1: Int, bands1x5: Int, bandsPooling: Int): NNLayer = {
    val network = new PipelineNetwork()
    newInceptionLayer(network, inputBands = inputBands, layerName = layerName, head = network.getHead,
      bands1x1 = bands1x1, bands1x3 = bands1x3, bands1x5 = bands1x5, bands3x1 = bands3x1,
      bands5x1 = bands5x1, bandsPooling = bandsPooling)
    network
  }
  def newInceptionLayer(network : PipelineNetwork, layerName : String, head: DAGNode, inputBands: Int, bands1x1: Int, bands3x1: Int, bands1x3: Int, bands5x1: Int, bands1x5: Int, bandsPooling: Int): DAGNode = {
    var conv1a: Double = 0.01
    var conv1b: Double = 0.01
    var conv3a: Double = 0.01
    var conv3b: Double = 0.01
    var conv5a: Double = 0.01
    var conv5b: Double = 0.01
    network.add(new ImgConcatLayer(),
      network.addAll(head,
        new ConvolutionLayer(1, 1, inputBands, bands1x1).setWeightsLog(conv1a).setName("conv_1x1_" + layerName),
        new ImgBandBiasLayer(bands1x1).setName("bias_1x1_" + layerName),
        new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_1x1_" + layerName)),
      network.addAll(head,
        new ConvolutionLayer(1, 1, inputBands, bands3x1).setWeightsLog(conv3a).setName("conv_3x1_" + layerName),
        new ImgBandBiasLayer(bands3x1).setName("bias_3x1_" + layerName),
        new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_3x1_" + layerName),
        new ConvolutionLayer(3, 3, bands3x1, bands1x3).setWeightsLog(conv3b).setName("conv_1x3_" + layerName),
        new ImgBandBiasLayer(bands1x3).setName("bias_1x3_" + layerName),
        new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_1x3_" + layerName)),
      network.addAll(head,
        new ConvolutionLayer(1, 1, inputBands, bands5x1).setWeightsLog(conv5a).setName("conv_5x1_" + layerName),
        new ImgBandBiasLayer(bands5x1).setName("bias_5x1_" + layerName),
        new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_5x1_" + layerName),
        new ConvolutionLayer(5, 5, bands5x1, bands1x5).setWeightsLog(conv5b).setName("conv_1x5_" + layerName),
        new ImgBandBiasLayer(bands1x5).setName("bias_1x5_" + layerName),
        new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_1x5_" + layerName)),
      network.addAll(head,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1).setName("pool_" + layerName),
        new ConvolutionLayer(1, 1, inputBands, bandsPooling).setWeightsLog(conv1b).setName("conv_pool_" + layerName),
        new ImgBandBiasLayer(bandsPooling).setName("bias_pool_" + layerName),
        new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_pool_" + layerName)))
  }

  def step_AddLayer1(trainingMin: Int, sampleSize: Int, categories: List[String]): Any = phase(modelName, (model: NNLayer) ⇒
    {
      val sourceNetwork = model.asInstanceOf[PipelineNetwork]
      val priorFeaturesNode = Option(sourceNetwork.getByLabel("features")).getOrElse(sourceNetwork.getHead)
      val trainingData = categories.map(c => data.data(c)).reduce(_.union(_))
      model.asInstanceOf[DAGNetwork].visitLayers((layer:NNLayer)=>if(layer.isInstanceOf[SchemaComponent]) {
        layer.asInstanceOf[SchemaComponent].setSchema(categories.toArray:_*)
      } : Unit)
      addLayer(
        rdd = preprocessFeatures(sourceNetwork, priorFeaturesNode, trainingData),
        sourceNetwork = sourceNetwork,
        priorFeaturesNode = priorFeaturesNode,
        additionalLayer = new PipelineNetwork(
          new ConvolutionLayer(7, 7, 3, 64).setWeightsLog(-4).setStrideXY(2, 2).setName("conv_1"),
          new ImgBandBiasLayer(64).setName("bias_1"),
          new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_1"),
          new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1).setName("pool_1")
        ), reconstructionLayer = new PipelineNetwork(
          new ConvolutionLayer(7, 7, 64, 48).setWeightsLog(-4),
          new ImgReshapeLayer(4, 4, true)
        ), trainingMin = trainingMin, sampleSize = sampleSize)
      sourceNetwork
    }: Unit, modelName)

  def step_AddLayer1(trainingMin: Int = 15, sampleSize: Int = 100, numberOfCategories: Int = 5): Unit = {
    step_AddLayer1(trainingMin = trainingMin, sampleSize = sampleSize, categories = data.selectCategories(numberOfCategories).keys.toList)
  }

  def step_AddLayer2(trainingMin: Int = 15, sampleSize: Int = 100, numberOfCategories: Int = 5): Unit = {
    step_AddLayer2(trainingMin = trainingMin, sampleSize = sampleSize, categories = data.selectCategories(numberOfCategories).keys.toList)
  }

  def step_AddLayer2(trainingMin: Int, sampleSize: Int, categories: List[String]): Any = phase(modelName, (model: NNLayer) ⇒
    {
      val sourceNetwork = model.asInstanceOf[PipelineNetwork]
      val priorFeaturesNode = sourceNetwork.getByLabel("features")
      sourceNetwork.visitLayers((layer:NNLayer)=>if(layer.isInstanceOf[SchemaComponent]) {
        layer.asInstanceOf[SchemaComponent].setSchema(categories.toArray:_*)
      } : Unit)
      addLayer(
        rdd = preprocessFeatures(sourceNetwork, priorFeaturesNode, categories.map(c => data.data(c)).reduce(_.union(_))),
        sourceNetwork = sourceNetwork,
        priorFeaturesNode = priorFeaturesNode,
        additionalLayer = new PipelineNetwork(
          new ConvolutionLayer(1, 1, 64, 64).setWeightsLog(-4).setName("conv_2"),
          new ImgBandBiasLayer(64).setName("bias_2"),
          new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_2"),
          new ConvolutionLayer(3, 3, 64, 192).setWeightsLog(-4).setName("conv_3"),
          new ImgBandBiasLayer(192).setName("bias_3"),
          new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_3"),
          new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1).setName("pool_3")
        ), reconstructionLayer = new PipelineNetwork(
          new ConvolutionLayer(3, 3, 192, 64*4).setWeightsLog(-4),
          new ImgReshapeLayer(2, 2, true)
        ), trainingMin = trainingMin, sampleSize = sampleSize)
      sourceNetwork
    }: Unit, modelName)


  def step_AddLayer3(trainingMin: Int = 15, sampleSize: Int = 100, numberOfCategories: Int = 5): Unit = {
    step_AddLayer3(trainingMin = trainingMin, sampleSize = sampleSize, categories = data.selectCategories(numberOfCategories).keys.toList)
  }

  def step_AddLayer3(trainingMin: Int, sampleSize: Int, categories: List[String]): Any = phase(modelName, (model: NNLayer) ⇒
    {
      val sourceNetwork = model.asInstanceOf[PipelineNetwork]
      val priorFeaturesNode = sourceNetwork.getByLabel("features")
      sourceNetwork.visitLayers((layer:NNLayer)=>if(layer.isInstanceOf[SchemaComponent]) {
        layer.asInstanceOf[SchemaComponent].setSchema(categories.toArray:_*)
      } : Unit)
      addLayer(
        rdd = preprocessFeatures(sourceNetwork, priorFeaturesNode, categories.map(c => data.data(c)).reduce(_.union(_))),
        sourceNetwork = sourceNetwork,
        priorFeaturesNode = priorFeaturesNode,
        additionalLayer = new PipelineNetwork(
          newInceptionLayer(layerName = "3a", inputBands = 192, bands1x1 = 64, bands3x1 = 96, bands1x3 = 128, bands5x1 = 16, bands1x5 = 32, bandsPooling = 32),
          newInceptionLayer(layerName = "3b", inputBands = 256, bands1x1 = 128, bands3x1 = 128, bands1x3 = 192, bands5x1 = 32, bands1x5 = 96, bandsPooling = 64),
          new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1).setName("pool_4")
        ), reconstructionLayer = new PipelineNetwork(
          new ConvolutionLayer(3, 3, 480, 192*4).setWeightsLog(-4),
          new ImgReshapeLayer(2, 2, true)
        ), trainingMin = trainingMin, sampleSize = sampleSize)
      sourceNetwork
    }: Unit, modelName)


  def step_AddLayer4(trainingMin: Int = 15, sampleSize: Int = 100, numberOfCategories: Int = 5): Unit = {
    step_AddLayer4(trainingMin = trainingMin, sampleSize = sampleSize, categories = data.selectCategories(numberOfCategories).keys.toList)
  }

  def step_AddLayer4(trainingMin: Int, sampleSize: Int, categories: List[String]): Any = phase(modelName, (model: NNLayer) ⇒
    {
      val sourceNetwork = model.asInstanceOf[PipelineNetwork]
      val priorFeaturesNode = sourceNetwork.getByLabel("features")
      sourceNetwork.visitLayers((layer:NNLayer)=>if(layer.isInstanceOf[SchemaComponent]) {
        layer.asInstanceOf[SchemaComponent].setSchema(categories.toArray:_*)
      } : Unit)
      addLayer(
        rdd = preprocessFeatures(sourceNetwork, priorFeaturesNode, categories.map(c => data.data(c)).reduce(_.union(_))),
        sourceNetwork = sourceNetwork,
        priorFeaturesNode = priorFeaturesNode,
        additionalLayer = new PipelineNetwork(
          newInceptionLayer(layerName = "4a", inputBands = 480, bands1x1 = 192, bands3x1 = 96, bands1x3 = 208, bands5x1 = 16, bands1x5 = 48, bandsPooling = 64),
          newInceptionLayer(layerName = "4b", inputBands = 512, bands1x1 = 160, bands3x1 = 112, bands1x3 = 224, bands5x1 = 24, bands1x5 = 64, bandsPooling = 64),
          newInceptionLayer(layerName = "4c", inputBands = 512, bands1x1 = 128, bands3x1 = 128, bands1x3 = 256, bands5x1 = 24, bands1x5 = 64, bandsPooling = 64),
          newInceptionLayer(layerName = "4d", inputBands = 512, bands1x1 = 112, bands3x1 = 144, bands1x3 = 288, bands5x1 = 32, bands1x5 = 64, bandsPooling = 64),
          newInceptionLayer(layerName = "4e", inputBands = 528, bands1x1 = 256, bands3x1 = 160, bands1x3 = 320, bands5x1 = 32, bands1x5 = 128, bandsPooling = 128),
          new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1).setName("pool_5")
        ), reconstructionLayer = new PipelineNetwork(
          new ConvolutionLayer(3, 3, 832, 480 * 4).setWeightsLog(-4),
          new ImgReshapeLayer(2, 2, true)
        ), trainingMin = trainingMin, sampleSize = sampleSize)
      sourceNetwork
    }: Unit, modelName)

  def step_AddLayer5(trainingMin: Int = 15, sampleSize: Int = 100, numberOfCategories: Int = 5): Unit = {
    step_AddLayer5(trainingMin = trainingMin, sampleSize = sampleSize, categories = data.selectCategories(numberOfCategories).keys.toList)
  }

  def step_AddLayer5(trainingMin: Int, sampleSize: Int, categories: List[String]): Any = phase(modelName, (model: NNLayer) ⇒
    {
      val sourceNetwork = model.asInstanceOf[PipelineNetwork]
      val priorFeaturesNode = sourceNetwork.getByLabel("features")
      sourceNetwork.visitLayers((layer:NNLayer)=>if(layer.isInstanceOf[SchemaComponent]) {
        layer.asInstanceOf[SchemaComponent].setSchema(categories.toArray:_*)
      } : Unit)
      addLayer(
        rdd = preprocessFeatures(sourceNetwork, priorFeaturesNode, categories.map(c => data.data(c)).reduce(_.union(_))),
        sourceNetwork = sourceNetwork,
        priorFeaturesNode = priorFeaturesNode,
        additionalLayer = new PipelineNetwork(
          newInceptionLayer(layerName = "5a", inputBands = 832, bands1x1 = 256, bands3x1 = 160, bands1x3 = 320, bands5x1 = 32, bands1x5 = 128, bandsPooling = 128),
          newInceptionLayer(layerName = "5b", inputBands = 832, bands1x1 = 384, bands3x1 = 192, bands1x3 = 384, bands5x1 = 48, bands1x5 = 128, bandsPooling = 128),
          new PoolingLayer().setWindowXY(7, 7).setStrideXY(1, 1).setPaddingXY(0, 0).setMode(PoolingMode.Avg).setName("pool_6")
        ), reconstructionLayer = new PipelineNetwork(
          new ConvolutionLayer(3, 3, 1024, 832 * 7 * 7).setWeightsLog(-4),
          new ImgReshapeLayer(7, 7, true)
        ), trainingMin = trainingMin, sampleSize = sampleSize)
      sourceNetwork
    }: Unit, modelName)

  private def preprocessFeatures(sourceNetwork: PipelineNetwork, priorFeaturesNode: DAGNode, trainingData: RDD[Array[Tensor]]): RDD[Array[Tensor]] = {
    trainingData.map(inputs=>{
      val gpu = Random.shuffle(CudaExecutionContext.gpuContexts.getAll.asScala).head
      val array: Array[Tensor] = priorFeaturesNode.get(gpu, sourceNetwork.buildExeCtx(
        NNResult.batchResultArray(Array(inputs)): _*)).getData.stream().collect(Collectors.toList()).asScala.toArray
      array.take(1) ++ inputs.tail
    })
  }



  private def addLayer(rdd: RDD[Array[Tensor]], sourceNetwork: PipelineNetwork, priorFeaturesNode: DAGNode, additionalLayer: NNLayer,
                       reconstructionLayer: PipelineNetwork,
                       trainingMin: Int, sampleSize: Int,
                       featuresLabel:String = "features"): DAGNode =
  {
    val numberOfCategories = rdd.take(1).head(1).dim()
    val newFeatureDimensions: Array[Int] = CudaExecutionContext.gpuContexts.map((cuda:CudaExecutionContext)=>additionalLayer.eval(cuda, rdd.take(1).head.head).getData.get(0).getDimensions)
    val trainingNetwork = new PipelineNetwork(2)
    val featuresNode = trainingNetwork.add(featuresLabel, additionalLayer, trainingNetwork.getInput(0))
    val dropoutNode = trainingNetwork.add(new DropoutNoiseLayer().setValue(0.2), featuresNode)
    trainingNetwork.add(
      new SumInputsLayer(),
      // Features should be relevant - predict the class given a final linear/softmax transform
      Array(
        trainingNetwork.add(new LinearActivationLayer().setScale(0.1).freeze(),
          trainingNetwork.add(new EntropyLossLayer(),
            trainingNetwork.add(new SoftmaxActivationLayer(),
              trainingNetwork.add(new BandPoolingLayer().setMode(BandPoolingLayer.PoolingMode.Avg),
                trainingNetwork.add(new ConvolutionLayer(1, 1, newFeatureDimensions(2), numberOfCategories, true).setWeightsLog(-4),
                  dropoutNode))
            ),
            trainingNetwork.getInput(1)
          )
        ),
        // Features should be able to reconstruct input - Preserve information
        trainingNetwork.add(new LinearActivationLayer().setScale(0.1).freeze(),
          trainingNetwork.add(new NthPowerActivationLayer().setPower(0.5).freeze(),
            trainingNetwork.add(new MeanSqLossLayer(),
              trainingNetwork.add(reconstructionLayer, dropoutNode),
              trainingNetwork.getInput(0)
            )
          )
        ),
        // Features signal should target a uniform magnitude to balance the network
        trainingNetwork.add(new LinearActivationLayer().setScale(1.0).freeze(),
          trainingNetwork.add(new AbsActivationLayer(),
            trainingNetwork.add(new LinearActivationLayer().setBias(-1).freeze(),
              trainingNetwork.add(new AvgReducerLayer(),
                trainingNetwork.add(new StdDevMetaLayer(), featuresNode))
            )
          )
        )
      ).map(node=>trainingNetwork.add(new NthPowerActivationLayer().setPower(2).freeze(),node)):_*
    )
    require(null != PipelineNetwork.fromJson(trainingNetwork.getJson))

    out.h1("Training New Layer")
    monitor.clear()
    model = trainingNetwork
    addMonitoring(model.asInstanceOf[DAGNetwork])
    out.eval {
      var inner: Trainable = new SparkTrainable(rdd, trainingNetwork, sampleSize).setPartitions(1).cached()
      val trainer = new IterativeTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(trainingMin, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(20)
      trainer.setOrientation(new QQN() {
        override def reset(): Unit = {
          model.asInstanceOf[DAGNetwork].visitLayers(Java8Util.cvt(layer => layer match {
            case layer: DropoutNoiseLayer => layer.shuffle()
            case _ =>
          }))
          super.reset()
        }
      }.setMinHistory(4).setMaxHistory(20))
      trainer.setLineSearchFactory(Java8Util.cvt((s: String) ⇒ (s match {
        case s if s.contains("QQN") ⇒ new ArmijoWolfeSearch().setAlpha(1.0)
        case _ ⇒ new ArmijoWolfeSearch().setAlpha(1e-2)
      })))
      trainer.setTerminateThreshold(0.0)
      trainer
    } run

    removeMonitoring(model.asInstanceOf[DAGNetwork])

    model = sourceNetwork

    val inputNode = Option(sourceNetwork.getByLabel(featuresLabel)).getOrElse(sourceNetwork.getInput(0))
    sourceNetwork.add(new EntropyLossLayer(),
      sourceNetwork.add("classify", new SoftmaxActivationLayer(),
        sourceNetwork.add(new SchemaBiasLayer(),
          sourceNetwork.add(new BandPoolingLayer().setMode(BandPoolingLayer.PoolingMode.Avg),
            sourceNetwork.add(new SchemaOutputLayer(newFeatureDimensions(2), -4).setSchema(data.categoryList:_*),
              sourceNetwork.add(new DropoutNoiseLayer(),
                sourceNetwork.add(featuresLabel, additionalLayer,
                  inputNode)))))
      ),
      sourceNetwork.getInput(1)
    )
  }

  final def addMonitoring(model: DAGNetwork) : Unit = {
    model.getNodes.asScala.foreach({
      case node: InnerNode =>
        node.getLayer() match {
          case _:MonitoringWrapper => // Ignore
          case layer: DAGNetwork =>
            addMonitoring(layer.asInstanceOf[DAGNetwork])
            node.setLayer(new MonitoringWrapper(layer).addTo(monitoringRoot))
          case layer =>
            node.setLayer(new MonitoringWrapper(layer).addTo(monitoringRoot))
        }
      case _ =>
    })
  }

  final def removeMonitoring(model: DAGNetwork) : Unit = {
    model.getNodes.asScala.foreach({
      case node: InnerNode =>
        node.getLayer() match {
          case layer : MonitoringWrapper => // Ignore
            node.setLayer(layer.getInner)
          case layer: DAGNetwork =>
            removeMonitoring(layer.asInstanceOf[DAGNetwork])
          case layer =>
        }
      case _ =>
    })
  }

  def step_Train(trainingMin: Int = 15, numberOfCategories: Int = 2, sampleSize: Int = 250, iterationsPerSample: Int = 5): Unit = {
    step_Train(trainingMin = trainingMin, categories = data.selectCategories(numberOfCategories).keys.toSet, sampleSize = sampleSize, iterationsPerSample = iterationsPerSample, 5)

  }

  def step_Train(trainingMin: Int, categories: Set[String], sampleSize: Int, iterationsPerSample: Int, ganImages: Int): Unit = {
    monitor.clear()
    val categoryArray = categories.toArray
    val categoryIndices = categoryArray.zipWithIndex.toMap
    val selectedCategories = categories.map(e=>{
      e -> data.data(e).map(_.take(1) ++ Array(data.toOutNDArray(categoryIndices.size, categoryIndices(e))))
    }).toMap
    phase(modelName, (model: NNLayer) ⇒ {
      out.h1("Integration Training")
      model.asInstanceOf[DAGNetwork].visitLayers((layer:NNLayer)=>if(layer.isInstanceOf[SchemaComponent]) {
        layer.asInstanceOf[SchemaComponent].setSchema(categoryArray:_*)
      } : Unit)
      out.eval {
        val rdd = selectedCategories.values.reduce(_.union(_)).cache()
        var inner: Trainable = new SparkTrainable(rdd, model, sampleSize).setPartitions(1).cached()
        val trainer = new IterativeTrainer(inner)
        trainer.setMonitor(monitor)
        trainer.setTimeout(trainingMin, TimeUnit.MINUTES)
        trainer.setIterationsPerSample(iterationsPerSample)
        trainer.setOrientation(new QQN() {
          override def reset(): Unit = {
            model.asInstanceOf[DAGNetwork].visitLayers(Java8Util.cvt(layer => layer match {
              case layer: DropoutNoiseLayer => layer.shuffle()
              case _ =>
            }))
            super.reset()
          }
        }.setMinHistory(4).setMaxHistory(20))
        trainer.setLineSearchFactory(Java8Util.cvt((s: String) ⇒ (s match {
          case s if s.contains("QQN") ⇒ new ArmijoWolfeSearch().setAlpha(1e-5)
          case _ ⇒ new ArmijoWolfeSearch().setAlpha(1e-5)
        })))
        trainer.setTerminateThreshold(0.0)
        trainer
      } run
    }: Unit, modelName)
    val thisModel = model
    (for (_ <- 1 to 2) yield Random.shuffle(selectedCategories.keys).take(2).toList).distinct.foreach {
      case Seq(from: String, to: String) =>
        gan(out, thisModel)(imageCount = ganImages, sourceCategory = from, targetCategory = to)
    }
  }

  def step_GAN(imageCount: Int = 10, sourceCategory: String = "fire-hydrant", targetCategory: String = "bear") = phase(modelName, (model: NNLayer) ⇒ {
    gan(out, model)(imageCount = imageCount, sourceCategory = sourceCategory, targetCategory = targetCategory)
  }: Unit, null)

  def gan(out: HtmlNotebookOutput with ScalaNotebookOutput, model: NNLayer)
           (imageCount: Int = 1, sourceCategory: String = "fire-hydrant", targetCategory: String = "bear") = {
    assert(null != model)
    model.asInstanceOf[DAGNetwork].visitLayers(Java8Util.cvt(layer => layer match {
      case layer: DropoutNoiseLayer => layer.setValue(0.0).shuffle()
      case _ =>
    }))
    val categoryArray = Array(sourceCategory, targetCategory)
    val categoryIndices = categoryArray.zipWithIndex.toMap
    val sourceClassId = categoryIndices(sourceCategory)
    val targetClassId = categoryIndices(targetCategory)
    out.h1(s"GAN Images Generation: $sourceCategory to $targetCategory (with 3x3 convolution)")
    val sourceClass = data.toOutNDArray(categoryArray.length, sourceClassId)
    val targetClass = data.toOutNDArray(categoryArray.length, targetClassId)
    model.asInstanceOf[DAGNetwork].visitLayers((layer:NNLayer)=>if(layer.isInstanceOf[SchemaComponent]) {
      layer.asInstanceOf[SchemaComponent].setSchema(categoryArray:_*)
    } : Unit)
    val imagesInput = data.data(sourceCategory).take(imageCount)
    out.eval {
      TableOutput.create(imagesInput.grouped(1).map(group => {
        val adversarialData: Array[Array[Tensor]] = group.map(_.take(1) ++ Array(targetClass)).toArray
        Map[String, AnyRef](
          "Original Image" → out.image(adversarialData.head.head.toRgbImage, ""),
          "Adversarial A" → out.image(ganA(model, adversarialData).toRgbImage, ""),
          //"Adversarial B" → out.image(ganB(model, adversarialData).toRgbImage, ""),
          "Adversarial C" → out.image(ganC(model, adversarialData).toRgbImage, "")
          //"Adversarial D" → out.image(ganD(model, adversarialData).toRgbImage, "")
        ).asJava
      }).toArray: _*)
    }
    summarizeHistory(out)
    monitor.clear()
    this.model = null
  }

  private def ganC(model: NNLayer, adversarialData: Array[Array[Tensor]]): Tensor = {
    val adaptationLayer = new BiasLayer(adversarialData.head.head.getDimensions:_*)
    val trainingNetwork = new PipelineNetwork(2)
    trainingNetwork.add(adaptationLayer)
    val pipelineNetwork = KryoUtil.kryo().copy(model).freeze().asInstanceOf[PipelineNetwork]
    pipelineNetwork.setHead(pipelineNetwork.getByLabel("classify")).removeLastInput()
    trainingNetwork.add(pipelineNetwork)
    trainingNetwork.add(new EntropyLossLayer(), trainingNetwork.getHead, trainingNetwork.getInput(1))
    this.model = trainingNetwork
    var inner: Trainable = new ArrayTrainable(adversarialData, trainingNetwork)
    val trainer = new IterativeTrainer(inner)
    trainer.setMonitor(monitor)
    trainer.setTimeout(1, TimeUnit.MINUTES)
    trainer.setOrientation(new GradientDescent)
    trainer.setLineSearchFactory(Java8Util.cvt((s: String) ⇒ new ArmijoWolfeSearch().setMaxAlpha(1e8): LineSearchStrategy))
    trainer.setTerminateThreshold(0.01)
    trainer.run()
    val evalNetwork = new PipelineNetwork()
    evalNetwork.add(adaptationLayer)
    val adversarialImage = evalNetwork.eval(new NNExecutionContext {}, adversarialData.head.head).getData.get(0)
    adversarialImage
  }

  private def ganA(model: NNLayer, adversarialData: Array[Array[Tensor]]): Tensor = {
    val adaptationLayer = new BiasLayer(adversarialData.head.head.getDimensions:_*)
    val trainingNetwork = new PipelineNetwork(2)
    trainingNetwork.add(adaptationLayer)
    val pipelineNetwork = KryoUtil.kryo().copy(model).freeze().asInstanceOf[PipelineNetwork]
    pipelineNetwork.setHead(pipelineNetwork.getByLabel("classify")).removeLastInput()
    trainingNetwork.add(pipelineNetwork)
    trainingNetwork.add(new EntropyLossLayer(), trainingNetwork.getHead, trainingNetwork.getInput(1))
    this.model = trainingNetwork
    var inner: Trainable = new ArrayTrainable(adversarialData, trainingNetwork)
    val trainer = new IterativeTrainer(inner)
    trainer.setMonitor(monitor)
    trainer.setTimeout(1, TimeUnit.MINUTES)
    trainer.setOrientation(new GradientDescent)
    trainer.setLineSearchFactory(Java8Util.cvt((_: String) ⇒ new QuadraticSearch))
    trainer.setTerminateThreshold(0.01)
    trainer.run()
    val evalNetwork = new PipelineNetwork()
    evalNetwork.add(adaptationLayer)
    val adversarialImage = evalNetwork.eval(new NNExecutionContext {}, adversarialData.head.head).getData.get(0)
    adversarialImage
  }

  private def ganB(model: NNLayer, adversarialData: Array[Array[Tensor]]): Tensor = {
    val adaptationLayer = new ConvolutionLayer(3, 3, 3, 3)
    for (i <- 0 until 3) adaptationLayer.filter.set(Array(1, 1, 4 * i), 1.0)
    val trainingNetwork = new PipelineNetwork(2)
    trainingNetwork.add(adaptationLayer)
    val pipelineNetwork = KryoUtil.kryo().copy(model).freeze().asInstanceOf[PipelineNetwork]
    pipelineNetwork.setHead(pipelineNetwork.getByLabel("classify")).removeLastInput()
    trainingNetwork.add(pipelineNetwork)

    trainingNetwork.add(new EntropyLossLayer(), trainingNetwork.getHead, trainingNetwork.getInput(1))
    this.model = trainingNetwork
    var inner: Trainable = new ArrayTrainable(adversarialData, trainingNetwork)
    val trainer = new IterativeTrainer(inner)
    trainer.setMonitor(monitor)
    trainer.setTimeout(1, TimeUnit.MINUTES)
    trainer.setOrientation(new GradientDescent)
    //trainer.setLineSearchFactory(Java8Util.cvt((s: String) ⇒ new ArmijoWolfeSearch().setMaxAlpha(1e8): LineSearchStrategy))
    trainer.setLineSearchFactory(Java8Util.cvt((_: String) ⇒ new QuadraticSearch))
    trainer.setTerminateThreshold(0.01)
    trainer.run()
    val evalNetwork = new PipelineNetwork()
    evalNetwork.add(adaptationLayer)
    val adversarialImage = CudaExecutionContext.gpuContexts.map((cuda:CudaExecutionContext)=>evalNetwork.eval(cuda, adversarialData.head.head).getData.get(0))
    adversarialImage
  }

  private def ganD(model: NNLayer, adversarialData: Array[Array[Tensor]]): Tensor = {
    val adaptationLayer = new ConvolutionLayer(3, 3, 3, 3)
    for (i <- 0 until 3) adaptationLayer.filter.set(Array(1, 1, 4 * i), 1.0)
    val trainingNetwork = new PipelineNetwork(2)
    trainingNetwork.add(adaptationLayer)
    val pipelineNetwork = KryoUtil.kryo().copy(model).freeze().asInstanceOf[PipelineNetwork]
    pipelineNetwork.setHead(pipelineNetwork.getByLabel("classify")).removeLastInput()
    trainingNetwork.add(pipelineNetwork)
    trainingNetwork.add(new SumInputsLayer(),
      trainingNetwork.add(new EntropyLossLayer(), trainingNetwork.getHead, trainingNetwork.getInput(1)),
      trainingNetwork.add(new SumReducerLayer(),
        trainingNetwork.add(new LinearActivationLayer().setScale(1e-1).freeze(),
          trainingNetwork.add(new NthPowerActivationLayer().setPower(2),
            trainingNetwork.add(new LinearActivationLayer().setBias(-1).freeze(),
              trainingNetwork.add(new SumReducerLayer(),
                trainingNetwork.add(new WeightExtractor(0,adaptationLayer))
              )
            )
          )
        ),
        trainingNetwork.add(new LinearActivationLayer().setScale(1e-3).freeze(),
          trainingNetwork.add(new SumReducerLayer(),
            trainingNetwork.add(new NthPowerActivationLayer().setPower(2),
              trainingNetwork.add(new WeightExtractor(0,adaptationLayer))
            )
          )
        )
      )
    )
    this.model = trainingNetwork
    var inner: Trainable = new ArrayTrainable(adversarialData, trainingNetwork)
    val trainer = new IterativeTrainer(inner)
    trainer.setMonitor(monitor)
    trainer.setTimeout(1, TimeUnit.MINUTES)
    trainer.setOrientation(new GradientDescent)
    trainer.setLineSearchFactory(Java8Util.cvt((s: String) ⇒ new ArmijoWolfeSearch().setMaxAlpha(1e8): LineSearchStrategy))
    trainer.setTerminateThreshold(0.01)
    trainer.run()
    val evalNetwork = new PipelineNetwork()
    evalNetwork.add(adaptationLayer)
    val adversarialImage = CudaExecutionContext.gpuContexts.map((cuda:CudaExecutionContext)=>evalNetwork.eval(cuda, adversarialData.head.head).getData.get(0))
    adversarialImage
  }


  def declareTestHandler() = {
    out.p("<a href='testCat.html'>Test Categorization</a><br/>")
    server.addSyncHandler("testCat.html", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(out ⇒ {
        testCategorization(out, getModelCheckpoint)
      })
    }), false)
    out.p("<a href='gan.html'>Generate Adversarial Images</a><br/>")
    server.addSyncHandler("gan.html", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(out ⇒ {
        gan(out, getModelCheckpoint)()
      })
    }), false)
  }

  def testCategorization(out: HtmlNotebookOutput with ScalaNotebookOutput, model: NNLayer) = {
    try {
      out.eval {
        TableOutput.create(data.selectCategories(2).values.reduce(_.union(_)).collect().map(testObj ⇒ Map[String, AnyRef](
          "Image" → out.image(testObj(0).toRgbImage(), ""),
          "Categorization" → categories.toList.sortBy(_._2).map(_._1)
            .zip(model.eval(new NNLayer.NNExecutionContext() {}, testObj(0)).getData.get(0).getData.map(_ * 100.0))
        ).asJava): _*)
      }
    } catch {
      case e: Throwable ⇒ e.printStackTrace()
    }
  }

  lazy val categories: Map[String, Int] = data.categoryList.zipWithIndex.toMap

}

