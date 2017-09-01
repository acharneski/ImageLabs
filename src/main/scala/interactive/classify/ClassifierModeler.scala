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

import java.awt.image.BufferedImage
import java.awt.{Graphics2D, RenderingHints}
import java.io._
import java.lang
import java.util.concurrent.TimeUnit
import java.util.function.{DoubleSupplier, IntToDoubleFunction}
import javax.imageio.ImageIO

import _root_.util.Java8Util.cvt
import _root_.util._
import com.simiacryptus.mindseye.data.Tensor
import com.simiacryptus.mindseye.layers.NNLayer
import com.simiacryptus.mindseye.layers.activation.{AbsActivationLayer, SoftmaxActivationLayer}
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer
import com.simiacryptus.mindseye.layers.meta.StdDevMetaLayer
import com.simiacryptus.mindseye.layers.reducers.{AvgReducerLayer, ProductInputsLayer, SumInputsLayer}
import com.simiacryptus.mindseye.layers.util.ConstNNLayer
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.network.graph.DAGNode
import com.simiacryptus.mindseye.layers.cudnn.f32._
import com.simiacryptus.mindseye.layers.media.MaxImageBandLayer
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line._
import com.simiacryptus.mindseye.opt.orient._
import com.simiacryptus.mindseye.opt.trainable._
import com.simiacryptus.util.io.HtmlNotebookOutput
import com.simiacryptus.util.text.TableOutput
import com.simiacryptus.util.{MonitoredObject, StreamNanoHTTPD, Util}
import interactive.superres.SimplexOptimizer
import util.NNLayerUtil._

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object ClassifierModeler extends Report {
  val modelName = System.getProperty("modelName","image_classifier_13")
  val tileSize = 64
  val scaleFactor: Double = (64 * 64.0) / (tileSize * tileSize)
  val categoryWhitelist = Set[String]("greyhound", "soccer-ball", "telephone-box", "windmill")
  val numberOfCategories = categoryWhitelist.size

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒ new ClassifierModeler(source, server, out).run()
      case _ ⇒ new ClassifierModeler("D:\\testImages\\256_ObjectCategories", server, out).run()
    })

  }

}
import ClassifierModeler._

object TestClassifier {

  val validSizes: Stream[Int] = Stream.from(1).map(i⇒List[Int⇒Int](
    _+2, _*2, _+3, _*2, _+4, _*2, _+4
  ).foldLeft(i)((x,fn)⇒fn(x)))

}
case class TestClassifier(
                           conv1 : Double = -3.1962815165239653,
                           conv2 : Double = 0.5,
                           conv3 : Double = -2.5,
                           conv4 : Double = -7.5,
                           auxLearn1: Double = -1,
                           auxLearn2: Double = -1,
                           auxLearn3: Double = -1,
                           reduction1: Double = -1,
                           reduction2: Double = -1
                                   ) {

  def getNetwork(monitor: TrainingMonitor,
                 monitoringRoot : MonitoredObject,
                 fitness : Boolean = false) : NNLayer = {
    var network: PipelineNetwork = new PipelineNetwork(2)
    val zeroSeed : IntToDoubleFunction = Java8Util.cvt(_ ⇒ 0.0)
    val normalizedPoints = new ArrayBuffer[DAGNode]()
    val subPrediction = new mutable.HashMap[DAGNode,(Integer,Double,String)]()
    def rand = Util.R.get.nextDouble() * 2 - 1
    def buildLayer(from: Int,
                   to: Int,
                   layerNumber: String,
                   weights: Double,
                   layerRadius: Int = 5,
                   simpleBorder: Boolean = false,
                   activationLayer: NNLayer = new ActivationLayer(ActivationLayer.Mode.RELU),
                   auxWeight : Double = Double.NaN): DAGNode = {
      def weightSeed : DoubleSupplier = Java8Util.cvt(() ⇒ rand * weights)
      network.add(new ImgBandBiasLayer(from).setWeights(zeroSeed).setName("bias_" + layerNumber).addTo(monitoringRoot))
      if (null != activationLayer) {
        network.add(activationLayer.setName("activation_" + layerNumber).freeze.addTo(monitoringRoot))
      }
      val node = network.add(new ConvolutionLayer(layerRadius, layerRadius, from * to, simpleBorder)
        .setWeights(weightSeed).setName("conv_" + layerNumber).addTo(monitoringRoot)
      )
      //network.add(new MonitoringSynapse().addTo(monitoringRoot).setName("output_" + layerNumber))
      normalizedPoints += node
      if(!auxWeight.isNaN) subPrediction(node) = (to, auxWeight, layerNumber)
      node
    }

    buildLayer(3, 8, "0", layerRadius = 5, weights = Math.pow(10, this.conv1), activationLayer = null, auxWeight = auxLearn1)
    network.add(new PoolingLayer().setName("avg_0").addTo(monitoringRoot))
    normalizedPoints += network.add(new ConvolutionLayer(1, 1, 8 * 4, false).setWeights(()=>Math.pow(10, this.reduction1)*rand).setName("reduce_1").addTo(monitoringRoot));
    buildLayer(4, 30, "1", layerRadius = 5, weights = Math.pow(10, this.conv2), auxWeight = auxLearn2)
    network.add(new PoolingLayer().setName("avg_1").addTo(monitoringRoot))
    normalizedPoints += network.add(new ConvolutionLayer(1, 1, 30 * 6, false).setWeights(()=>Math.pow(10, this.reduction2)*rand).setName("reduce_2").addTo(monitoringRoot));
    buildLayer(6, 24, "2", layerRadius = 4, weights = Math.pow(10, this.conv3), auxWeight = auxLearn3)
    network.add(new PoolingLayer().setName("avg_3").addTo(monitoringRoot))
    buildLayer(24, numberOfCategories, "3", layerRadius = 1, weights = Math.pow(10, this.conv4))
    network.add(new MaxImageBandLayer().setName("avgFinal").addTo(monitoringRoot))
    val prediction = network.add(new SoftmaxActivationLayer)

    def auxEntropyLayer(source: DAGNode, bands: Int, weight: Double, layerNumber: String) = {
      val convolution = network.add(new ConvolutionLayer(1, 1, bands * numberOfCategories, false).setWeights(() ⇒ Random.nextDouble() * weight).setName("learningStrut_" + layerNumber), source)
      normalizedPoints += convolution
      network.add(new EntropyLossLayer(), network.add(new SoftmaxActivationLayer, network.add(new MaxImageBandLayer(), convolution)), network.getInput(1))
    }

    val learningStruts = subPrediction.map(e ⇒ auxEntropyLayer(e._1, e._2._1, e._2._2, e._2._3))
    network.add(new SumInputsLayer(), (List(network.add(new EntropyLossLayer(), prediction, network.getInput(1))) ++
          learningStruts):_*)

    if(fitness) {
      def auxRmsLayer(layer:DAGNode, target:Double) = network.add(new AbsActivationLayer(), network.add(new SumInputsLayer(),
                network.add(new AvgReducerLayer(), network.add(new StdDevMetaLayer(), layer)),
                network.add(new ConstNNLayer(new Tensor(1).set(0,-target)))
              ))
      val prediction = network.getHead
      network.add(new ProductInputsLayer(), prediction, network.add(new SumInputsLayer(),
                (List(network.add(new ConstNNLayer(new Tensor(1).set(0,0.1)))) ++ normalizedPoints.map(auxRmsLayer(_,1))):_*
              ))
    }

    network
  }

  def fitness(monitor: TrainingMonitor, monitoringRoot : MonitoredObject, data: Array[Array[Tensor]], n: Int = 3) : Double = {
    val values = (1 to n).map(i ⇒ {
      val network = getNetwork(monitor, monitoringRoot, fitness = true)
      val measure = new ArrayTrainable(data, network).measure()
      measure.value
    }).toList
    val avg = values.sum / n
    monitor.log(s"Numeric Opt: $this => $avg ($values)")
    avg
  }

}

class ClassifierModeler(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  def run(awaitExit:Boolean=true): Unit = {
    defineHeader()
    declareTestHandler()
    out.out("<hr/>")
    if(findFile(modelName).isEmpty || System.getProperties.containsKey("rebuild")) step_Generate()
    while(true) step_LBFGS((250 * scaleFactor).toInt, 60, 200)
    out.out("<hr/>")
    if(awaitExit) waitForExit()
  }

  def resize(source: BufferedImage, size: Int) = {
    val image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB)
    val graphics = image.getGraphics.asInstanceOf[Graphics2D]
    graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
    graphics.drawImage(source, 0, 0, size, size, null)
    image
  }

  def resize(source: BufferedImage, width: Int, height: Int) = {
    val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
    val graphics = image.getGraphics.asInstanceOf[Graphics2D]
    graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
    graphics.drawImage(source, 0, 0, width, height, null)
    image
  }

  def step_Generate() = {
    phase({
      lazy val optTraining: Array[Array[Tensor]] = Random.shuffle(data.toStream).take((5 * scaleFactor).ceil.toInt).toArray
      SimplexOptimizer[TestClassifier](
        TestClassifier(),
        x ⇒ x.fitness(monitor, monitoringRoot, optTraining, n=3), relativeTolerance=0.01
      ).getNetwork(monitor, monitoringRoot)
//      TestClassifier().getNetwork(monitor, monitoringRoot)
    }, (model: NNLayer) ⇒ {
      out.h1("Model Initialization")
      val trainer = out.eval {
        assert(null != data)
        var inner: Trainable = new StochasticArrayTrainable(data, model, (20 * scaleFactor).toInt)
        val trainer = new IterativeTrainer(inner)
        trainer.setMonitor(monitor)
        trainer.setTimeout(5, TimeUnit.MINUTES)
        trainer.setIterationsPerSample(1)
        trainer.setOrientation(new GradientDescent())
        trainer.setLineSearchFactory(Java8Util.cvt((s) ⇒ new QuadraticSearch))
        trainer.setTerminateThreshold(1.0)
        trainer
      }
      trainer.run()
    }: Unit, modelName)
  }

  def step_diagnostics_layerRates(sampleSize : Int = (100 * scaleFactor).toInt) = phase[Map[NNLayer, LayerRateDiagnosticTrainer.LayerStats]](
    modelName, (model: NNLayer) ⇒ {
    out.h1("Diagnostics - Layer Rates")
    out.eval {
      var inner: Trainable = new StochasticArrayTrainable(data, model, sampleSize)
      val trainer = new LayerRateDiagnosticTrainer(inner).setStrict(true).setMaxIterations(1)
      trainer.setMonitor(monitor)
      trainer.run()
      trainer.getLayerRates().asScala.toMap
    }
  }, modelName)

  def step_SGD(sampleSize: Int, timeoutMin: Int, termValue: Double = 0.0, momentum: Double = 0.2, maxIterations: Int = Integer.MAX_VALUE, reshufflePeriod: Int = 1,rates: Map[String, Double] = Map.empty) = phase(modelName, (model: NNLayer) ⇒ {
    monitor.clear()
    out.h1(s"SGD(sampleSize=$sampleSize,timeoutMin=$timeoutMin)")
    val trainer = out.eval {
      var inner: Trainable = new StochasticArrayTrainable(data, model, sampleSize)
      val trainer = new IterativeTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(timeoutMin, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(reshufflePeriod)
      val momentumStrategy = new MomentumStrategy(new GradientDescent()).setCarryOver(momentum)
      val reweight = new LayerReweightingStrategy(momentumStrategy) {
        override def getRegionPolicy(layer: NNLayer): lang.Double = layer.getName match {
          case key if rates.contains(key) ⇒ rates(key)
          case _ ⇒ 1.0
        }
        override def reset(): Unit = {}
      }
      trainer.setOrientation(reweight)
      trainer.setLineSearchFactory(Java8Util.cvt((s)⇒new ArmijoWolfeSearch().setAlpha(1e-12).setC1(0).setC2(1)))
      trainer.setTerminateThreshold(termValue)
      trainer.setMaxIterations(maxIterations)
      trainer
    }
    trainer.run()
  }, modelName)

  def step_LBFGS(sampleSize: Int, timeoutMin: Int, iterationSize: Int): Unit = phase(modelName, (model: NNLayer) ⇒ {
    monitor.clear()
    out.h1(s"LBFGS(sampleSize=$sampleSize,timeoutMin=$timeoutMin)")
    val trainer = out.eval {
      val inner = new StochasticArrayTrainable(data, model, sampleSize)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(timeoutMin, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(iterationSize)
      val lbfgs = new LBFGS().setMaxHistory(35).setMinHistory(4)
      trainer.setOrientation(lbfgs)
      trainer.setLineSearchFactory(Java8Util.cvt((s:String)⇒(s match {
        case s if s.contains("LBFGS") ⇒ new StaticLearningRate().setRate(1.0)
        case _ ⇒ new ArmijoWolfeSearch().setAlpha(1e-5)
      })))
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    trainer.run()
  }, modelName)

  def declareTestHandler() = {
    out.p("<a href='testCat.html'>Test Categorization</a><br/>")
    server.addSyncHandler("testCat.html", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(out ⇒ {
        testCategorization(out, getModelCheckpoint)
      })
    }), false)
  }

  def testCategorization(out: HtmlNotebookOutput with ScalaNotebookOutput, model : NNLayer) = {
    try {
      out.eval {
        TableOutput.create(Random.shuffle(data.toList).take(100).map(testObj ⇒ Map[String, AnyRef](
          "Image" → out.image(testObj(0).toRgbImage(), ""),
          "Categorization" → categories.toList.sortBy(_._2).map(_._1)
            .zip(model.eval(new NNLayer.NNExecutionContext() {}, testObj(0)).getData.get(0).getData.map(_ * 100.0))
        ).asJava): _*)
      }
    } catch {
      case e: Throwable ⇒ e.printStackTrace()
    }
  }

  lazy val categories: Map[String, Int] = categoryList.zipWithIndex.toMap
  lazy val (categoryList: List[String], data: Array[Array[Tensor]]) = {
    monitor.log("Valid Processing Sizes: " + TestClassifier.validSizes.take(100).toList)
    out.p("Preparing training dataset")
    out.p("Loading data from " + source)
    val (categoryList: Seq[String], data: Array[Array[Tensor]]) = load()
    val categories: Map[String, Int] = categoryList.zipWithIndex.toMap
    out.p("<ol>" + categories.toList.sortBy(_._2).map(x ⇒ "<li>" + x + "</li>").mkString("\n") + "</ol>")
    out.eval {
      TableOutput.create(data.take(100).map((e: Array[Tensor]) ⇒ {
        Map[String, AnyRef](
          "Image" → out.image(e(0).toRgbImage(), e(1).toString),
          "Classification" → e(1)
        ).asJava
      }): _*)
    }
    out.p("Loading data complete")
    (categoryList, data)
  }

  def load(maxDim: Int = 256,
           numberOfCategories: Int = numberOfCategories,
           imagesPerCategory: Int = 100
          ): (Seq[String], Array[Array[Tensor]]) = {

    def toOut(label: String, max: Int): Int = {
      (0 until max).find(label == "[" + _ + "]").get
    }

    def toOutNDArray(out: Int, max: Int): Tensor = {
      val ndArray = new Tensor(max)
      ndArray.set(out, 1)
      ndArray
    }

    val images: Seq[(Tensor, String)] = Random.shuffle(Random.shuffle(new File(source).listFiles().toStream)
      .filter(dir=>categoryWhitelist.find(str=>dir.getName.contains(str)).isDefined)
      .take(numberOfCategories)
      .flatMap((categoryDirectory: File) ⇒ {
      val categoryName = categoryDirectory.getName.split('.').last
      Random.shuffle(categoryDirectory.listFiles().toStream).take(imagesPerCategory)
        .filterNot(_==null).map(ImageIO.read).filterNot(_==null)
        .map(original ⇒ {

        //def fit(x:Int) = TestClassifier.validSizes.takeWhile(_<x).last
        def fit(x: Int) = x

        val fromWidth = original.getWidth()
        val fromHeight = original.getHeight()
        val scale = maxDim.toDouble / Math.min(fromWidth, fromHeight)
        val toWidth = fit((fromWidth * scale).toInt)
        val toHeight = fit((fromHeight * scale).toInt)
        val resized = new BufferedImage(maxDim, maxDim, BufferedImage.TYPE_INT_ARGB)
        val graphics = resized.getGraphics.asInstanceOf[Graphics2D]
        graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
        if (toWidth < toHeight) {
          graphics.drawImage(original, 0, (toWidth - toHeight) / 2, toWidth, toHeight, null)
        } else {
          graphics.drawImage(original, (toHeight - toWidth) / 2, 0, toWidth, toHeight, null)
        }
        Tensor.fromRGB(resized) → categoryName
      })
    }).toList)
    val categoryList = images.map(_._2).distinct.sorted
    val categories: Map[String, Int] = categoryList.zipWithIndex.toMap
    val data: Array[Array[Tensor]] = images.map(e ⇒ {
      Array(e._1, toOutNDArray(categories(e._2), categories.size))
    }).toArray
    (categoryList, data)
  }
}

