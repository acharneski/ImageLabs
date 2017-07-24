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
import com.google.gson.{GsonBuilder, JsonObject}
import com.simiacryptus.mindseye.layers.activation._
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer
import com.simiacryptus.mindseye.layers.media._
import com.simiacryptus.mindseye.layers.meta.StdDevMetaLayer
import com.simiacryptus.mindseye.layers.reducers.{AvgReducerLayer, ProductInputsLayer, SumInputsLayer}
import com.simiacryptus.mindseye.layers.util.ConstNNLayer
import com.simiacryptus.mindseye.layers.{NNLayer, NNResult}
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.network.graph.DAGNode
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line._
import com.simiacryptus.mindseye.opt.orient._
import com.simiacryptus.mindseye.opt.trainable._
import com.simiacryptus.util.io.HtmlNotebookOutput
import com.simiacryptus.util.ml.Tensor
import com.simiacryptus.util.test.ImageTiles.ImageTensorLoader
import com.simiacryptus.util.test.LabeledObject
import com.simiacryptus.util.text.TableOutput
import com.simiacryptus.util.{MonitoredObject, StreamNanoHTTPD, Util}
import interactive.superres.SimplexOptimizer
import org.apache.commons.io.IOUtils
import util.NNLayerUtil._

import scala.collection.JavaConverters._
import scala.util.Random

object TestClassifier {

  val validSizes: Stream[Int] = Stream.from(1).map(i⇒List[Int⇒Int](
    _+2, _*2, _+3, _*2, _+4, _*2, _+4
  ).foldLeft(i)((x,fn)⇒fn(x)))

}
case class TestClassifier(
                                     weight1 : Double,
                                     weight2 : Double,
                                     weight3 : Double,
                                     weight4 : Double
                                   ) {

  def getNetwork(monitor: TrainingMonitor,
                 monitoringRoot : MonitoredObject,
                 fitness : Boolean = false) : NNLayer = {
    val parameters = this
    var network: PipelineNetwork = new PipelineNetwork(2)
    val zeroSeed : IntToDoubleFunction = Java8Util.cvt(_ ⇒ 0.0)
    def buildLayer(from: Int,
                   to: Int,
                   layerNumber: String,
                   weights: Double,
                   layerRadius: Int = 5,
                   simpleBorder: Boolean = false,
                   activationLayer: NNLayer = new ReLuActivationLayer()) = {
      def weightSeed : DoubleSupplier = Java8Util.cvt(() ⇒ {
        val r = Util.R.get.nextDouble() * 2 - 1
        r * weights
      })
      network.add(new ImgBandBiasLayer(from).setWeights(zeroSeed).setName("bias_" + layerNumber).addTo(monitoringRoot))
      if (null != activationLayer) {
        network.add(activationLayer.setName("activation_" + layerNumber).freeze.addTo(monitoringRoot))
      }
      network.add(new ImgConvolutionSynapseLayer(layerRadius, layerRadius, from * to, simpleBorder)
        .setWeights(weightSeed).setName("conv_" + layerNumber).addTo(monitoringRoot));
      //network.add(new MonitoringSynapse().addTo(monitoringRoot).setName("output_" + layerNumber))
    }

    // 64 x 64 x 3
    val l1 = buildLayer(3, 5, "0", layerRadius = 5, weights = Math.pow(10, parameters.weight1), activationLayer = null)
    network.add(new MaxSubsampleLayer(2, 2, 1).setName("avg0").addTo(monitoringRoot))
    // 30
    val l2 = buildLayer(5, 10, "1", layerRadius = 5, weights = Math.pow(10, parameters.weight2), activationLayer = new HyperbolicActivationLayer().setScale(5).freeze())
    network.add(new MaxSubsampleLayer(2, 2, 1).setName("avg1").addTo(monitoringRoot))
    // 14
    val l3 = buildLayer(10, 20, "2", layerRadius = 4, weights = Math.pow(10, parameters.weight3), activationLayer = new HyperbolicActivationLayer().setScale(5).freeze())
    network.add(new MaxSubsampleLayer(2, 2, 1).setName("avg3").addTo(monitoringRoot))
    // 5
    val l4 = buildLayer(20, 3, "3", layerRadius = 3, weights = Math.pow(10, parameters.weight4), activationLayer = new HyperbolicActivationLayer().setScale(5).freeze())

    network.add(new MaxImageBandLayer().setName("avgFinal").addTo(monitoringRoot))
    network.add(new SoftmaxActivationLayer)

    //network.add(new ImgReshapeLayer(2,2,true))
    //network.add(new HyperbolicActivationLayer().setScale(5).freeze().setName("activation4").addTo(monitoringRoot))

    val output = network.getHead
    if(fitness) {
      def normalizeStdDev(layer:DAGNode, target:Double) = network.add(new AbsActivationLayer(),
        network.add(new SumInputsLayer(),
          network.add(new AvgReducerLayer(), network.add(new StdDevMetaLayer(), layer)),
          network.add(new ConstNNLayer(new Tensor(1).set(0,-target)))
        )
      )
      network.add(new ProductInputsLayer(),
        network.add(new EntropyLossLayer(), output, network.getInput(1)),
        network.add(new SumInputsLayer(),
          network.add(new ConstNNLayer(new Tensor(1).set(0,0.1))),
          normalizeStdDev(l1,1),
          normalizeStdDev(l2,1),
          normalizeStdDev(l3,1),
          normalizeStdDev(l4,1)
        )
      )
    } else {
      network.add(new EntropyLossLayer(), output, network.getInput(1))
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

  val modelName = System.getProperty("modelName","image_classifier_1")
  val tileSize = 64
  val scaleFactor: Double = (64 * 64.0) / (tileSize * tileSize)

  def run(awaitExit:Boolean=true): Unit = {
    defineHeader()
    declareTestHandler()
    out.out("<hr/>")
    if(findFile(modelName).isEmpty || System.getProperties.containsKey("rebuild")) step_Generate()
    step_LBFGS((50 * scaleFactor).toInt, 30, 50)
    step_SGD((100 * scaleFactor).toInt, 30, reshufflePeriod = 5)
    step_LBFGS((500 * scaleFactor).toInt, 60, 50)
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
      val optTraining: Array[Array[Tensor]] = Random.shuffle(data.toStream).take((10 * scaleFactor).ceil.toInt).toArray
      SimplexOptimizer[TestClassifier](
        TestClassifier(-3.1962815165239653,0.5,-2.5,-7.5),//(-2.1962815165239653,1.0,-2.0,-6.0),
        x ⇒ x.fitness(monitor, monitoringRoot, optTraining, n=3), relativeTolerance=0.01
      ).getNetwork(monitor, monitoringRoot)
    }, (model: NNLayer) ⇒ {
      out.h1("Model Initialization")
      val trainer = out.eval {
        var inner: Trainable = new StochasticArrayTrainable(data, model, (50 * scaleFactor).toInt)
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

  lazy val forwardNetwork = loadModel("downsample_1")

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
            .zip(model.eval(testObj(0)).data.head.getData.map(_ * 100.0))
        ).asJava): _*)
      }
    } catch {
      case e: Throwable ⇒ e.printStackTrace()
    }
  }

  lazy val categories: Map[String, Int] = categoryList.zipWithIndex.toMap
  lazy val (categoryList: List[String], data: Array[Array[Tensor]]) = {
    def toOut(label: String, max: Int): Int = {
      (0 until max).find(label == "[" + _ + "]").get
    }
    def toOutNDArray(out: Int, max: Int): Tensor = {
      val ndArray = new Tensor(max)
      ndArray.set(out, 1)
      ndArray
    }
    out.p("Preparing training dataset")


    out.p("Loading data from " + source)

    monitor.log("Valid Processing Sizes: " + TestClassifier.validSizes.take(100).toList)
    val maxDim = 256
    val maxCategories = 3
    val imageCount = 100
    val images: Seq[(Tensor,String)] = Random.shuffle(new File(source).listFiles().toStream).take(maxCategories).flatMap((categoryDirectory: File) ⇒{
      val categoryName = categoryDirectory.getName.split('.')(1)
      Random.shuffle(categoryDirectory.listFiles().toStream).map(imageFile⇒{
        val original = ImageIO.read(imageFile)
        //def fit(x:Int) = TestClassifier.validSizes.takeWhile(_<x).last
        def fit(x:Int) = x

        val fromWidth = original.getWidth()
        val fromHeight = original.getHeight()
        val scale = maxDim.toDouble / Math.min(fromWidth, fromHeight)
        val toWidth = fit((fromWidth * scale).toInt)
        val toHeight = fit((fromHeight * scale).toInt)
        val resized = new BufferedImage(maxDim, maxDim, BufferedImage.TYPE_INT_ARGB)
        val graphics = resized.getGraphics.asInstanceOf[Graphics2D]
        graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
        if(toWidth < toHeight) {
          graphics.drawImage(original, 0, (toWidth-toHeight)/2, maxDim, maxDim, null)
        } else {
          graphics.drawImage(original, (toHeight-toWidth)/2, 0, maxDim, maxDim, null)
        }
        Tensor.fromRGB(resized) → categoryName
      })
    }).take(imageCount).toList
    val categoryList = images.map(_._2).distinct.sorted
    val categories: Map[String, Int] = categoryList.zipWithIndex.toMap




    out.p("<ol>" + categories.toList.sortBy(_._2).map(x ⇒ "<li>" + x + "</li>").mkString("\n") + "</ol>")
    val data: Array[Array[Tensor]] = images.map(e⇒{
      Array(e._1, toOutNDArray(categories(e._2), categories.size))
    }).toArray



    out.eval {
      TableOutput.create(images.take(100).map((e: (Tensor, String)) ⇒ {
        Map[String, AnyRef](
          "Image" → out.image(e._1.toRgbImage(), e._2),
          "Label" → e._2
        ).asJava
      }): _*)
    }
    out.p("Loading data complete")
    (categoryList, data)
  }

}


object ClassifierModeler extends Report {

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒ new ClassifierModeler(source, server, out).run()
      case _ ⇒ new ClassifierModeler("E:\\testImages\\256_ObjectCategories", server, out).run()
    })

  }

}
