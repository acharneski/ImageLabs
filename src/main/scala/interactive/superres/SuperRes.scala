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

package interactive.superres

import java.awt.image.BufferedImage
import java.awt.{Graphics2D, RenderingHints}
import java.io._
import java.lang
import java.util.concurrent.TimeUnit
import java.util.function.{DoubleSupplier, IntToDoubleFunction, Supplier}

import _root_.util._
import com.simiacryptus.mindseye.layers.activation.{HyperbolicActivationLayer, ReLuActivationLayer}
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer
import com.simiacryptus.mindseye.layers.media.{ImgBandBiasLayer, ImgConvolutionSynapseLayer, ImgReshapeLayer}
import com.simiacryptus.mindseye.layers.reducers.{ProductInputsLayer, SumInputsLayer}
import com.simiacryptus.mindseye.layers.util.{ConstNNLayer, MonitoringWrapper}
import com.simiacryptus.mindseye.layers.{DeltaBuffer, NNLayer}
import com.simiacryptus.mindseye.network.graph.DAGNode
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line._
import com.simiacryptus.mindseye.opt.orient.{GradientDescent, LBFGS, MomentumStrategy, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region._
import com.simiacryptus.mindseye.opt.trainable._
import com.simiacryptus.util.{MonitoredObject, StreamNanoHTTPD, Util}
import com.simiacryptus.util.data.DoubleStatistics
import com.simiacryptus.util.io.HtmlNotebookOutput
import com.simiacryptus.util.ml.{Coordinate, Tensor}
import com.simiacryptus.util.test.ImageTiles.ImageTensorLoader
import com.simiacryptus.util.text.TableOutput
import util.Java8Util.cvt

import scala.collection.JavaConverters._
import scala.util.Random


object SuperRes extends ServiceNotebook {

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒ new SuperRes(source, server, out).run()
      case _ ⇒ new SuperRes("E:\\testImages\\256_ObjectCategories", server, out).run()
    })

  }

}

case class QuadraticNetworkSuperRes(
                             weight1 : Double,
                             weight2 : Double,
                             weight3 : Double,
                             weight4 : Double
                           ) {
  def getNetwork(monitor: TrainingMonitor, monitoringRoot : MonitoredObject) : PipelineNetwork = {
    val parameters = this
    monitor.log(s"Building network with parameters $parameters")
    var network: PipelineNetwork = new PipelineNetwork
    val zeroSeed : IntToDoubleFunction = Java8Util.cvt(_ ⇒ 0.0)
    val layerRadius = 3
    def buildLayer(from: Int, to: Int, layerNumber: String, root: DAGNode = network.getHead, activation: ⇒ NNLayer = new ReLuActivationLayer(), weights: Double = 0.1): DAGNode = {
      def weightSeed : DoubleSupplier = Java8Util.cvt(() ⇒ {
        val r = Util.R.get.nextDouble() * 2 - 1
        r * weights
      })
      network.add(new MonitoringWrapper(new ImgBandBiasLayer(from).setWeights(zeroSeed).setName("bias_" + layerNumber)).addTo(monitoringRoot), root);
      if (!layerNumber.startsWith("0") && activation != null) {
        network.add(new MonitoringWrapper(activation.setName("activation_" + layerNumber)).addTo(monitoringRoot));
      }
      network.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(layerRadius, layerRadius, from * to).setWeights(weightSeed).setName("conv_" + layerNumber)).addTo(monitoringRoot));
      //network.add(new MonitoringSynapse().addTo(monitoringRoot).setName("output_" + layerNumber))
    }

    val input = network.getHead
    val layer2 = network.add(new SumInputsLayer(),
      buildLayer(3, 48, "0a", input, weights = parameters.weight1),
      network.add(new ProductInputsLayer(),
        buildLayer(3, 48, "0c", input, weights = parameters.weight2),
        buildLayer(3, 48, "0b", input, weights = parameters.weight3)))
    buildLayer(3, 48, "1", input, weights = parameters.weight4)
    network.add(new ImgReshapeLayer(4,4,true))

    network
  }

}


class SuperRes(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  val modelName = System.getProperty("modelName","oracle2")

  val fitnessBorderPadding = 8

  def run(): Unit = {
    defineHeader()
    defineTestHandler()
    out.out("<hr/>")
    if(findFile(modelName).isEmpty || System.getProperties.containsKey("rebuild")) step_Generate()
    //step_diagnostic()
    step_SGD(50, 2*60, reshufflePeriod = 10)
    step_SGD(100, 2*60, reshufflePeriod = 10)
    step_SGD(200, 2*60, reshufflePeriod = 10)
    step_SGD(500, 2*60, reshufflePeriod = 5)
    step_SGD(1000, 4*60, reshufflePeriod = 5)
    step_LBFGS(1000, 4 * 60)
    summarizeHistory()
    out.out("<hr/>")
    waitForExit()
  }

  def resize(source: BufferedImage, size: Int) = {
    val image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB)
    val graphics = image.getGraphics.asInstanceOf[Graphics2D]
    graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
    graphics.drawImage(source, 0, 0, size, size, null)
    image
  }

  lazy val data : List[Array[Tensor]] = {
    out.p("Loading data from " + source)
    val rawList: List[Tensor] = rawData
    System.gc()
    val data: List[Array[Tensor]] = rawList.map(tile ⇒ Array(Tensor.fromRGB(resize(tile.toRgbImage, 16)), tile))
    out.eval {
      TableOutput.create(Random.shuffle(data).take(100).map(testObj ⇒ Map[String, AnyRef](
        "Source" → out.image(testObj(1).toRgbImage(), ""),
        "Resized" → out.image(testObj(0).toRgbImage(), "")
      ).asJava): _*)
    }
    out.p("Loading data complete")
    data
  }

  private def rawData() = {
    val loader = new ImageTensorLoader(new File(source), 64, 64, 64, 64, 10, 10)
    val rawList = loader.stream().iterator().asScala.take(10000).toList
    loader.stop()
    rawList
  }


  def evalNetwork_GradientDistribution(network : PipelineNetwork) : Double = {
    val N = 2
    (0 until N).map(i⇒{
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(network, lossNetwork)
      var inner: Trainable = new StochasticArrayTrainable(data.toArray, trainingNetwork, 100)
      inner = new ConstL12Normalizer(inner).setFactor_L1(0.001)
      val measure = inner.measure()
      val zeroTol = 1e-15
      val vecMap = measure.delta.map.asScala.map((x: (NNLayer, DeltaBuffer)) ⇒(x._1, (measure.weights.map.get(x._1).sumSq(), x._2.sumSq()))).toMap
      val average = new DoubleStatistics().accept(vecMap.values.filterNot(x⇒Math.abs(x._1)<zeroTol || Math.abs(x._2)<zeroTol).map(x⇒{
        val (wx,dx) = x
        Math.abs(Math.log(wx / dx))
      }).toArray).getStandardDeviation
      monitor.log(s"Network entropy network $average: $vecMap")
      average
    }).sum/N
  }

  def evalNetwork_Value(network : PipelineNetwork) : Double = {
    val N = 2
    (0 until N).map(i⇒{
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(network, lossNetwork)
      var inner: Trainable = new StochasticArrayTrainable(data.toArray, trainingNetwork, 50)
      inner = new ConstL12Normalizer(inner).setFactor_L1(0.001)
      val measure = inner.measure()
      measure.value
      monitor.log(s"Network result: ${measure.value}")
      measure.value
    }).sum/N
  }

  def step_Generate() = {
    phase({
        QuadraticNetworkSuperRes(5.28671875,4.95546875,4.89296875,0.01796875372529031)
//      SimplexOptimizer.apply[QuadraticNetworkSuperRes](
//        x ⇒ evalNetwork_Value(x.getNetwork(monitor, monitoringRoot))
//      )
        .getNetwork(monitor, monitoringRoot)
    }, (model: NNLayer) ⇒ {
      out.h1("Step 1")
      val trainer = out.eval {
        val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, lossNetwork)
        val dataArray = data.toArray
        var inner: Trainable = new StochasticArrayTrainable(dataArray, trainingNetwork, 50)
        inner = new ConstL12Normalizer(inner).setFactor_L1(0.001)
        val trainer = new IterativeTrainer(inner)
        trainer.setMonitor(monitor)
        trainer.setTimeout(45, TimeUnit.MINUTES)
        trainer.setIterationsPerSample(1)
        val momentum = new MomentumStrategy(new GradientDescent()).setCarryOver(0.2)
        trainer.setOrientation(momentum)
        trainer.setLineSearchFactory(() ⇒ new ArmijoWolfeSearch)
        trainer.setTerminateThreshold(5000.0)
        trainer
      }
      trainer.run()
    }: Unit, modelName)
  }

  def step_diagnostic(sampleSize : Int = 1000) = phase(modelName, (model: NNLayer) ⇒ {
    out.h1("Diagnostics - Evaluation Stability")
    val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, lossNetwork)
    val dataArray = data.toArray
    val n = 3
    val factory: Supplier[Trainable] = Java8Util.cvt(() ⇒ new StochasticArrayTrainable(dataArray, trainingNetwork, sampleSize / n))
    var uncertiantyEstimator = new UncertiantyEstimateTrainable(n, factory, monitor)
    val uncertiantyProbe = out.eval {
      val trainer = new IterativeTrainer(uncertiantyEstimator).setMaxIterations(1)
      trainer.setMonitor(monitor)
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    uncertiantyProbe.run()
    out.eval {
      uncertiantyEstimator.getUncertianty()
    }
    out.h1("Diagnostics - Layer Rates")
    val layerRateProbe = out.eval {
      var inner: Trainable = new StochasticArrayTrainable(dataArray, trainingNetwork, sampleSize)
      val trainer = new LayerRateDiagnosticTrainer(inner).setStrict(true).setMaxIterations(1)
      trainer.setMonitor(monitor)
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    layerRateProbe.run()
    out.eval {
      layerRateProbe.getLayerRates()
    }
  }: Unit, modelName)

  def step_SGD(sampleSize: Int, timeoutMin: Int, termValue: Double = 0.0, momentum: Double = 0.2, maxIterations: Int = Integer.MAX_VALUE, reshufflePeriod: Int = 1) = phase(modelName, (model: NNLayer) ⇒ {
    out.h1(s"SGD(sampleSize=$sampleSize,timeoutMin=$timeoutMin)")
    val trainer = out.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, lossNetwork)
      val dataArray = data.toArray
      var inner: Trainable = new StochasticArrayTrainable(dataArray, trainingNetwork, sampleSize)
      //inner = new ConstL12Normalizer(inner).setFactor_L1(0.001)
      val trainer = new IterativeTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(timeoutMin, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(reshufflePeriod)
      val momentumStrategy = new MomentumStrategy(new GradientDescent()).setCarryOver(momentum)
      trainer.setOrientation(momentumStrategy)
      trainer.setLineSearchFactory(()⇒new ArmijoWolfeSearch().setAlpha(1e-12))
      trainer.setTerminateThreshold(termValue)
      trainer.setMaxIterations(maxIterations)
      trainer
    }
    trainer.run()
  }: Unit, modelName)

  def step_LBFGS(sampleSize: Int, timeoutMin: Int): Unit = phase(modelName, (model: NNLayer) ⇒ {
    out.h1(s"LBFGS(sampleSize=$sampleSize,timeoutMin=$timeoutMin)")
    val trainer = out.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, lossNetwork)
      var allowReset = true
      val factory: Supplier[Trainable] = Java8Util.cvt(() ⇒ new StochasticArrayTrainable(data.toArray, trainingNetwork, sampleSize) {
        override def resetSampling(): Boolean = if(allowReset) super.resetSampling() else false
      })
      var inner = new UncertiantyEstimateTrainable(5, factory, monitor)
      allowReset = false
      val trainer = new com.simiacryptus.mindseye.opt.RoundRobinTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(timeoutMin, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(1)
      val lbfgs = new LBFGS().setMaxHistory(25).setMinHistory(3)
      trainer.setOrientations(new TrustRegionStrategy(lbfgs) {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _: HyperbolicActivationLayer ⇒ new StaticConstraint
          case _: ReLuActivationLayer ⇒ new StaticConstraint
          case _: ImgBandBiasLayer ⇒ new LinearSumConstraint
          case _ ⇒ null
        }
      }, new TrustRegionStrategy(lbfgs) {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _: HyperbolicActivationLayer ⇒ new StaticConstraint
          case _: ReLuActivationLayer ⇒ new StaticConstraint
          case _: ImgBandBiasLayer ⇒ new StaticConstraint
          case _ ⇒ null
        }
      })
      trainer.setLineSearchFactory(Java8Util.cvt((s:String)⇒(s match {
        case s if s.contains("LBFGS") ⇒ new StaticLearningRate().setRate(1)
        case _ ⇒ new BisectionSearch().setCurrentRate(1e-5)
      })))
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    trainer.run()
  }: Unit, modelName)

  def lossNetwork = {
    val mask: Tensor = new Tensor(64, 64, 3).map(Java8Util.cvt((v: lang.Double, c: Coordinate) ⇒ {
      if (c.coords(0) < fitnessBorderPadding || c.coords(0) >= (64 - fitnessBorderPadding)) {
        0.0
      } else if (c.coords(1) < fitnessBorderPadding || c.coords(1) >= (64 - fitnessBorderPadding)) {
        0.0
      } else {
        1.0
      }
    }))
    val lossNetwork = new PipelineNetwork(2)
    val maskNode = lossNetwork.add(new ConstNNLayer(mask).freeze())
    lossNetwork.add(new MeanSqLossLayer(),
      lossNetwork.add(new ProductInputsLayer(), lossNetwork.getInput(0), maskNode),
      lossNetwork.add(new ProductInputsLayer(), lossNetwork.getInput(1), maskNode)
    )
    lossNetwork
  }

  def defineTestHandler() = {
    out.p("<a href='test.html'>Test Reconstruction</a>")
    server.addSyncHandler("test.html", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(out ⇒ {
        try {
          out.eval {
            TableOutput.create(Random.shuffle(data).take(100).map(testObj ⇒ Map[String, AnyRef](
              "Source Truth" → out.image(testObj(1).toRgbImage(), ""),
              "Corrupted" → out.image(testObj(0).toRgbImage(), ""),
              "Reconstruction" → out.image(getModelCheckpoint.eval(testObj(0)).data.head.toRgbImage(), "")
            ).asJava): _*)
          }
        } catch {
          case e: Throwable ⇒ e.printStackTrace()
        }
      })
    }), false)
  }

}