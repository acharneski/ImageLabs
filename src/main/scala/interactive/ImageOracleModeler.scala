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

package interactive

import java.awt.image.BufferedImage
import java.awt.{Graphics2D, RenderingHints}
import java.io._
import java.lang
import java.util.concurrent.TimeUnit
import java.util.function.{DoubleSupplier, IntToDoubleFunction, Supplier}

import _root_.util._
import com.simiacryptus.mindseye.layers.NNLayer
import com.simiacryptus.mindseye.layers.activation.{HyperbolicActivationLayer, LinearActivationLayer, ReLuActivationLayer}
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer
import com.simiacryptus.mindseye.layers.media.{ImgBandBiasLayer, ImgConvolutionSynapseLayer}
import com.simiacryptus.mindseye.layers.reducers.{ProductInputsLayer, SumInputsLayer}
import com.simiacryptus.mindseye.layers.util.{ConstNNLayer, MonitoringSynapse, MonitoringWrapper}
import com.simiacryptus.mindseye.network.graph.DAGNode
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line.{ArmijoWolfeConditions, LineBracketSearch, LineSearchStrategy, StaticLearningRate}
import com.simiacryptus.mindseye.opt.region.{LinearSumConstraint, StaticConstraint, TrustRegion, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.trainable._
import com.simiacryptus.util.io.HtmlNotebookOutput
import com.simiacryptus.util.ml.{Coordinate, Tensor}
import com.simiacryptus.util.test.ImageTiles.ImageTensorLoader
import com.simiacryptus.util.text.TableOutput
import com.simiacryptus.util.{StreamNanoHTTPD, Util}
import util.Java8Util.cvt

import scala.collection.JavaConverters._
import scala.util.Random

object ImageOracleModeler extends ServiceNotebook {

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒ new ImageOracleModeler(source, server, out).run()
      case _ ⇒ new ImageOracleModeler("E:\\testImages\\256_ObjectCategories", server, out).run()
    })

  }
}

class ImageOracleModeler(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  def run(): Unit = {
    defineHeader()
    defineTestHandler()
    out.out("<hr/>")
    if(findFile("oracle").isEmpty || System.getProperties.contains("rebuild")) step1()
    step2()
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

  val corruptors = Map[String, Tensor ⇒ Tensor](
    "resample4x" → (imgTensor ⇒ {
      Tensor.fromRGB(resize(resize(imgTensor.toRgbImage, 16), 64))
    })
  )

  lazy val data : List[Array[Tensor]] = {
    out.p("Loading data from " + source)
    val rawList: List[Tensor] = rawData
    System.gc()
    val data: List[Array[Tensor]] = rawList.flatMap(tile ⇒ corruptors.map(e ⇒ {
      Array(e._2(tile), tile)
    }))
    out.eval {
      TableOutput.create(Random.shuffle(data).take(100).map(testObj ⇒ Map[String, AnyRef](
        "Source" → out.image(testObj(1).toRgbImage(), ""),
        "Transformed" → out.image(testObj(0).toRgbImage(), "")
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

  val fitnessBorderPadding = 6

  def step1() = phase({
    var network: PipelineNetwork = new PipelineNetwork
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "input1"))
    val zeroSeed : IntToDoubleFunction = Java8Util.cvt(_ ⇒ 0.0)
    val layerRadius = 5
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
      network.add(new MonitoringSynapse().addTo(monitoringRoot).setName("output_" + layerNumber))
    }
//    List(3,12,12,3).sliding(2).map(x⇒x(0)→x(1)).zipWithIndex.foreach(x⇒{
//      val ((from,to), layerNumber) = x
//      buildLayer(from, to, layerNumber.toString)
//    })

    val input = network.getHead
    val layer2 = network.add(new SumInputsLayer(),
      buildLayer(3, 12, "0a", input),
      network.add(new ProductInputsLayer(),
        buildLayer(3, 12, "0c", input, weights = 0.01),
        buildLayer(3, 12, "0b", input, weights = 0.01)))
    buildLayer(12, 3, "1")

//    val input = network.getHead
//    val layer1 = buildLayer(3, 12, "0a", input)
//    val layer2 = network.add(new ProductInputsLayer(),
//      buildLayer(12, 12, "1", layer1, activation = new HyperbolicActivationLayer()),
//      buildLayer(3, 12, "0b", input))
//    buildLayer(12, 3, "2")

//    val layer3 = network.add(new ProductInputsLayer(),
//      buildLayer(12, 3, "2a", layer1),
//      buildLayer(12, 3, "2b", layer1))
    //network.add(new SumInputsLayer(), network.getInput(0), network.getHead)
    network
  }, (model: NNLayer) ⇒ {
    out.h1("Step 1")
    val trainer = out.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, lossNetwork)
      val inner = new StochasticArrayTrainable(data.toArray, trainingNetwork, 1000)
      val trainer = new com.simiacryptus.mindseye.opt.RoundRobinTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(3 * 60, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(1)
      val momentum = new MomentumStrategy(new GradientDescent()).setCarryOver(0.2)
      trainer.setOrientations(new TrustRegionStrategy(momentum) {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _: HyperbolicActivationLayer ⇒ new StaticConstraint
          case _: ReLuActivationLayer ⇒ new StaticConstraint
          case _: ImgBandBiasLayer ⇒ new StaticConstraint
          case _ ⇒ null
        }
      }, new TrustRegionStrategy(momentum) {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _: HyperbolicActivationLayer ⇒ new StaticConstraint
          case _: ReLuActivationLayer ⇒ new StaticConstraint
          case _ ⇒ null
        }
      })
      trainer.setLineSearchFactory(Java8Util.cvt((s:String)⇒{
        //new StaticLearningRate().setRate(1e-5)
        new ArmijoWolfeConditions().setAlpha(1e-8)
      }.asInstanceOf[LineSearchStrategy]))
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    trainer.run()
  }: Unit, "oracle")

  def step2() = phase("oracle", (model: NNLayer) ⇒ {
    out.h1("Step 2")
    val trainer = out.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, lossNetwork)
      val inner = new ConstL12Normalizer(new ArrayTrainable(data.toArray, trainingNetwork, 5000)).setFactor_L1(0.001)
      val trainer = new com.simiacryptus.mindseye.opt.RoundRobinTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(3 * 60, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(1)
      val lbfgs = new LBFGS().setMaxHistory(50).setMinHistory(3)
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
          case _ ⇒ null
        }
      })
      trainer.setLineSearchFactory(Java8Util.cvt((s:String)⇒(s match {
        case s if s.contains("LBFGS") ⇒ new StaticLearningRate().setRate(1)
        case _ ⇒ new LineBracketSearch().setCurrentRate(1e-5)
      }).asInstanceOf[LineSearchStrategy]))
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    trainer.run()
  }: Unit, "oracle")

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