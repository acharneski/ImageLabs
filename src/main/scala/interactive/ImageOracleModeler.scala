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

import _root_.util._
import com.simiacryptus.mindseye.layers.NNLayer
import com.simiacryptus.mindseye.layers.activation.{HyperbolicActivationLayer, LinearActivationLayer, ReLuActivationLayer}
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer
import com.simiacryptus.mindseye.layers.media.{ImgBandBiasLayer, ImgConvolutionSynapseLayer}
import com.simiacryptus.mindseye.layers.reducers.{ProductInputsLayer, SumInputsLayer}
import com.simiacryptus.mindseye.layers.util.{ConstNNLayer, MonitoringSynapse, MonitoringWrapper}
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line.{ArmijoWolfeConditions, LineBracketSearch, LineSearchStrategy}
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
    val loader = new ImageTensorLoader(new File(source), 64, 64, 64, 64, 10, 10)
    val data: List[Array[Tensor]] = loader.stream().iterator().asScala.toStream.flatMap(tile ⇒ corruptors.map(e ⇒ {
      Array(e._2(tile), tile)
    })).take(1000).toList
    loader.stop()
    out.eval {
      TableOutput.create(Random.shuffle(data).take(100).map(testObj ⇒ Map[String, AnyRef](
        "Source" → out.image(testObj(1).toRgbImage(), ""),
        "Transformed" → out.image(testObj(0).toRgbImage(), "")
      ).asJava): _*)
    }
    out.p("Loading data complete")
    data
  }

  def step1() = phase({
    var network: PipelineNetwork = new PipelineNetwork

    network.add(new MonitoringSynapse().addTo(monitoringRoot, "input1"))
    network.add(new MonitoringWrapper(new ImgBandBiasLayer(3).setWeights(Java8Util.cvt(i⇒0.0)).setName("Bias1In")).addTo(monitoringRoot));
    network.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(5,5,18).setWeights(Java8Util.cvt(() ⇒ Util.R.get.nextGaussian * 0.01))
      .setName("Conv1a")).addTo(monitoringRoot));
    //network.add(new MonitoringWrapper(new HyperbolicActivationLayer().setName("Activation1")).addTo(monitoringRoot));
    network.add(new MonitoringWrapper(new ReLuActivationLayer().setName("Activation1")).addTo(monitoringRoot));
    network.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(5,5,18).setWeights(Java8Util.cvt(() ⇒ Util.R.get.nextGaussian * 0.01))
      .setName("Conv2a")).addTo(monitoringRoot));

//    network.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(5,5,18).setWeights(Java8Util.cvt(() ⇒ Util.R.get.nextGaussian * 0.01))
//      .setName("Conv1a")).addTo(monitoringRoot));
//    network.add(new MonitoringWrapper(new HyperbolicActivationLayer().setName("Activation1")).addTo(monitoringRoot));
//    network.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(5,5,18).setWeights(Java8Util.cvt(() ⇒ Util.R.get.nextGaussian * 0.01))
//      .setName("Conv1b")).addTo(monitoringRoot));
//    network.add(new MonitoringWrapper(new ImgBandBiasLayer(3).setName("Bias1Out")).addTo(monitoringRoot));

    network.add(new SumInputsLayer(), network.getInput(0), network.getHead)
    network
  }, (model: NNLayer) ⇒ {
    val trainer = out.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, lossNetwork)
      val inner = new StochasticArrayTrainable(data.toArray, trainingNetwork, 2500)
      val trainer = new com.simiacryptus.mindseye.opt.RoundRobinTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(60, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(3)
      val lbfgs = new LBFGS().setMaxHistory(50).setMinHistory(5)
      trainer.setOrientations(new TrustRegionStrategy(lbfgs) {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _: HyperbolicActivationLayer ⇒ new StaticConstraint
          case _: ImgBandBiasLayer ⇒ new StaticConstraint
          case x: ImgConvolutionSynapseLayer if x.getName() == "Conv1a" ⇒ new StaticConstraint
          case _ ⇒ null
        }
      }, new TrustRegionStrategy(lbfgs) {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _: HyperbolicActivationLayer ⇒ new StaticConstraint
          case _: ReLuActivationLayer ⇒ new StaticConstraint
          case _: ImgBandBiasLayer ⇒ new StaticConstraint
          case x: ImgConvolutionSynapseLayer if x.getName() == "Conv2a" ⇒ new StaticConstraint
          case _ ⇒ null
        }
      })
//      trainer.setOrientations(new TrustRegionStrategy(new GradientDescent()) {
//        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
//          case _: HyperbolicActivationLayer ⇒ new StaticConstraint
//          case _: ImgBandBiasLayer ⇒ new LinearSumConstraint
//          case _ ⇒ null
//        }
//      })
      //trainer.setOrientations(new GradientDescent())
      trainer.setLineSearchFactory(Java8Util.cvt((s:String)⇒(s match {
        case s if s.contains("LBFGS") ⇒ new LineBracketSearch().setCurrentRate(1)
        case _ ⇒ new LineBracketSearch().setCurrentRate(1e-5)
      }).asInstanceOf[LineSearchStrategy]))
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    trainer.run()
  }: Unit, "oracle")

  def lossNetwork = {
    val radius = 4
    val mask: Tensor = new Tensor(64, 64, 3).map(Java8Util.cvt((v: lang.Double, c: Coordinate) ⇒ {
      if (c.coords(0) < radius || c.coords(0) >= (64 - radius)) {
        0.0
      } else if (c.coords(1) < radius || c.coords(1) >= (64 - radius)) {
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

  def step2() = phase("oracle", (model: NNLayer) ⇒ {
    val trainer = out.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new MeanSqLossLayer)
      val inner = new StochasticArrayTrainable(data.toArray, trainingNetwork, 500)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(3, TimeUnit.HOURS)
      trainer.setIterationsPerSample(5)
      trainer.setOrientation(new TrustRegionStrategy(new MomentumStrategy(
        new LBFGS().setMinHistory(10).setMaxHistory(30)
      ).setCarryOver(0.2)) {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _: HyperbolicActivationLayer ⇒ new StaticConstraint
          case _: ImgBandBiasLayer ⇒ new StaticConstraint
          case _ ⇒ null
        }
      })
      trainer.setLineSearchFactory(()⇒new ArmijoWolfeConditions().setMaxAlpha(5))
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    trainer.run()
  }: Unit, "oracle")

  def run(): Unit = {
    defineHeader()
    defineTestHandler()
    out.out("<hr/>")
    step1()
    step2()
    summarizeHistory()
    out.out("<hr/>")
    waitForExit()
  }

  def defineTestHandler() = {
    out.p("<a href='test.html'>Test Reconstruction</a>")
    server.addAsyncHandler("test.html", "text/html", cvt(o ⇒ {
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