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
import java.util.function.{DoubleUnaryOperator, ToDoubleFunction}

import _root_.util._
import com.simiacryptus.mindseye.layers.NNLayer
import com.simiacryptus.mindseye.layers.activation._
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer
import com.simiacryptus.mindseye.layers.media.{ImgBandBiasLayer, ImgConvolutionSynapseLayer}
import com.simiacryptus.mindseye.layers.reducers.SumInputsLayer
import com.simiacryptus.mindseye.layers.synapse.{BiasLayer, DenseSynapseLayer}
import com.simiacryptus.mindseye.layers.util.{MonitoringSynapse, MonitoringWrapper}
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.region.{LinearSumConstraint, StaticConstraint, TrustRegion, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.trainable.{ScheduledSampleTrainable, SparkTrainable, Trainable}
import com.simiacryptus.util.StreamNanoHTTPD
import com.simiacryptus.util.io.{HtmlNotebookOutput, IOUtil, KryoUtil}
import com.simiacryptus.util.ml.Tensor
import com.simiacryptus.util.test.ImageTiles.ImageTensorLoader
import com.simiacryptus.util.text.TableOutput
import org.apache.spark.sql.SparkSession
import util.Java8Util.cvt

import scala.collection.JavaConverters._
import scala.util.Random


object ImageAutoencodingModeler extends ServiceNotebook {

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒ new ImageAutoencodingModeler(source, server, out).run()
      case _ ⇒ new ImageAutoencodingModeler("E:\\testImages\\256_ObjectCategories", server, out).run()
    })

  }
}

class ImageAutoencodingModeler(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  val corruptors = Map[String, Tensor ⇒ Tensor](
    "noise" → (imgTensor ⇒ {
      imgTensor.map(Java8Util.cvt(v ⇒ v + 0.5 * (Math.random() - 0.5)) : DoubleUnaryOperator)
    })
  )

  lazy val encoder = {
    var network: PipelineNetwork = new PipelineNetwork
    network.add(new ImgBandBiasLayer(3))
    network.add(new ImgConvolutionSynapseLayer(5,5,90)
      .setWeights(cvt(() ⇒ 0.001 * (Math.random()-0.5))))
    network
  }

  lazy val decoder = {
    var network: PipelineNetwork = new PipelineNetwork
    network.add(new ImgConvolutionSynapseLayer(5,5,90)
      .setWeights(cvt(() ⇒ 0.001 * (Math.random()-0.5))))
    network.add(new ImgBandBiasLayer(3))
    network.add(new LinearActivationLayer())
    network
  }

  def train() = {
    val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new MeanSqLossLayer)
    val executor = ScheduledSampleTrainable.Pow(data.toArray, trainingNetwork, 500, 2.0, 0.0).setShuffled(true)
    out.eval {
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(executor)
      trainer.setIterationsPerSample(5)
      trainer.setMonitor(monitor)
      trainer.setTimeout(12, TimeUnit.HOURS)
      trainer.setOrientation(new GradientDescent)
      trainer.setTerminateThreshold(5000.0)
      trainer
    }.run()
    out.eval {
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(executor)
      trainer.setIterationsPerSample(5)
      trainer.setMonitor(monitor)
      trainer.setTimeout(12, TimeUnit.HOURS)
      trainer.setOrientation(new TrustRegionStrategy(new MomentumStrategy(
        new LBFGS().setMinHistory(10).setMaxHistory(20)
      ).setCarryOver(0.2)) {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _: MonitoringWrapper ⇒ getRegionPolicy(layer.asInstanceOf[MonitoringWrapper].inner)
          case _: LinearActivationLayer ⇒ new StaticConstraint
          case _ ⇒ null
        }
      })
      trainer.setTerminateThreshold(0.0)
      trainer
    }.run()
  }

  private val dropoutFactor = 0.3
  val dropoutNoiseLayer = new DropoutNoiseLayer(dropoutFactor)
  val gainAdjLayer = new LinearActivationLayer().freeze().asInstanceOf[LinearActivationLayer]
  lazy val model: PipelineNetwork = {
    var network: PipelineNetwork = new PipelineNetwork
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "input"))
    network.add(new MonitoringWrapper(encoder).addTo(monitoringRoot, "encoder"))
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "encoded"))
    network.add(dropoutNoiseLayer)
    network.add(gainAdjLayer)
    network.add(new MonitoringWrapper(decoder).addTo(monitoringRoot, "decoder"))
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "output"))
    network
  }

  override def onStepComplete(currentPoint: IterativeTrainer.Step): Unit = {
    dropoutNoiseLayer.setValue(0.0)
    gainAdjLayer.setScale(1/(1-dropoutFactor))
    modelCheckpoint = KryoUtil.kryo().copy(model)
    dropoutNoiseLayer.setValue(dropoutFactor)
    gainAdjLayer.setScale(1.0)
    dropoutNoiseLayer.shuffle()
  }

  lazy val data = {
    out.p("Loading data from " + source)
    val loader = new ImageTensorLoader(new File(source), 64, 64, 64, 64, 10, 10)
    val data: List[Array[Tensor]] = loader.stream().iterator().asScala.toStream.flatMap(tile ⇒ corruptors.map(e ⇒ {
      Array(e._2(tile), tile)
    })).take(1000).toList
    loader.stop()
    out.eval {
      TableOutput.create(Random.shuffle(data).take(100).map(testObj ⇒ Map[String, AnyRef](
        "Original" → out.image(testObj(1).toRgbImage(), ""),
        "Distorted" → out.image(testObj(0).toRgbImage(), "")
      ).asJava): _*)
    }
    out.p("Loading data complete")
    data
  }

  var blockAtEnd: Boolean = true

  def run(): Unit = {
    defineMonitorReports()
    require(null != data)
    out.p("<a href='test.html'>Test Reconstruction</a>")
    server.addAsyncHandler("test.html", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(out ⇒ {
        try {
          out.eval {
            TableOutput.create(Random.shuffle(data).take(10).map(testObj ⇒ Map[String, AnyRef](
              "Original" → out.image(testObj(1).toRgbImage(), ""),
              "Distorted" → out.image(testObj(0).toRgbImage(), ""),
              "Reconstructed" → out.image(getModelCheckpoint.eval(testObj(0)).data.head.toRgbImage(), "")
            ).asJava): _*)
          }
        } catch {
          case e : Throwable ⇒ e.printStackTrace()
        }
      })
    }), false)
    out.out("<hr/>")
    train()
    IOUtil.writeKryo(model, out.file("model_final.kryo"))
    IOUtil.writeString(model.getJsonString, out.file("../model_final.json"))
    summarizeHistory(out)
    out.out("<hr/>")
    if(blockAtEnd) waitForExit(out)
  }

  def resize(source: BufferedImage, size: Int) = {
    val image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB)
    val graphics = image.getGraphics.asInstanceOf[Graphics2D]
    graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
    graphics.drawImage(source, 0, 0, size, size, null)
    image
  }

  def toOut(label: String, max: Int): Int = {
    (0 until max).find(label == "[" + _ + "]").get
  }

  def toOutNDArray(out: Int, max: Int): Tensor = {
    val ndArray = new Tensor(max)
    ndArray.set(out, 1)
    ndArray
  }

}