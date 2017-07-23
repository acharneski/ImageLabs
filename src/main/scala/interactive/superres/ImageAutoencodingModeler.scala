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
import java.util.concurrent.TimeUnit

import _root_.util._
import com.simiacryptus.mindseye.layers.activation._
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer
import com.simiacryptus.mindseye.layers.media.{ImgBandBiasLayer, ImgConvolutionSynapseLayer}
import com.simiacryptus.mindseye.layers.util.{MonitoringSynapse, MonitoringWrapper, VariableLayer}
import com.simiacryptus.mindseye.layers.{NNLayer, NNResult}
import com.simiacryptus.mindseye.network.graph.DAGNetwork
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.orient.{LBFGS, MomentumStrategy, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.{LinearSumConstraint, StaticConstraint, TrustRegion}
import com.simiacryptus.mindseye.opt.trainable.StochasticArrayTrainable
import com.simiacryptus.util.StreamNanoHTTPD
import com.simiacryptus.util.io.HtmlNotebookOutput
import com.simiacryptus.util.ml.Tensor
import com.simiacryptus.util.test.ImageTiles.ImageTensorLoader
import com.simiacryptus.util.text.TableOutput
import util.Java8Util.cvt

import scala.collection.JavaConverters._
import scala.util.Random


object ImageAutoencodingModeler extends Report {

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒ new ImageAutoencodingModeler(source, server, out).run()
      case _ ⇒ new ImageAutoencodingModeler("E:\\testImages\\256_ObjectCategories", server, out).run()
    })

  }
}

class ImageAutoencodingModeler(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  def run(): Unit = {
    defineHeader()
    require(null != data)
    //if(findFile("autoencoder").isEmpty)
    initialize()
    addLayer(0, 30, 10)
    addLayer(1, 10, 5)
    addLayer(2, 5, 3)
    waitForExit()
  }

  val corruptors = {
    def resize(source: BufferedImage, size: Int) = {
      val image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB)
      val graphics = image.getGraphics.asInstanceOf[Graphics2D]
      graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
      graphics.drawImage(source, 0, 0, size, size, null)
      image
    }
    Map[String, Tensor ⇒ Tensor](
      "noise" → (imgTensor ⇒ {
        imgTensor.map(Java8Util.cvt((x:Double)⇒Math.min(Math.max(x+(50.0*(Random.nextDouble()-0.5)), 0.0), 256.0)))
      }), "resample" → (imgTensor ⇒ {
        Tensor.fromRGB(resize(resize(imgTensor.toRgbImage, 8), 32))
      })
    )
  }

  val dropoutFactor = 0.3

  def initialize() = phase({
    val encoder = {
      var encoder: PipelineNetwork = new PipelineNetwork
      encoder.add(new ImgBandBiasLayer(3).setName("encoder_bias_0"))
      encoder.add(new ImgConvolutionSynapseLayer(5,5,90)
        .setWeights(cvt(() ⇒ 0.001 * (Math.random()-0.5))).setName("encoder_conv_0"))
      encoder
    }
    val decoder = {
      var decoder: PipelineNetwork = new PipelineNetwork
      decoder.add(new ImgConvolutionSynapseLayer(5,5,90)
        .setWeights(cvt(() ⇒ 0.001 * (Math.random()-0.5))).setName("decoder_conv_0"))
      decoder.add(new ImgBandBiasLayer(3).setName("decoder_bias_0"))
      decoder
    }
    buildCodecNetwork(encoder, decoder)
  },(model:NNLayer)⇒{
    out.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new MeanSqLossLayer)
      val executor = new StochasticArrayTrainable(data.toArray, trainingNetwork, 100)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(executor)
      trainer.setMonitor(autoencodingMonitor(model))
      trainer.setTimeout(90, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(5)
      trainer.setOrientation(new TrustRegionStrategy(new MomentumStrategy(
        new LBFGS().setMinHistory(10).setMaxHistory(30)
      ).setCarryOver(0.2)) {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _: LinearActivationLayer ⇒ new StaticConstraint
          case _: ImgBandBiasLayer ⇒ new LinearSumConstraint
          case _ ⇒ null
        }
      })
      trainer.setTerminateThreshold(350.0)
      trainer.run()
    }
  },"autoencoder0")

  private def buildCodecNetwork(encoder: PipelineNetwork, decoder: PipelineNetwork) = {
    var network: PipelineNetwork = new PipelineNetwork
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "input"))
    network.add(new MonitoringWrapper(encoder.setName("encoder")).addTo(monitoringRoot))
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "encoded"))
    network.add(new DropoutNoiseLayer(dropoutFactor).setName("dropoutNoise"))
    network.add(new LinearActivationLayer().freeze().setName("gain"))
    network.add(new MonitoringWrapper(new VariableLayer(decoder).setName("decoder")).addTo(monitoringRoot))
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "output"))
    network
  }

  def addLayer(i : Int, fromBands : Int, toBands : Int) = phase("autoencoder"+i, (model:NNLayer)⇒{

    val encoder = {
      var encoder: PipelineNetwork = new PipelineNetwork
      encoder.add(new ImgBandBiasLayer(fromBands).setName("encoder_bias_"+(i+1)))
      encoder.add(new ImgConvolutionSynapseLayer(5,5,toBands*fromBands)
        .setWeights(cvt(() ⇒ 0.001 * (Math.random()-0.5))).setName("encoder_conv_"+(i+1)))
      encoder
    }
    val decoder = {
      var decoder: PipelineNetwork = new PipelineNetwork
      decoder.add(new ImgConvolutionSynapseLayer(5,5,toBands*fromBands)
        .setWeights(cvt(() ⇒ 0.001 * (Math.random()-0.5))).setName("decoder_conv_"+(i+1)))
      decoder.add(new ImgBandBiasLayer(fromBands).setName("decoder_bias_"+(i+1)))
      decoder
    }

    val encoderPipeline = model.asInstanceOf[DAGNetwork].getByName[PipelineNetwork]("encoder")
    val transformedData = encoderPipeline.eval(NNResult.singleResultArray(Array(data.map(x ⇒ x.head).toArray)): _*).data.map(x⇒Array(x,x))
    encoder.getLayers().asScala.foreach(encoderPipeline.add)

    val decoderPlaceholder = model.asInstanceOf[DAGNetwork].getByName[VariableLayer]("decoder")
    val prevDecoder = decoderPlaceholder.getInner.asInstanceOf[PipelineNetwork]
    val decoderPipeline = new PipelineNetwork()
    decoder.getLayers().asScala.foreach(decoderPipeline.add)
    prevDecoder.getLayers().asScala.foreach(decoderPipeline.add)
    decoderPlaceholder.setInner(decoderPipeline)

    out.p("Training new layer in isolation")

    {
      val codecNetwork = buildCodecNetwork(encoder, decoder)
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(codecNetwork, new MeanSqLossLayer)
      val executor = new StochasticArrayTrainable(transformedData, trainingNetwork, 100)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(executor)
      trainer.setIterationsPerSample(5)
      trainer.setMonitor(autoencodingMonitor(codecNetwork))
      trainer.setTimeout(90, TimeUnit.MINUTES)
      trainer.setOrientation(new TrustRegionStrategy(new MomentumStrategy(
        new LBFGS().setMinHistory(10).setMaxHistory(30)
      ).setCarryOver(0.2)) {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _: LinearActivationLayer ⇒ new StaticConstraint
          case _: ImgBandBiasLayer ⇒ new LinearSumConstraint
          case _ ⇒ null
        }
      })
      trainer.setTerminateThreshold(0.0)
      trainer.run()
    }

    out.p("Integration training")

    {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new MeanSqLossLayer)
      val executor = new StochasticArrayTrainable(data.toArray, trainingNetwork, 100)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(executor)
      trainer.setIterationsPerSample(5)
      trainer.setMonitor(autoencodingMonitor(model))
      trainer.setTimeout(90, TimeUnit.MINUTES)
      trainer.setOrientation(new TrustRegionStrategy(new MomentumStrategy(
        new LBFGS().setMinHistory(10).setMaxHistory(30)
      ).setCarryOver(0.2)) {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _: LinearActivationLayer ⇒ new StaticConstraint
          case _: ImgBandBiasLayer ⇒ new LinearSumConstraint
          case _ ⇒ null
        }
      })
      trainer.setTerminateThreshold(350.0)
      trainer.run()
    }

  },"autoencoder"+(i+1))

  def autoencodingMonitor(model:NNLayer) = {
    val dropoutNoiseLayer = model.asInstanceOf[DAGNetwork].getByName[DropoutNoiseLayer]("dropoutNoise")
    val gainAdjLayer = model.asInstanceOf[DAGNetwork].getByName[LinearActivationLayer]("gain")
    new TrainingMonitor() {
      override def log(msg: String): Unit = monitor.log(msg)

      override def onStepComplete(currentPoint: Step): Unit = {
        dropoutNoiseLayer.setValue(0.0)
        gainAdjLayer.setScale(1.0)
        monitor.onStepComplete(currentPoint)
        dropoutNoiseLayer.setValue(dropoutFactor)
        gainAdjLayer.setScale(1 / (1 - dropoutFactor))
        dropoutNoiseLayer.shuffle()
      }
    }
  }

  lazy val data = {
    out.p("Loading data from " + source)
    val loader = new ImageTensorLoader(new File(source), 32, 32, 32, 32, 10, 10)
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


  override def defineReports(log: HtmlNotebookOutput with ScalaNotebookOutput) = {
    log.p("Interactive Reports: <a href='/history.html'>Convergence History</a> <a href='/test.html'>Model Validation</a>")
    server.addSyncHandler("test.html", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(out ⇒ {
        try {
          val model = getModelCheckpoint.asInstanceOf[DAGNetwork]
          val dropoutNoiseLayer = model.getByName[DropoutNoiseLayer]("dropoutNoise")
          val gainAdjLayer = model.getByName[LinearActivationLayer]("gain")
          dropoutNoiseLayer.setValue(0.0)
          gainAdjLayer.setScale(1.0)
          out.eval {
            TableOutput.create(Random.shuffle(data).take(10).map(testObj ⇒ {
              Map[String, AnyRef](
                "Original" → out.image(testObj(1).toRgbImage(), ""),
                "Distorted" → out.image(testObj(0).toRgbImage(), ""),
                "Reconstructed" → out.image(model.eval(testObj(0)).data.head.toRgbImage(), "")
              ).asJava
            }): _*)
          }
        } catch {
          case e : Throwable ⇒ e.printStackTrace()
        }
      })
    }), false)
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