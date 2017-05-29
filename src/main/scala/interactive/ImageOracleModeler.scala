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
import java.util.concurrent.{Semaphore, TimeUnit}

import _root_.util._
import com.aparapi.internal.kernel.KernelManager
import com.fasterxml.jackson.databind.ObjectMapper
import com.simiacryptus.mindseye.layers.NNLayer
import com.simiacryptus.mindseye.layers.activation.ReLuActivationLayer
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer
import com.simiacryptus.mindseye.layers.media.{ImgBandBiasLayer, ImgConvolutionSynapseLayer}
import com.simiacryptus.mindseye.layers.util.{MonitoringSynapse, MonitoringWrapper}
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeConditions
import com.simiacryptus.mindseye.opt.region.{TrustRegionStrategy, LinearSumConstraint, TrustRegion}
import com.simiacryptus.mindseye.opt.trainable.{ConstL12Normalizer, SparkTrainable, StochasticArrayTrainable, Trainable}
import com.simiacryptus.util.io.{HtmlNotebookOutput, IOUtil, TeeOutputStream}
import com.simiacryptus.util.ml.Tensor
import com.simiacryptus.util.test.ImageTiles.ImageTensorLoader
import com.simiacryptus.util.text.TableOutput
import com.simiacryptus.util.{MonitoredObject, StreamNanoHTTPD, Util}
import fi.iki.elonen.NanoHTTPD
import fi.iki.elonen.NanoHTTPD.IHTTPSession
import org.apache.spark.sql.SparkSession
import smile.plot.{PlotCanvas, ScatterPlot}
import util.Java8Util.cvt

import scala.collection.JavaConverters._
import scala.util.Random

object ImageOracleModeler extends ServiceNotebook {

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source, master) ⇒ new ImageOracleModeler(source, server, out).runSpark(master)
      case Array(source) ⇒ new ImageOracleModeler(source, server, out).runLocal()
      case _ ⇒ new ImageOracleModeler("E:\\testImages\\256_ObjectCategories", server, out).runLocal()
    })

  }
}

class ImageOracleModeler(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) {

  val corruptors = Map[String, Tensor ⇒ Tensor](
    "resample4x" → (imgTensor ⇒ {
      Tensor.fromRGB(resize(resize(imgTensor.toRgbImage, 16), 64))
    })
  )

  private def corruptionDetectionModel(monitoringRoot: MonitoredObject) = {
    var network: PipelineNetwork = new PipelineNetwork

//    network.add(new MonitoringWrapper(new InceptionLayer(Array(
//      Array(Array(1, 1, 9)),
//      Array(Array(3, 3, 6)),
//      Array(Array(5, 5, 3))
//    )).setWeights(cvt(() ⇒ Util.R.get.nextGaussian * 0.01)))
//      .addTo(monitoringRoot, "inception_1"))
//    network.add(new ImgBandBiasLayer(64,64,6))
//    network.add(new ReLuActivationLayer())
//    network.add(new MonitoringSynapse().addTo(monitoringRoot, "activationLayer1"))

//    network.add(new MonitoringWrapper(new InceptionLayer(Array(
//      Array(Array(1, 1, 36)),
//      Array(Array(3, 3, 12)),
//      Array(Array(5, 5, 6))
//    )).setWeights(cvt(() ⇒ Util.R.get.nextGaussian * 0.01)))
//      .addTo(monitoringRoot, "inception_2"))
//    network.add(new ImgBandBiasLayer(64,64,9))
//    network.add(new ReLuActivationLayer())
//    network.add(new MonitoringSynapse().addTo(monitoringRoot, "activationLayer2"))

//    network.add(new MonitoringWrapper(new InceptionLayer(Array(
//      Array(Array(1, 1, 81)),
//      Array(Array(3, 3, 18)),
//      Array(Array(5, 5, 9))
//    )).setWeights(cvt(() ⇒ Util.R.get.nextGaussian * 0.01)))
//      .addTo(monitoringRoot, "inception_3"))
//    network.add(new ImgBandBiasLayer(64,64,12))
//    network.add(new ReLuActivationLayer())
//    network.add(new MonitoringSynapse().addTo(monitoringRoot, "activationLayer3"))

    network.add(new MonitoringWrapper(
      new ImgBandBiasLayer(3))
      .addTo(monitoringRoot, "Bias1"));
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "input"))

    network.add(new MonitoringWrapper(
      new ImgConvolutionSynapseLayer(3,3,18).setWeights(Java8Util.cvt(() ⇒ Util.R.get.nextGaussian * 0.01)))
      .addTo(monitoringRoot, "Conv1"))
    network.add(new ReLuActivationLayer())
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "activationLayer1"))
    network.add(new MonitoringWrapper(
      new ImgConvolutionSynapseLayer(1,1,18).setWeights(Java8Util.cvt(() ⇒ Util.R.get.nextGaussian * 0.01)))
      .addTo(monitoringRoot, "Conv2"))
    network.add(new ReLuActivationLayer())
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "activationLayer2"))
    network.add(new MonitoringWrapper(
      new ImgBandBiasLayer(3))
      .addTo(monitoringRoot, "Bias2"))
    network
  }

  private def train(data: List[Array[Tensor]],
                    model: PipelineNetwork,
                    executorFactory: (List[Array[Tensor]], SupervisedNetwork) ⇒ Trainable) =
  {
    val monitor = new TrainingMonitor {
      var lastCheckpoint = System.currentTimeMillis()

      override def log(msg: String): Unit = {
        println(msg)
        logPrintStream.println(msg)
        logPrintStream.flush()
      }

      override def onStepComplete(currentPoint: IterativeTrainer.Step): Unit = {
        history += currentPoint
        if ((System.currentTimeMillis() - lastCheckpoint) > TimeUnit.MINUTES.toMillis(5)) {
          lastCheckpoint = System.currentTimeMillis()
          IOUtil.writeKryo(model, out.file("model_checkpoint_" + currentPoint.iteration + ".kryo"))
        }
      }
    }
    val trainer = out.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new MeanSqLossLayer)
      val inner = executorFactory(data, trainingNetwork)
      val normalized = new ConstL12Normalizer(inner).setFactor_L1(0.01)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(72, TimeUnit.HOURS)
      trainer.setOrientation(new TrustRegionStrategy(new LBFGS().setMinHistory(10).setMaxHistory(20)) {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _:MonitoringWrapper ⇒ getRegionPolicy(layer.asInstanceOf[MonitoringWrapper].inner)
          //case _:ImgConvolutionSynapseLayer ⇒ new GrowthSphere().setGrowthFactor(1.0).setMinRadius(0.0)
          case _:ImgConvolutionSynapseLayer ⇒ new LinearSumConstraint
          //case _:DenseSynapseLayer ⇒ new SingleOrthant()
          case _ ⇒ null
        }
      })
      trainer.setScaling(new ArmijoWolfeConditions().setMaxAlpha(5))
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    trainer.run()
  }

  def runSpark(masterUrl: String): Unit = {
    var builder = SparkSession.builder
      .appName("Spark MindsEye Demo")
    builder = masterUrl match {
      case "auto" ⇒ builder
      case _ ⇒ builder.master(masterUrl)
    }
    val sparkSession = builder.getOrCreate()
    run((data, network) ⇒ new SparkTrainable(sparkSession.sparkContext.makeRDD(data, 8), network))
    sparkSession.stop()
  }

  def runLocal(): Unit = {
    run((data, network) ⇒ new StochasticArrayTrainable(data.toArray, network, 100))
    //run((data,network)⇒new ArrayTrainable(data.toArray, network))
  }

  private def loadData(out: HtmlNotebookOutput with ScalaNotebookOutput, model: PipelineNetwork) = {
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
    out.p("<a href='test.html'>Test Reconstruction</a>")
    server.addAsyncHandler("test.html", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(out ⇒ {
        try {
          out.eval {
            TableOutput.create(Random.shuffle(data).take(100).map(testObj ⇒ Map[String, AnyRef](
              "Original" → out.image(testObj(1).toRgbImage(), ""),
              "Distorted" → out.image(testObj(0).toRgbImage(), ""),
              "Reconstructed" → out.image(model.eval(testObj(0)).data.head.toRgbImage(), "")
            ).asJava): _*)
          }
        } catch {
          case e : Throwable ⇒ e.printStackTrace()
        }
      })
    }), false)
    out.p("Loading data complete")
    data
  }

  def run(executor: (List[Array[Tensor]], SupervisedNetwork) ⇒ Trainable): Unit = {
    out.p("View the convergence history: <a href='/history.html'>/history.html</a>")
    out.p("<a href='/netmon.json'>Network Monitoring</a>")
    out.p("View the log: <a href='/log'>/log</a>")
    out.out("<hr/>")
    var model: PipelineNetwork = corruptionDetectionModel(monitoringRoot)
    val data: List[Array[Tensor]] = loadData(out, model)
    train(data, model, executor)
    IOUtil.writeKryo(model, out.file("model_final.kryo"))
    summarizeHistory(out, history.toList)
    out.out("<hr/>")
    logOut.close()
    val onExit = new Semaphore(0)
    out.p("To exit the sever: <a href='/exit'>/exit</a>")
    server.addAsyncHandler("exit", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(log ⇒ {
        log.h1("OK")
        onExit.release(1)
      })
    }), false)
    onExit.acquire()
  }

  val logOut = new TeeOutputStream(new FileOutputStream("training.log", true), true)
  val history = new scala.collection.mutable.ArrayBuffer[IterativeTrainer.Step]()
  server.addAsyncHandler("history.html", "text/html", cvt(o ⇒ {
    Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(log ⇒ {
      summarizeHistory(log, history.toList)
    })
  }), false)
  val monitoringRoot = new MonitoredObject()
  monitoringRoot.addField("openCL",Java8Util.cvt(()⇒{
    val sb = new java.lang.StringBuilder()
    KernelManager.instance().reportDeviceUsage(sb,true)
    sb.toString()
  }))
  server.addAsyncHandler("netmon.json", "application/json", cvt(out ⇒ {
    val mapper = new ObjectMapper().enableDefaultTyping(ObjectMapper.DefaultTyping.NON_FINAL)
    val buffer = new ByteArrayOutputStream()
    mapper.writeValue(buffer, monitoringRoot.getMetrics)
    out.write(buffer.toByteArray)
  }), false)
  val logPrintStream = new PrintStream(logOut)
  server.addSessionHandler("log", cvt((session: IHTTPSession) ⇒ {
    NanoHTTPD.newChunkedResponse(NanoHTTPD.Response.Status.OK, "text/plain", logOut.newInputStream())
  }))

  def resize(source: BufferedImage, size: Int) = {
    val image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB)
    val graphics = image.getGraphics.asInstanceOf[Graphics2D]
    graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
    graphics.drawImage(source, 0, 0, size, size, null)
    image
  }

  private def summarizeHistory(log: ScalaNotebookOutput, history: List[com.simiacryptus.mindseye.opt.IterativeTrainer.Step]) = {
    if (!history.isEmpty) {
      log.eval {
        val step = Math.max(Math.pow(10, Math.ceil(Math.log(history.size) / Math.log(10)) - 2), 1).toInt
        TableOutput.create(history.filter(0 == _.iteration % step).map(state ⇒
          Map[String, AnyRef](
            "iteration" → state.iteration.toInt.asInstanceOf[Integer],
            "time" → state.time.toDouble.asInstanceOf[lang.Double],
            "fitness" → state.point.value.toDouble.asInstanceOf[lang.Double]
          ).asJava
        ): _*)
      }
      log.eval {
        val plot: PlotCanvas = ScatterPlot.plot(history.map(item ⇒ Array[Double](
          item.iteration, Math.log(item.point.value)
        )).toArray: _*)
        plot.setTitle("Convergence Plot")
        plot.setAxisLabels("Iteration", "log(Fitness)")
        plot.setSize(600, 400)
        plot
      }
    }
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