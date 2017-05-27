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
import java.io.{ByteArrayOutputStream, File, FileOutputStream, PrintStream}
import java.lang
import java.util.concurrent.{Semaphore, TimeUnit}

import _root_.util._
import com.aparapi.internal.kernel.KernelManager
import com.fasterxml.jackson.databind.ObjectMapper
import com.simiacryptus.mindseye.layers.NNLayer
import com.simiacryptus.mindseye.layers.activation.{ReLuActivationLayer, SoftmaxActivationLayer}
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer
import com.simiacryptus.mindseye.layers.media.{ImgConvolutionSynapseLayer, MaxSubsampleLayer, SumSubsampleLayer}
import com.simiacryptus.mindseye.layers.synapse.{BiasLayer, DenseSynapseLayer}
import com.simiacryptus.mindseye.layers.util.{MonitoringSynapse, MonitoringWrapper}
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.region.{LayerTrustRegion, LinearSumConstraint, TrustRegion}
import com.simiacryptus.mindseye.opt.trainable.{ConstL12Normalizer, ScheduledSampleTrainable, SparkTrainable, Trainable}
import com.simiacryptus.util.io.{HtmlNotebookOutput, IOUtil, TeeOutputStream}
import com.simiacryptus.util.ml.Tensor
import com.simiacryptus.util.test.ImageTiles.ImageTensorLoader
import com.simiacryptus.util.test.LabeledObject
import com.simiacryptus.util.text.TableOutput
import com.simiacryptus.util.{MonitoredObject, StreamNanoHTTPD}
import fi.iki.elonen.NanoHTTPD
import fi.iki.elonen.NanoHTTPD.IHTTPSession
import org.apache.spark.sql.SparkSession
import smile.plot.{PlotCanvas, ScatterPlot}
import util.Java8Util._

import scala.collection.JavaConverters._
import scala.util.Random


object ImageCorruptionModeler extends ServiceNotebook {

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source, master) ⇒ new ImageCorruptionModeler(source, server, out).runSpark(master)
      case Array(source) ⇒ new ImageCorruptionModeler(source, server, out).runLocal()
      case _ ⇒ new ImageCorruptionModeler("E:\\testImages\\256_ObjectCategories", server, out).runLocal()
    })

  }
}

class ImageCorruptionModeler(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) {

  val corruptors = Map[String, Tensor ⇒ Tensor](
    "resample2x" → (imgTensor ⇒ {
      Tensor.fromRGB(resize(resize(imgTensor.toRgbImage, 32), 64))
    }),"resample4x" → (imgTensor ⇒ {
      Tensor.fromRGB(resize(resize(imgTensor.toRgbImage, 16), 64))
    })
  )

  private def corruptionDetectionModel(categoryCount: Int, monitoringRoot: MonitoredObject) = {
    val outputSize = Array[Int](categoryCount)
    var network: PipelineNetwork = new PipelineNetwork

//    // 64 x 64 x 3 (RGB)
//    network.add(new MonitoringWrapper(new InceptionLayer(Array(
//      Array(Array(5, 5, 3)),
//      Array(Array(3, 3, 9))
//    )).setWeights(cvt(() ⇒ Util.R.get.nextGaussian * 0.01)))
//      .addTo(monitoringRoot, "inception_1"))
//    network.add(new MaxSubsampleLayer(2, 2, 1))
//    // 32 x 32 x 4
//    network.add(new MonitoringWrapper(new InceptionLayer(Array(
//      Array(Array(5, 5, 4)),
//      Array(Array(3, 3, 16))
//    )).setWeights(cvt(() ⇒ Util.R.get.nextGaussian * 0.01)))
//      .addTo(monitoringRoot, "inception_2"))
//    network.add(new MaxSubsampleLayer(2, 2, 1))
//    // 16 x 16 x 5
//    network.add(new MonitoringWrapper(new InceptionLayer(Array(
//      Array(Array(5, 5, 5)),
//      Array(Array(3, 3, 25))
//    )).setWeights(cvt(() ⇒ Util.R.get.nextGaussian * 0.01)))
//      .addTo(monitoringRoot, "inception_3"))
//    network.add(new MaxSubsampleLayer(2, 2, 1))
//    // 8 x 8 x 6
//    network.add(new MonitoringWrapper(new InceptionLayer(Array(
//      Array(Array(5, 5, 12)),
//      Array(Array(3, 3, 36))
//    )).setWeights(cvt(() ⇒ Util.R.get.nextGaussian * 0.01)))
//      .addTo(monitoringRoot, "inception_4"))
//    network.add(new MaxSubsampleLayer(2, 2, 1))
//    // 4 x 4 x 8
//    network.add(new MonitoringWrapper(new InceptionLayer(Array(
//      Array(Array(5, 5, 64)),
//      Array(Array(3, 3, 64))
//    )).setWeights(cvt(() ⇒ Util.R.get.nextGaussian * 0.01)))
//      .addTo(monitoringRoot, "inception_5"))
//    network.add(new MaxSubsampleLayer(2, 2, 1))
//    // 2 x 2 x 16
//    network.add(new MonitoringWrapper(
//      new DenseSynapseLayer(Array[Int](2,2,16), outputSize)
//      .setWeights(cvt(() ⇒ Util.R.get.nextGaussian * 0.01)))
//      .addTo(monitoringRoot, "final_dense"))
//    network.add(new BiasLayer(outputSize: _*))

    network.add(new MonitoringWrapper(new BiasLayer(64,64,3)).addTo(monitoringRoot, "inbias"))
    network.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(3,3,18)
      .setWeights(Java8Util.cvt(() ⇒ 0.1 * Random.nextGaussian()))).addTo(monitoringRoot, "synapse1"))
    network.add(new MonitoringWrapper(new ReLuActivationLayer).addTo(monitoringRoot, "relu1"))
    network.add(new MonitoringWrapper(new MaxSubsampleLayer(2,2,1)).addTo(monitoringRoot, "max1"))
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "output1"))
    network.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(3,3,60)
      .setWeights(Java8Util.cvt(() ⇒ 0.05 * Random.nextGaussian()))).addTo(monitoringRoot, "synapse2"))
    network.add(new MonitoringWrapper(new ReLuActivationLayer).addTo(monitoringRoot, "relu2"))
    network.add(new MonitoringWrapper(new SumSubsampleLayer(32,32,1)).addTo(monitoringRoot, "max2"))
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "output2"))
    network.add(new MonitoringWrapper(new DenseSynapseLayer(Array[Int](1,1,10), outputSize)
      .setWeights(Java8Util.cvt(() ⇒ 0.1 * Random.nextGaussian()))).addTo(monitoringRoot, "synapse3"))
    network.add(new MonitoringWrapper(new ReLuActivationLayer).addTo(monitoringRoot, "relu3"))
    network.add(new MonitoringWrapper(new BiasLayer(outputSize: _*)).addTo(monitoringRoot, "outbias"))
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "output3"))
    network.add(new SoftmaxActivationLayer)
    network
  }
  val dataTable = new TableOutput()

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
        def flatten(prefix:String,data:Map[String,AnyRef]) : Map[String,AnyRef] = {
          data.flatMap({
            case (key, value) ⇒ value match {
              case value : java.util.Map[String,AnyRef] ⇒ flatten(prefix+key+".", value.asScala.toMap)
              case value : Map[String,AnyRef] ⇒ flatten(prefix+key, value)
              case value if prefix.isEmpty ⇒ Map(key → value)
              case value ⇒ Map((prefix + key) → value)
            }
          }).map(e⇒(if(e._1.startsWith(".")) e._1.substring(1) else e._1)→e._2)
        }
        dataTable.putRow((flatten(".",monitoringRoot.getMetrics.asScala.toMap)++Map(
          "epoch" → currentPoint.iteration.asInstanceOf[java.lang.Long],
          "time" → currentPoint.time.asInstanceOf[java.lang.Long],
          "value" → currentPoint.point.value.asInstanceOf[java.lang.Double]
        )).asJava)
      }
    }
    out.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
      val executorFunction = executorFactory(data, trainingNetwork)
      val effectiveFunction = executorFunction
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(executorFunction)
      trainer.setOrientation(new LayerTrustRegion(new LBFGS().setMinHistory(10).setMaxHistory(30)) {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _: MonitoringWrapper ⇒ getRegionPolicy(layer.asInstanceOf[MonitoringWrapper].inner)
          case _: DenseSynapseLayer ⇒ new LinearSumConstraint()
          case _: ImgConvolutionSynapseLayer ⇒ new LinearSumConstraint()
          case _ ⇒ null
        }
      })
      trainer.setMonitor(monitor)
      trainer.setTimeout(1, TimeUnit.HOURS)
      trainer.setTerminateThreshold(Double.NegativeInfinity)
      trainer.setMaxIterations(20)
      trainer
    }.run()
    out.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
      val executorFunction = executorFactory(data, trainingNetwork)
      val effectiveFunction = new ConstL12Normalizer(executorFunction).setFactor_L1(0.001)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(executorFunction)
      trainer.setOrientation(new LBFGS().setMinHistory(10).setMaxHistory(30))
      trainer.setMonitor(monitor)
      trainer.setTimeout(72, TimeUnit.HOURS)
      trainer.setTerminateThreshold(Double.NegativeInfinity)
      trainer
    }.run()
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
    run((data, network) ⇒ ScheduledSampleTrainable.Pow(data.toArray, network, 50,1.0,0.0).setShuffled(true))
    //run((data,network)⇒new ArrayTrainable(data.toArray, network))
  }

  private def loadData(out: HtmlNotebookOutput with ScalaNotebookOutput) = {
    out.p("Loading data from " + source)
    val loader = new ImageTensorLoader(new File(source), 64, 64, 64, 64, 10, 10)
    val rawData: List[LabeledObject[Tensor]] = loader.stream().iterator().asScala.toStream.flatMap(tile ⇒ List(
      new LabeledObject[Tensor](tile, "original")
    ) ++ corruptors.map(e ⇒ {
      new LabeledObject[Tensor](e._2(tile), e._1)
    })).take(1000).toList
    loader.stop()
    val labels = List("original") ++ corruptors.keys.toList.sorted
    val categories: Map[String, Int] = labels.zipWithIndex.toMap
    out.p("<ol>" + categories.toList.sortBy(_._2).map(x ⇒ "<li>" + x + "</li>").mkString("\n") + "</ol>")
    val data: List[Array[Tensor]] = rawData.map((labeledObj: LabeledObject[Tensor]) ⇒ {
      Array(labeledObj.data, toOutNDArray(categories(labeledObj.label), categories.size))
    })
    out.eval {
      TableOutput.create(rawData.take(100).map(testObj ⇒ Map[String, AnyRef](
        "Image" → out.image(testObj.data.toRgbImage(), testObj.data.toString),
        "Label" → testObj.label
      ).asJava): _*)
    }
    out.p("Loading data complete")
    (categories, data)
  }

  def run(executor: (List[Array[Tensor]], SupervisedNetwork) ⇒ Trainable): Unit = {
    out.p("View the convergence history: <a href='/history.html'>/history.html</a>")
    out.p("<a href='/netmon.json'>Network Monitoring</a>")
    out.p("View the log: <a href='/log'>/log</a>")
    out.p("<a href='/table.html'>Parameter History Data Table</a>")
    server.addSyncHandler("table.html", "text/html", Java8Util.cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(log ⇒ {
        log.h1("Parameter History Data Table")
        log.p(dataTable.toHtmlTable(true))
      })
    }), false)
    out.out("<hr/>")
    val (categories: Map[String, Int], data: List[Array[Tensor]]) = loadData(out)
    var model: PipelineNetwork = corruptionDetectionModel(categories.size, monitoringRoot)

    out.p("<a href='test.html'>Test Reconstruction</a>")
    server.addSyncHandler("test.html", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(out ⇒ {
        try {
          out.eval {
            TableOutput.create(Random.shuffle(data).take(100).map(testObj ⇒ Map[String, AnyRef](
              "Image" → out.image(testObj(0).toRgbImage(), ""),
              "Categorization" → categories.toList.sortBy(_._2).map(_._1)
                .zip(model.eval(testObj(0)).data.head.getData.map(_*100.0))
            ).asJava): _*)
          }
        } catch {
          case e : Throwable ⇒ e.printStackTrace()
        }
      })
    }), false)

    train(data, model, executor)
    IOUtil.writeKryo(model, out.file("model_final.kryo"))
    summarizeHistory(out, history.toList)
    out.out("<hr/>")
    logOut.close()
    val onExit = new Semaphore(0)
    out.p("To exit the sever: <a href='/exit'>/exit</a>")
    server.addSyncHandler("exit", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(log ⇒ {
        log.h1("OK")
        onExit.release(1)
      })
    }), false)
    onExit.acquire()
  }

  val logOut = new TeeOutputStream(new FileOutputStream("training.log", true), true)
  val history = new scala.collection.mutable.ArrayBuffer[IterativeTrainer.Step]()
  server.addSyncHandler("history.html", "text/html", cvt(o ⇒ {
    Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(log ⇒ {
      summarizeHistory(log, history.toList)
    })
  }), false)
  val monitoringRoot = new MonitoredObject()
  monitoringRoot.addField("openCL",Java8Util.cvt(()⇒{
    val sb = new java.lang.StringBuilder()
    KernelManager.instance().reportDeviceUsage(sb,true)
    sb.toString().split("\n")
  }))
  server.addSyncHandler("netmon.json", "application/json", cvt(out ⇒ {
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