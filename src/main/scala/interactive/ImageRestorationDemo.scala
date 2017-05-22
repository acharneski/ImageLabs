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
import java.io.{ByteArrayOutputStream, File, PrintStream}
import java.lang
import java.util.concurrent.{Semaphore, TimeUnit}

import _root_.util._
import com.fasterxml.jackson.databind.ObjectMapper
import com.simiacryptus.mindseye.graph.{InceptionLayer, PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer
import com.simiacryptus.mindseye.net.media.MaxSubsampleLayer
import com.simiacryptus.mindseye.net.synapse.{BiasLayer, DenseSynapseLayer}
import com.simiacryptus.mindseye.net.util.{MonitoredObject, MonitoringWrapper}
import com.simiacryptus.mindseye.opt.{IterativeTrainer, TrainingMonitor}
import com.simiacryptus.util.io.{HtmlNotebookOutput, IOUtil, TeeOutputStream}
import com.simiacryptus.util.ml.Tensor
import com.simiacryptus.util.test.ImageTiles.ImageTensorLoader
import com.simiacryptus.util.test.LabeledObject
import com.simiacryptus.util.text.TableOutput
import com.simiacryptus.util.{StreamNanoHTTPD, Util}
import fi.iki.elonen.NanoHTTPD
import fi.iki.elonen.NanoHTTPD.IHTTPSession
import smile.plot.{PlotCanvas, ScatterPlot}

import scala.collection.JavaConverters._
import scala.util.Random


object ImageRestorationDemo extends ServiceNotebook {

  def main(args: Array[String]): Unit = {
//    var builder = SparkSession.builder
//      .appName("Spark MindsEye Demo")
//    builder = args match {
//      case Array(masterUrl) ⇒ builder.master(masterUrl)
//      case _ ⇒ builder
//    }
//    val sparkSession = builder
//      .getOrCreate()
//    sparkSession.sparkContext
//    sparkSession.stop()

    args match {
      case Array(source) ⇒ new ImageRestorationDemo(source)
      case _ ⇒ new ImageRestorationDemo("E:\\testImages\\256_ObjectCategories")
    }

  }
}

class ImageRestorationDemo(source : String) {

  def run(server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) {
    val history = new scala.collection.mutable.ArrayBuffer[IterativeTrainer.Step]()
    out.p("View the convergence history: <a href='/history.html'>/history.html</a>")
    server.addHandler("history.html", "text/html", Java8Util.cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(log ⇒ {
        summarizeHistory(log, history.toList)
      })
    }), false)
    val monitoringRoot = new MonitoredObject()
    out.p("<a href='/netmon.json'>Network Monitoring</a>")
    server.addHandler("netmon.json", "application/json", Java8Util.cvt(out ⇒ {
      val mapper = new ObjectMapper().enableDefaultTyping(ObjectMapper.DefaultTyping.NON_FINAL)
      val buffer = new ByteArrayOutputStream()
      mapper.writeValue(buffer, monitoringRoot.getMetrics)
      out.write(buffer.toByteArray)
    }), false)
    out.p("View the log: <a href='/log'>/log</a>")
    val logOut = new TeeOutputStream(out.file("log.txt"), true)
    val logPrintStream = new PrintStream(logOut)
    server.addHandler2("log", Java8Util.cvt((session : IHTTPSession)⇒{
      NanoHTTPD.newChunkedResponse(NanoHTTPD.Response.Status.OK, "text/plain", logOut.newInputStream())
    }))
    out.out("<hr/>")


    out.h2("Data")

    def corrupt(imgTensor : Tensor) : Tensor = {
      def resize(source: BufferedImage, size:Int) = {
        val image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB)
        val graphics = image.getGraphics.asInstanceOf[Graphics2D]
        graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
        graphics.drawImage(source, 0, 0, size, size, null)
        image
      }
      Tensor.fromRGB(resize(resize(imgTensor.toRgbImage, 64), 256))
    }

    val loader = new ImageTensorLoader(new File(source), 256, 256, 126, 126, 10, 10)
    val rawData: List[LabeledObject[Tensor]] = loader
      .stream().iterator().asScala.toStream.flatMap(tile⇒List(
      new LabeledObject[Tensor](tile,"original"),
      new LabeledObject[Tensor](corrupt(tile),"corrupt")
    )).take(1000).toList;
    loader.stop();

    val categories: Map[String, Int] = Map("original"→0, "corrupt"→1)
    out.p("<ol>"+categories.toList.sortBy(_._2).map(x⇒"<li>"+x+"</li>").mkString("\n")+"</ol>")

    val tensors: List[LabeledObject[Tensor]] = Random.shuffle(rawData)
    val trainingData: List[LabeledObject[Tensor]] = tensors.take(100).toList
    val validationStream: List[LabeledObject[Tensor]] = tensors.reverse.take(100).toList
    val data: List[Array[Tensor]] = trainingData.map((labeledObj: LabeledObject[Tensor]) ⇒ {
      Array(labeledObj.data, toOutNDArray(categories(labeledObj.label), categories.size))
    })

    out.eval {
      TableOutput.create(data.take(10).map(testObj ⇒ Map[String, AnyRef](
        "Input1 (as Image)" → out.image(testObj(0).toRgbImage(), testObj(0).toString),
        "Input2 (as String)" → testObj(1).toString
      ).asJava): _*)
    }

    out.h2("Model")
    out.p("Here we define the logic network that we are about to newTrainer: ")
    var model: PipelineNetwork = out.eval {
      val outputSize = Array[Int](categories.size)
      var model: PipelineNetwork = new PipelineNetwork
      model.add(new MonitoringWrapper(new InceptionLayer(Array(
        Array(Array(5,5,3)),
        Array(Array(3,3,9))
      ))).addTo(monitoringRoot,"inception1"))
      model.add(new MaxSubsampleLayer(2,2,1))
      model.add(new MonitoringWrapper(new InceptionLayer(Array(
        Array(Array(5,5,4)),
        Array(Array(3,3,16))
      ))).addTo(monitoringRoot,"inception2"))
      model.add(new MaxSubsampleLayer(2,2,1))
      model.add(new MonitoringWrapper(new DenseSynapseLayer(Array[Int](64, 64, 5), outputSize)
        .setWeights(Java8Util.cvt(()⇒Util.R.get.nextGaussian * 0.01))).addTo(monitoringRoot,"synapse1"))
      model.add(new BiasLayer(outputSize: _*))
      model.add(new SoftmaxActivationLayer)
      model
    }
    out.h2("Training")
    val monitor = new TrainingMonitor {
      override def log(msg: String): Unit = {
        System.err.println(msg);
        logPrintStream.println(msg);
      }

      override def onStepComplete(currentPoint: IterativeTrainer.Step): Unit = {
        history += currentPoint
        IOUtil.writeKryo(model, out.file("model_checkpoint_" + (currentPoint.iteration % 5)+ ".kryo"))
      }
    }
    val trainer = out.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
      //val trainable = new com.simiacryptus.mindseye.opt.SparkTrainable(sparkContext.makeRDD(data, 8), trainingNetwork)
      val trainable = new com.simiacryptus.mindseye.opt.StochasticArrayTrainable(data.toArray, trainingNetwork, 10)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
      trainer.setMonitor(monitor)
      trainer.setTimeout(30, TimeUnit.MINUTES)
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    out.eval {
      trainer.run()
    }
    IOUtil.writeKryo(model, out.file("model_final.kryo"))
    summarizeHistory(out, history.toList)


    out.out("<hr/>")
    logOut.close()
    val onExit = new Semaphore(0)
    out.p("To exit the sever: <a href='/exit'>/exit</a>")
    server.addHandler("exit", "text/html", Java8Util.cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(log ⇒ {
        log.h1("OK")
        onExit.release(1)
      })
    }), false)
    onExit.acquire()
  }

  private def summarizeHistory(log: ScalaNotebookOutput, history: List[com.simiacryptus.mindseye.opt.IterativeTrainer.Step]) = {
    if(!history.isEmpty) {
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

  def toOut(label: String, max:Int): Int = {
    (0 until max).find(label == "[" + _ + "]").get
  }

  def toOutNDArray(out: Int, max: Int): Tensor = {
    val ndArray = new Tensor(max)
    ndArray.set(out, 1)
    ndArray
  }

  private def writeMislassificationMatrix(log: HtmlNotebookOutput, categorizationMatrix: Map[Int, Map[Int, Int]], max : Int) = {
    log.out("<table>")
    log.out("<tr>")
    log.out((List("Actual \\ Predicted | ") ++ (0 until max).map("<td>"+_+"</td>")).mkString(""))
    log.out("</tr>")
    (0 until max).foreach(actual ⇒ {
      log.out("<tr>")
      log.out(s"<td>$actual</td>" + (0 until max).map(prediction ⇒ {
        categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(prediction, 0)
      }).map("<td>"+_+"</td>").mkString(""))
      log.out("</tr>")
    })
    log.out("</table>")
  }

}