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

import java.awt.{Graphics2D, RenderingHints}
import java.awt.image.BufferedImage
import java.io.{ByteArrayOutputStream, PrintStream}
import java.lang
import java.util.concurrent.{Semaphore, TimeUnit}

import com.fasterxml.jackson.databind.ObjectMapper
import com.simiacryptus.mindseye.data.{Caltech101, Tensor}
import com.simiacryptus.mindseye.layers.NNLayer
import com.simiacryptus.mindseye.network.{InceptionLayer, PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.layers.activation.SoftmaxActivationLayer
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer
import com.simiacryptus.mindseye.layers.media.MaxSubsampleLayer
import com.simiacryptus.mindseye.layers.synapse.{BiasLayer, DenseSynapseLayer}
import com.simiacryptus.mindseye.layers.util.MonitoringWrapper
import com.simiacryptus.mindseye.opt.trainable.StochasticArrayTrainable
import com.simiacryptus.mindseye.opt.{IterativeTrainer, Step, TrainingMonitor}
import com.simiacryptus.util.{MonitoredObject, StreamNanoHTTPD, Util}
import com.simiacryptus.util.io.{HtmlNotebookOutput, TeeOutputStream}
import com.simiacryptus.util.lang.SupplierWeakCache
import com.simiacryptus.util.test.{Caltech101, LabeledObject}
import com.simiacryptus.util.text.TableOutput
import fi.iki.elonen.NanoHTTPD
import fi.iki.elonen.NanoHTTPD.IHTTPSession
import smile.plot.{PlotCanvas, ScatterPlot}
import util._

import scala.collection.JavaConverters._
import scala.util.Random


object Caltech101Demo extends Report {

  def main(args: Array[String]): Unit = {
    report(new Caltech101Demo().run)
    System.exit(0)
  }
}
class Caltech101Demo {

  def run(server: StreamNanoHTTPD, log: HtmlNotebookOutput with ScalaNotebookOutput) {
    val inputSize = Array[Int](256, 256, 3)
    log.h1("Caltech 101")
    val history = new scala.collection.mutable.ArrayBuffer[Step]()
    log.p("View the convergence history: <a href='/history.html'>/history.html</a>")
    server.addAsyncHandler("history.html", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        summarizeHistory(log, history.toList)
      })
    }), false)
    val monitoringRoot = new MonitoredObject()
    log.p("<a href='/netmon.json'>Network Monitoring</a>")
    server.addAsyncHandler("netmon.json", "application/json", Java8Util.cvt(out ⇒ {
      val mapper = new ObjectMapper().enableDefaultTyping(ObjectMapper.DefaultTyping.NON_FINAL)
      val buffer = new ByteArrayOutputStream()
      mapper.writeValue(buffer, monitoringRoot.getMetrics)
      out.write(buffer.toByteArray)
    }), false)
    log.p("View the log: <a href='/log'>/log</a>")
    val logOut = new TeeOutputStream(log.file("log.txt"), true)
    val logPrintStream = new PrintStream(logOut)
    server.addSessionHandler("log", Java8Util.cvt((session : IHTTPSession)⇒{
      NanoHTTPD.newChunkedResponse(NanoHTTPD.Response.Status.OK, "text/plain", logOut.newInputStream())
    }))
    val monitor = new TrainingMonitor {
      override def log(msg: String): Unit = {
        System.err.println(msg);
        logPrintStream.println(msg);
      }

      override def onStepComplete(currentPoint: Step): Unit = {
        history += currentPoint
      }
    }
    log.out("<hr/>")


    log.h2("Data")
    val rawData: Stream[LabeledObject[SupplierWeakCache[BufferedImage]]] = log.eval {
      Random.shuffle(Caltech101.trainingDataStream().iterator().asScala.toStream)
    }
    val categories = {
      val list = rawData.map(_.label).distinct
      list.zip(0 until list.size).toMap
    }
    log.p("<ol>"+categories.toList.sortBy(_._2).map(x⇒"<li>"+x+"</li>").mkString("\n")+"</ol>")

    def normalize(img : BufferedImage) = {
      val aspect = img.getWidth * 1.0 / img.getHeight
      val scale = 256.0 / Math.min(img.getWidth, img.getHeight)
      val normalizedImage = new BufferedImage(256, 256, BufferedImage.TYPE_INT_ARGB)
      val graphics = normalizedImage.getGraphics.asInstanceOf[Graphics2D]
      graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
      if(aspect > 1.0) {
        graphics.drawImage(img, 128-(aspect * 128).toInt, 0, (scale * img.getWidth).toInt, (scale * img.getHeight).toInt, null)
      } else {
        graphics.drawImage(img, 0, 128-(128 / aspect).toInt, (scale * img.getWidth).toInt, (scale * img.getHeight).toInt, null)
      }
      normalizedImage
    }
    val whitelist = Set("octopus","lobster")//,"dolphin"


    val tensors: List[LabeledObject[Tensor]] = rawData.filter(x⇒whitelist.contains(x.label)).map(_.map(Java8Util.cvt(x⇒Tensor.fromRGB(normalize(x.get()))))).toList
    val trainingData: List[LabeledObject[Tensor]] = tensors.take(100).toList
    val validationStream: List[LabeledObject[Tensor]] = tensors.reverse.take(100).toList
    val data: List[Array[Tensor]] = trainingData.map((labeledObj: LabeledObject[Tensor]) ⇒ {
      Array(labeledObj.data, toOutNDArray(categories(labeledObj.label), categories.size))
    })

    log.eval {
      TableOutput.create(data.take(10).map(testObj ⇒ Map[String, AnyRef](
        "Input1 (as Image)" → log.image(testObj(0).toRgbImage(), testObj(0).toString),
        "Input2 (as String)" → testObj(1).toString
      ).asJava): _*)
    }

    log.h2("Model")
    log.p("Here we define the logic network that we are about to newTrainer: ")
    var model: PipelineNetwork = log.eval {
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

    log.p("We encapsulate our model network within a supervisory network that applies a loss function: ")
    val trainingNetwork: SupervisedNetwork = log.eval {
      new SimpleLossNetwork(model, new EntropyLossLayer)
    }

    log.h2("Training")
    log.p("We newTrainer using a standard iterative L-BFGS strategy: ")
    val trainer = log.eval {
      val trainable = new StochasticArrayTrainable(data.toArray, trainingNetwork, 1000)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
      trainer.setMonitor(monitor)
      trainer.setTimeout(5, TimeUnit.MINUTES)
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    log.eval {
      trainer.run()
    }
    log.p("After training, we have the following parameterized model: ")
    log.eval {
      model.toString
    }
    log.p("A summary of the training timeline: ")
    summarizeHistory(log, history.toList)

    log.h2("Validation")
    log.p("Here we examine a sample of validation rows, randomly selected: ")
    log.eval {
      TableOutput.create(validationStream.take(10).map(testObj ⇒ {
        val result = model.eval(new NNLayer.NNExecutionContext() {}, testObj.data).getData.get(0)
        Map[String, AnyRef](
          "Input" → log.image(testObj.data.toRgbImage(), testObj.label),
          "Predicted Label" → (0 to 9).maxBy(i ⇒ result.get(i)).asInstanceOf[Integer],
          "Actual Label" → testObj.label,
          "Network Output" → result
        ).asJava
      }): _*)
    }
    log.p("Validation rows that are mispredicted are also sampled: ")
    log.eval {
      TableOutput.create(validationStream.filterNot(testObj ⇒ {
        val result = model.eval(new NNLayer.NNExecutionContext() {}, testObj.data).getData.get(0)
        val prediction: Int = (0 to 9).maxBy(i ⇒ result.get(i))
        val actual = categories(testObj.label)
        prediction == actual
      }).take(10).map(testObj ⇒ {
        val result = model.eval(new NNLayer.NNExecutionContext() {}, testObj.data).getData.get(0)
        Map[String, AnyRef](
          "Input" → log.image(testObj.data.toRgbImage(), testObj.label),
          "Predicted Label" → (0 to 9).maxBy(i ⇒ result.get(i)).asInstanceOf[Integer],
          "Actual Label" → testObj.label,
          "Network Output" → result
        ).asJava
      }): _*)
    }
    log.p("To summarize the accuracy of the model, we calculate several summaries: ")
    log.p("The (mis)categorization matrix displays a count matrix for every actual/predicted category: ")
    val categorizationMatrix: Map[Int, Map[Int, Int]] = log.eval {
      validationStream.map(testObj ⇒ {
        val result = model.eval(new NNLayer.NNExecutionContext() {}, testObj.data).getData.get(0)
        val prediction: Int = (0 until categories.size).maxBy(i ⇒ result.get(i))
        val actual: Int = categories(testObj.label)
        actual → prediction
      }).groupBy(_._1).mapValues(_.groupBy(_._2).mapValues(_.size))
    }
    writeMislassificationMatrix(log, categorizationMatrix, categories.size)
    log.out("")
    log.p("The accuracy, summarized per category: ")
    log.eval {
      (0 until categories.size).map(actual ⇒ {
        actual → (categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0) * 100.0 / categorizationMatrix.getOrElse(actual, Map.empty).values.sum)
      }).toMap
    }
    log.p("The accuracy, summarized over the entire validation set: ")
    log.eval {
      (0 until categories.size).map(actual ⇒ {
        categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0)
      }).sum.toDouble * 100.0 / categorizationMatrix.values.flatMap(_.values).sum
    }

    log.out("<hr/>")
    logOut.close()
    val onExit = new Semaphore(0)
    log.p("To exit the sever: <a href='/exit'>/exit</a>")
    server.addAsyncHandler("exit", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        log.h1("OK")
        onExit.release(1)
      })
    }), false)
    onExit.acquire()
  }

  private def summarizeHistory(log: ScalaNotebookOutput, history: List[com.simiacryptus.mindseye.opt.Step]) = {
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