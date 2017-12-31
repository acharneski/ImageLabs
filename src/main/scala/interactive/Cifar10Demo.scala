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

import java.io.{ByteArrayOutputStream, PrintStream}
import java.lang
import java.util.concurrent.{Semaphore, TimeUnit}

import com.fasterxml.jackson.databind.ObjectMapper
import com.simiacryptus.mindseye.eval.SampledArrayTrainable
import com.simiacryptus.mindseye.lang.{NNExecutionContext, Tensor}
import com.simiacryptus.mindseye.layers.java._
import com.simiacryptus.mindseye.network.util.InceptionLayer
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt.{Step, TrainingMonitor}
import com.simiacryptus.mindseye.test.data.CIFAR10
import com.simiacryptus.util.io.{HtmlNotebookOutput, TeeOutputStream}
import com.simiacryptus.util.test.LabeledObject
import com.simiacryptus.util.{MonitoredObject, StreamNanoHTTPD, TableOutput, Util}
import fi.iki.elonen.NanoHTTPD
import fi.iki.elonen.NanoHTTPD.IHTTPSession
import smile.plot.{PlotCanvas, ScatterPlot}
import util._

import scala.collection.JavaConverters._
import scala.util.Random


object Cifar10Demo extends Report {

  def main(args: Array[String]): Unit = {
    report(new Cifar10Demo().run)
    System.exit(0)
  }
}
class Cifar10Demo {

  def run(server: StreamNanoHTTPD, log: HtmlNotebookOutput with ScalaNotebookOutput) {
    val inputSize = Array[Int](256, 256, 3)
    log.h1("CIFAR 10")
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
    val rawData: Stream[LabeledObject[Tensor]] = Random.shuffle(CIFAR10.trainingDataStream().iterator().asScala.toStream.take(100000))
    val trainingData: List[LabeledObject[Tensor]] = rawData.take(10000).toList
    val validationStream: List[LabeledObject[Tensor]] = rawData.drop(trainingData.size).take(1000).toList
    val data: List[Array[Tensor]] = trainingData.map((labeledObj: LabeledObject[Tensor]) ⇒ {
      Array(labeledObj.data, toOutNDArray(toOut(labeledObj.label), 10))
    })
    CIFAR10.halt()
    System.gc()

    log.eval {
      TableOutput.create(data.take(10).map(testObj ⇒ Map[String, AnyRef](
        "Input1 (as Image)" → log.image(testObj(0).toRgbImage(), testObj(0).toString),
        "Input2 (as String)" → testObj(1).toString,
        "Input1 (as String)" → testObj(0).toString
      ).asJava): _*)
    }

    log.eval {
      TableOutput.create(data.filter(_(1).get(0) == 1).take(10).map(testObj ⇒ Map[String, AnyRef](
        "Input1 (as Image)" → log.image(testObj(0).toRgbImage(), testObj(0).toString),
        "Input2 (as String)" → testObj(1).toString,
        "Input1 (as String)" → testObj(0).toString
      ).asJava): _*)
    }

    log.h2("Model")
    log.p("Here we define the logic network that we are about to newTrainer: ")
    var model: PipelineNetwork = log.eval {
      val outputSize = Array[Int](10)
      var model: PipelineNetwork = new PipelineNetwork

      //      model.fn(new MonitoringWrapperLayer(new ImgConvolutionSynapseLayer(5,5,4)
      //        .setByCoord(Java8Util.cvt(_⇒Util.R.get.nextGaussian * 0.01))).addTo(monitoringRoot,"conv1"))

      model.add(new MonitoringWrapperLayer(new InceptionLayer(Array(
        Array(Array(5,5,3)),
        Array(Array(3,3,9))
      ))).addTo(monitoringRoot,"inception1"))
      //  .setByCoord(Java8Util.cvt(_⇒Util.R.get.nextGaussian * 0.01))).addTo(monitoringRoot,"conv1"))


      model.add(new MaxPoolingLayer(2, 2, 1))
      model.add(new MonitoringWrapperLayer(new FullyConnectedLayer(Array[Int](16, 16, 4), outputSize)
        .set(Java8Util.cvt(() ⇒ Util.R.get.nextGaussian * 0.01))).addTo(monitoringRoot, "synapse1"))
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
      val trainable = new SampledArrayTrainable(data.toArray, trainingNetwork, 1000)
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
        val result = model.eval(new NNExecutionContext() {}, testObj.data).getData.get(0)
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
        val result = model.eval(new NNExecutionContext() {}, testObj.data).getData.get(0)
        val prediction: Int = (0 to 9).maxBy(i ⇒ result.get(i))
        val actual = toOut(testObj.label)
        prediction == actual
      }).take(10).map(testObj ⇒ {
        val result = model.eval(new NNExecutionContext() {}, testObj.data).getData.get(0)
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
        val result = model.eval(new NNExecutionContext() {}, testObj.data).getData.get(0)
        val prediction: Int = (0 to 9).maxBy(i ⇒ result.get(i))
        val actual: Int = toOut(testObj.label)
        actual → prediction
      }).groupBy(_._1).mapValues(_.groupBy(_._2).mapValues(_.size))
    }
    writeMislassificationMatrix(log, categorizationMatrix)
    log.out("")
    log.p("The accuracy, summarized per category: ")
    log.eval {
      (0 to 9).map(actual ⇒ {
        actual → (categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0) * 100.0 / categorizationMatrix.getOrElse(actual, Map.empty).values.sum)
      }).toMap
    }
    log.p("The accuracy, summarized over the entire validation setByCoord: ")
    log.eval {
      (0 to 9).map(actual ⇒ {
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

  private def writeMislassificationMatrix(log: HtmlNotebookOutput, categorizationMatrix: Map[Int, Map[Int, Int]]) = {
    log.out("<table>")
    log.out("<tr>")
    log.out((List("Actual \\ Predicted | ") ++ (0 to 9)).map("<td>"+_+"</td>").mkString(""))
    log.out("</tr>")
    (0 to 9).foreach(actual ⇒ {
      log.out("<tr>")
      log.out(s"<td>$actual</td>" + (0 to 9).map(prediction ⇒ {
        categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(prediction, 0)
      }).map("<td>"+_+"</td>").mkString(""))
      log.out("</tr>")
    })
    log.out("</table>")
  }

  private def summarizeHistory(log: ScalaNotebookOutput, history: List[com.simiacryptus.mindseye.opt.Step]) = {
    if(!history.isEmpty) {
      log.eval {
        val step = Math.max(Math.pow(10, Math.ceil(Math.log(history.size) / Math.log(10)) - 2), 1).toInt
        TableOutput.create(history.filter(0 == _.iteration % step).map(state ⇒
          Map[String, AnyRef](
            "iteration" → state.iteration.toInt.asInstanceOf[Integer],
            "time" → state.time.toDouble.asInstanceOf[lang.Double],
            "fitness" → state.point.sum.toDouble.asInstanceOf[lang.Double]
          ).asJava
        ): _*)
      }
      log.eval {
        val plot: PlotCanvas = ScatterPlot.plot(history.map(item ⇒ Array[Double](
          item.iteration, Math.log(item.point.sum)
        )).toArray: _*)
        plot.setTitle("Convergence Plot")
        plot.setAxisLabels("Iteration", "log(Fitness)")
        plot.setSize(600, 400)
        plot
      }
    }
  }

  def toOut(label: String): Int = {
    (0 until 10).find(label == "[" + _ + "]").get
  }

  def toOutNDArray(out: Int, max: Int): Tensor = {
    val ndArray = new Tensor(max)
    ndArray.set(out, 1)
    ndArray
  }

}