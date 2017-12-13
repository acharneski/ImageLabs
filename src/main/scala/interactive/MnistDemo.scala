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

import java.util.concurrent.TimeUnit

import _root_.util.NetworkViz._
import _root_.util._
import com.simiacryptus.mindseye.data.MNIST
import com.simiacryptus.mindseye.eval.{L12Normalizer, SampledArrayTrainable}
import com.simiacryptus.mindseye.lang.{NNExecutionContext, NNLayer, Tensor}
import com.simiacryptus.mindseye.layers.java._
import com.simiacryptus.mindseye.network.{DAGNetwork, PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.orient.GradientDescent
import com.simiacryptus.util.{StreamNanoHTTPD, TableOutput}
import com.simiacryptus.util.io.{HtmlNotebookOutput, KryoUtil, MarkdownNotebookOutput}
import guru.nidi.graphviz.engine.{Format, Graphviz}

import scala.collection.JavaConverters._
import scala.util.Random


object MnistDemo extends Report {

  def main(args: Array[String]): Unit = {
    report((s,log)⇒new MnistDemo(s,log).run)
    System.exit(0)
  }
}

object MnistDemo_L1Normalizations extends Report {

  def main(args: Array[String]): Unit = {
    report((s,log)⇒new MnistDemo(s,log){
      override def buildTrainer(data: Seq[Array[Tensor]]): Stream[IterativeTrainer] = Stream(log.eval {
        val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
        val executor = new SampledArrayTrainable(data.toArray, trainingNetwork, 1000)
        val normalized = new L12Normalizer(executor) {
          override protected def getL1(layer: NNLayer): Double = layer match {
            case _ : FullyConnectedLayer ⇒ -0.001
            case _ ⇒ 0.0
          }
          override protected def getL2(layer: NNLayer): Double = 0.0
        }
        val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(normalized)
        trainer.setMonitor(monitor)
        trainer.setOrientation(new GradientDescent)
        trainer.setTimeout(trainingTime, TimeUnit.MINUTES)
        trainer.setTerminateThreshold(0.0)
        trainer
      })
    }.run)
    System.exit(0)
  }
}

class MnistDemo(server: StreamNanoHTTPD, log: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, log) {

  val inputSize = Array[Int](28, 28, 1)
  val outputSize = Array[Int](10)
  val trainingTime = 5

  model = log.eval {
    var model: PipelineNetwork = new PipelineNetwork
    model.add(new MonitoringWrapperLayer(new BiasLayer(inputSize: _*)).addTo(monitoringRoot, "inbias"))
    model.add(new MonitoringWrapperLayer(new FullyConnectedLayer(inputSize, outputSize)
      .setWeights(Java8Util.cvt(() ⇒ 0.001 * (Random.nextDouble() - 0.5)))).addTo(monitoringRoot, "synapse"))
    model.add(new MonitoringWrapperLayer(new ReLuActivationLayer).addTo(monitoringRoot, "relu"))
    model.add(new MonitoringWrapperLayer(new BiasLayer(outputSize: _*)).addTo(monitoringRoot, "outbias"))
    model.add(new SoftmaxActivationLayer)
    model
  }

  def buildTrainer(data: Seq[Array[Tensor]]): Stream[IterativeTrainer] = Stream(log.eval {
    val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
    val trainable = new SampledArrayTrainable(data.toArray, trainingNetwork, 1000)
    val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
    trainer.setMonitor(monitor)
    trainer.setOrientation(new GradientDescent);
    trainer.setTimeout(trainingTime, TimeUnit.MINUTES)
    trainer.setTerminateThreshold(0.0)
    trainer
  })

  def run {

    log.p("In this demo we newTrainer a simple neural network against the MNIST handwritten digit dataset")

    log.h2("Data")
    log.p("First, we cache the training dataset: ")
    val data: Seq[Array[Tensor]] = log.eval {
      MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
        Array(labeledObj.data, toOutNDArray(toOut(labeledObj.label), 10))
      })
    }
    log.p("<a href='/sample.html'>View a preview table here</a>")
    server.addSyncHandler("sample.html", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        log.eval {
          TableOutput.create(data.take(10).map(testObj ⇒ Map[String, AnyRef](
            "Input1 (as Image)" → log.image(testObj(0).toGrayImage(), testObj(0).toString),
            "Input2 (as String)" → testObj(1).toString,
            "Input1 (as String)" → testObj(0).toString
          ).asJava): _*)
        }
      })
    }), false)

    log.h2("Model")
    log.p("Here we define the logic network that we are about to train: ")
    defineHeader()

    log.p("<a href='/test.html'>Validation Report</a>")
    server.addSyncHandler("test.html", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        validation(log, KryoUtil.kryo().copy(model))
      })
    }), false)

    log.p("We train using a the following strategy: ")
    buildTrainer(data).foreach(trainer ⇒ log.eval {
      trainer.run()
    })

    log.p("A summary of the training timeline: ")
    summarizeHistory(log)

    log.p("Validation Report")
    validation(log, model)

    log.p("Parameter History Data Table")
    log.p(dataTable.toHtmlTable)

    waitForExit()
  }

  def validation(log: HtmlNotebookOutput with ScalaNotebookOutput, model: NNLayer) = {
    log.h2("Validation")
    log.p("Here we examine a sample of validation rows, randomly selected: ")
    log.eval {
      TableOutput.create(MNIST.validationDataStream().iterator().asScala.toStream.take(10).map(testObj ⇒ {
        val result = model.eval(new NNExecutionContext() {}, testObj.data).getData.get(0)
        Map[String, AnyRef](
          "Input" → log.image(testObj.data.toGrayImage(), testObj.label),
          "Predicted Label" → (0 to 9).maxBy(i ⇒ result.get(i)).asInstanceOf[Integer],
          "Actual Label" → testObj.label,
          "Network Output" → result
        ).asJava
      }): _*)
    }
    log.p("Validation rows that are mispredicted are also sampled: ")
    log.eval {
      TableOutput.create(MNIST.validationDataStream().iterator().asScala.toStream.filterNot(testObj ⇒ {
        val result = model.eval(new NNExecutionContext() {}, testObj.data).getData.get(0)
        val prediction: Int = (0 to 9).maxBy(i ⇒ result.get(i))
        val actual = toOut(testObj.label)
        prediction == actual
      }).take(10).map(testObj ⇒ {
        val result = model.eval(new NNExecutionContext() {}, testObj.data).getData.get(0)
        Map[String, AnyRef](
          "Input" → log.image(testObj.data.toGrayImage(), testObj.label),
          "Predicted Label" → (0 to 9).maxBy(i ⇒ result.get(i)).asInstanceOf[Integer],
          "Actual Label" → testObj.label,
          "Network Output" → result
        ).asJava
      }): _*)
    }
    log.p("To summarize the accuracy of the model, we calculate several summaries: ")
    log.p("The (mis)categorization matrix displays a count matrix for every actual/predicted category: ")
    val categorizationMatrix: Map[Int, Map[Int, Int]] = log.eval {
      MNIST.validationDataStream().iterator().asScala.toStream.map(testObj ⇒ {
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
    log.p("The accuracy, summarized over the entire validation set: ")
    log.eval {
      (0 to 9).map(actual ⇒ {
        categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0)
      }).sum.toDouble * 100.0 / categorizationMatrix.values.flatMap(_.values).sum
    }
  }

  def writeMislassificationMatrix(log: HtmlNotebookOutput, categorizationMatrix: Map[Int, Map[Int, Int]]) = {
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

  def writeMislassificationMatrix(log: MarkdownNotebookOutput, categorizationMatrix: Map[Int, Map[Int, Int]]) = {
    log.out("Actual \\ Predicted | " + (0 to 9).mkString(" | "))
    log.out((0 to 10).map(_ ⇒ "---").mkString(" | "))
    (0 to 9).foreach(actual ⇒ {
      log.out(s" **$actual** | " + (0 to 9).map(prediction ⇒ {
        categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(prediction, 0)
      }).mkString(" | "))
    })
  }

  def toOut(label: String): Int = {
    (0 until 10).find(label == "[" + _ + "]").get
  }

  def networkGraph(log: ScalaNotebookOutput, network: DAGNetwork, width: Int = 1200, height: Int = 1000) = {
    try {
      log.eval {
        Graphviz.fromGraph(toGraph(network)).height(height).width(width).render(Format.PNG).toImage
      }
    } catch {
      case e : Throwable ⇒ e.printStackTrace()
    }
  }

  def toOutNDArray(out: Int, max: Int): Tensor = {
    val ndArray = new Tensor(max)
    ndArray.set(out, 1)
    ndArray
  }

}