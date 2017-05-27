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
import com.simiacryptus.mindseye.layers.NNLayer
import com.simiacryptus.mindseye.layers.activation._
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer
import com.simiacryptus.mindseye.layers.media.{ImgConvolutionSynapseLayer, MaxSubsampleLayer}
import com.simiacryptus.mindseye.layers.synapse.{BiasLayer, DenseSynapseLayer}
import com.simiacryptus.mindseye.layers.util.{MonitoringSynapse, MonitoringWrapper}
import com.simiacryptus.mindseye.network.graph._
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt.region.{LayerTrustRegion, LinearSumConstraint, TrustRegion}
import com.simiacryptus.mindseye.opt.trainable.ScheduledSampleTrainable
import com.simiacryptus.util.StreamNanoHTTPD
import com.simiacryptus.util.io.{HtmlNotebookOutput, KryoUtil, MarkdownNotebookOutput}
import com.simiacryptus.util.ml.Tensor
import com.simiacryptus.util.test.MNIST
import com.simiacryptus.util.text.TableOutput
import guru.nidi.graphviz.engine.{Format, Graphviz}

import scala.collection.JavaConverters._
import scala.util.Random


object ConvolutionalMnistDemo extends ServiceNotebook {

  def main(args: Array[String]): Unit = {
    report((s,l)⇒new ConvolutionalMnistDemo(s,l).run)
    System.exit(0)
  }
}

class ConvolutionalMnistDemo(server: StreamNanoHTTPD, log: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, log) {

  val inputSize = Array[Int](28, 28, 1)
  val outputSize = Array[Int](10)

  lazy val model = log.eval {
    var model: PipelineNetwork = new PipelineNetwork
    model.add(new MonitoringWrapper(new BiasLayer(inputSize: _*)).addTo(monitoringRoot, "inbias"))
    model.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(3, 3, 8)
      .setWeights(Java8Util.cvt(() ⇒ 0.1 * (Random.nextDouble() - 0.5)))).addTo(monitoringRoot, "synapse1"))
    model.add(new MonitoringWrapper(new MaxSubsampleLayer(4, 4, 1)).addTo(monitoringRoot, "max1"))
    //model.add(new MonitoringWrapper(new ImgBandBiasLayer(28,28,8)).addTo(monitoringRoot, "imgbias1"))
    model.add(new MonitoringWrapper(new ReLuActivationLayer).addTo(monitoringRoot, "relu1"))
    model.add(new MonitoringSynapse().addTo(monitoringRoot, "hidden1"))

    //      model.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(3,3,32)
    //        .setWeights(Java8Util.cvt(() ⇒ 0.01 * Random.nextGaussian()))).addTo(monitoringRoot, "synapse2"))
    //      model.add(new MonitoringWrapper(new ReLuActivationLayer).addTo(monitoringRoot, "relu2"))
    //      model.add(new MonitoringWrapper(new MaxSubsampleLayer(2,2,1)).addTo(monitoringRoot, "max2"))
    //model.add(new MonitoringWrapper(new ImgBandBiasLayer(7,7,8)).addTo(monitoringRoot, "imgbias2"))
    model.add(new MonitoringWrapper(new DenseSynapseLayer(Array[Int](7, 7, 8), outputSize)
      .setWeights(Java8Util.cvt(() ⇒ 0.001 * (Random.nextDouble() - 0.5)))).addTo(monitoringRoot, "synapse3"))

    model.add(new MonitoringWrapper(new ReLuActivationLayer).addTo(monitoringRoot, "relu3"))
    model.add(new MonitoringSynapse().addTo(monitoringRoot, "output"))
    model.add(new MonitoringWrapper(new BiasLayer(outputSize: _*)).addTo(monitoringRoot, "outbias"))
    model.add(new SoftmaxActivationLayer)
    model
  }

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
    model
    defineMonitorReports()

    log.p("<a href='/test.html'>Validation Report</a>")
    server.addSyncHandler("test.html", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        validation(log, KryoUtil.kryo().copy(model))
      })
    }), false)

    log.p("We train using a the following strategy: ")
    val trainer = log.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
      val trainable = ScheduledSampleTrainable.Pow(data.toArray, trainingNetwork, 100, 10,0).setShuffled(true)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
      trainer.setMonitor(monitor)
      trainer.setOrientation(new LayerTrustRegion() {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _:MonitoringWrapper ⇒ getRegionPolicy(layer.asInstanceOf[MonitoringWrapper].inner)
          //case _:DenseSynapseLayer ⇒ new GrowthSphere().setGrowthFactor(2.0).setMinRadius(1.0)
          //case _:DenseSynapseLayer ⇒ new SingleOrthant()
//          case _:DenseSynapseLayer ⇒ new GrowthSphere().setGrowthFactor(0.0).setMinRadius(0.01)
//          case _:ImgConvolutionSynapseLayer ⇒ new GrowthSphere().setGrowthFactor(0.0).setMinRadius(0.01)
          case _:DenseSynapseLayer ⇒ new LinearSumConstraint()
          case _:ImgConvolutionSynapseLayer ⇒ new LinearSumConstraint()
          case _ ⇒ null
        }
      });
      trainer.setTimeout(60, TimeUnit.MINUTES)
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    log.eval {
      trainer.run()
    }

    log.p("A summary of the training timeline: ")
    summarizeHistory(log)

    log.p("Parameter History Data Table")
    log.p(dataTable.toHtmlTable)

    log.p("Validation Report")
    validation(log, model)

    waitForExit()
  }

  private def validation(log: HtmlNotebookOutput with ScalaNotebookOutput, model: NNLayer) = {
    log.h2("Validation")
    log.p("Here we examine a sample of validation rows, randomly selected: ")
    log.eval {
      TableOutput.create(MNIST.validationDataStream().iterator().asScala.toStream.take(10).map(testObj ⇒ {
        val result = model.eval(testObj.data).data.head
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
        val result = model.eval(testObj.data).data.head
        val prediction: Int = (0 to 9).maxBy(i ⇒ result.get(i))
        val actual = toOut(testObj.label)
        prediction == actual
      }).take(10).map(testObj ⇒ {
        val result = model.eval(testObj.data).data.head
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
        val result = model.eval(testObj.data).data.head
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

  private def writeMislassificationMatrix(log: MarkdownNotebookOutput, categorizationMatrix: Map[Int, Map[Int, Int]]) = {
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