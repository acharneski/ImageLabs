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
import com.simiacryptus.mindseye.eval.SampledArrayTrainable
import com.simiacryptus.mindseye.lang.{NNExecutionContext, NNLayer, Tensor}
import com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer
import com.simiacryptus.mindseye.layers.java._
import com.simiacryptus.mindseye.network.{DAGNetwork, PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy
import com.simiacryptus.mindseye.opt.region.TrustRegion
import com.simiacryptus.mindseye.test.data.MNIST
import com.simiacryptus.util.io.{HtmlNotebookOutput, KryoUtil, MarkdownNotebookOutput}
import com.simiacryptus.util.{StreamNanoHTTPD, TableOutput}
import guru.nidi.graphviz.engine.{Format, Graphviz}

import scala.collection.JavaConverters._
import scala.concurrent.Await
import scala.concurrent.duration.Duration


object MnistAutoinitDemo extends Report {

  def main(args: Array[String]): Unit = {
    report((s,log)⇒new MnistAutoinitDemo(s,log).run)
    System.exit(0)
  }
}

object MnistAutoinitDemoConv extends Report {

  def main(args: Array[String]): Unit = {
    report((s,log)⇒new MnistAutoinitDemo(s,log){

      override lazy val component1 = log.eval {
        var model: PipelineNetwork = new PipelineNetwork
        model.add(new MonitoringWrapperLayer(new ConvolutionLayer(3,3,5).setWeights(Java8Util.cvt(()⇒0.01*Math.random()))).addTo(monitoringRoot, "synapse1"))
        model.add(new MonitoringWrapperLayer(new MaxSubsampleLayer(4,4,1)).addTo(monitoringRoot, "max1"))
        model.add(new MonitoringWrapperLayer(new ReLuActivationLayer).addTo(monitoringRoot, "relu1"))
        model.add(new MonitoringWrapperLayer(new BiasLayer(7,7,5)).addTo(monitoringRoot, "bias1"))
        model
      }

    }.run)
    System.exit(0)
  }
}

class MnistAutoinitDemo(server: StreamNanoHTTPD, log: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, log) {

  override def shouldReplotMetrics(iteration: Long) = iteration match {
    case _ if 100 > iteration ⇒ false
    case _ if 0 == iteration % 100 ⇒ true
    case _ ⇒ false
  }

  val inputSize = Array[Int](28, 28, 1)
  val midSize = Array[Int](100)
  val outputSize = Array[Int](10)
  val trainingTime = 5

  lazy val component1 = log.eval {
    var model: PipelineNetwork = new PipelineNetwork
    //model.fn(new MonitoringWrapperLayer(new BiasLayer(inputSize: _*).setName("bias1a")).addTo(monitoringRoot))
    model.add(new MonitoringWrapperLayer(new FullyConnectedLayer(inputSize, midSize)).addTo(monitoringRoot, "synapse1"))
    model.add(new MonitoringWrapperLayer(new ReLuActivationLayer).addTo(monitoringRoot, "relu1"))
    model.add(new MonitoringWrapperLayer(new BiasLayer(midSize: _*)).addTo(monitoringRoot, "bias1b"))
    model
  }

  def normalizePosition(component : NNLayer) = log.eval {
    var model: PipelineNetwork = new PipelineNetwork
    val componentNode = model.add(component)
    val means = model.add(new AvgMetaLayer(), componentNode)
    model.add(new BiasMetaLayer(), componentNode, model.add(new LinearActivationLayer().setScale(-1).freeze(), means))
    model
  }

  def normalizeScale(component : NNLayer) = log.eval {
    var model: PipelineNetwork = new PipelineNetwork
    val componentNode = model.add(component)
    model.add(new SqActivationLayer(), componentNode)
    val variances = model.add(new AvgMetaLayer(), componentNode)
    model.add(new ScaleMetaLayer(), componentNode, model.add(new NthPowerActivationLayer().setPower(-0.5), variances))
    model
  }

  def autoinitializer(component : NNLayer) = log.eval {
    var model: PipelineNetwork = new PipelineNetwork
    model.add(new MonitoringWrapperLayer(component).addTo(monitoringRoot, "component1init"))
    val componentNode = model.add(new MonitoringSynapse().addTo(monitoringRoot, "component1out"))

//    model.fn(new AbsActivationLayer(), componentNode)
//    model.fn(new MaxMetaLayer())
//    model.fn(new LinearActivationLayer().setBias(-1).freeze())
//    model.fn(new SqActivationLayer())
//    val maxLimiter = model.fn(new MonitoringSynapse().addTo(monitoringRoot, "valueMean"))

    model.add(new AvgMetaLayer(), componentNode)
    val means = model.add(new MonitoringSynapse().addTo(monitoringRoot, "valueMean"))
    model.add(new BiasMetaLayer(), componentNode, model.add(new LinearActivationLayer().setScale(-1).freeze(), means))
    val recentered = model.add(new MonitoringSynapse().addTo(monitoringRoot, "recentered"))

    model.add(new SqActivationLayer(), recentered)
    model.add(new AvgMetaLayer())
    //model.fn(new LinearActivationLayer().setBias(1e-15).freeze())
    val variances = model.add(new MonitoringSynapse().addTo(monitoringRoot, "valueVariance"))

    model.add(new ScaleMetaLayer(), recentered, model.add(new NthPowerActivationLayer().setPower(-0.5), variances))
    val rescaled = model.add(new MonitoringSynapse().addTo(monitoringRoot, "rescaled"))

    val logVariance = model.add(new HyperbolicActivationLayer().freeze(), model.add(new LogActivationLayer(), variances))

    model.add(new NthPowerActivationLayer().setPower(0.5), variances)
    model.add(new LinearActivationLayer().setBias(-1).freeze())
    val varOffset = model.add(new SqActivationLayer())

//    model.fn(new CrossProductLayer(), rescaled)
//    model.fn(new SumMetaLayer())
//    model.fn(new HyperbolicActivationLayer().setScale(1.0).freeze())
//    model.fn(new MonitoringSynapse().addTo(monitoringRoot, "dotNormalizer"))
//    model.fn(new SumReducerLayer())
//    val dotNormalizer = model.fn(new LinearActivationLayer().setScale(1.0).freeze())

    model.add(new SumReducerLayer(), model.add(new SumInputsLayer(),
          varOffset,
          logVariance
          //  model.fn(new HyperbolicActivationLayer(), means)
          ))

    model
  }

  def pretrain(component:NNLayer, data:Array[Array[Tensor]]) = {
    log.eval {
      val autoinitializerNetwork = autoinitializer(component)
      val trainable = new SampledArrayTrainable(data, autoinitializerNetwork, 2000)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
      trainer.setMonitor(monitor)
      trainer.setTimeout(5, TimeUnit.MINUTES)
      trainer.setTerminateThreshold(Double.NegativeInfinity)
      trainer.setOrientation(new TrustRegionStrategy() {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _ ⇒ null//new LinearSumConstraint
        }
      })
      trainer.run()
    }
  }

  model = log.eval {
    var model: PipelineNetwork = new PipelineNetwork
    model.add(new MonitoringWrapperLayer(component1).addTo(monitoringRoot, "component1"))
    model.add(new MonitoringSynapse().addTo(monitoringRoot, "pre-softmax"))
    model.add(new LinearActivationLayer().setScale(0.01))
    model.add(new MonitoringWrapperLayer(new FullyConnectedLayer(midSize, outputSize)).addTo(monitoringRoot, "synapseEnd"))
    model.add(new MonitoringWrapperLayer(new BiasLayer(outputSize: _*)).addTo(monitoringRoot, "outbias"))
    model.add(new SoftmaxActivationLayer)
    model
  }

  def buildTrainer(data: Seq[Array[Tensor]], model: NNLayer): IterativeTrainer = {
    val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
    val trainable = new SampledArrayTrainable(data.toArray, trainingNetwork, 1000)
    val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
    trainer.setMonitor(monitor)
    //trainer.setOrientations(new GradientDescent);
    trainer.setTimeout(5, TimeUnit.MINUTES)
    trainer.setTerminateThreshold(0.0)
    trainer
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
    log.p("Here we define the logic network that we are about to trainCjGD: ")
    defineHeader()

    log.p("<a href='/test.html'>Validation Report</a>")
    server.addSyncHandler("test.html", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        validation(log, KryoUtil.kryo().copy(model))
      })
    }), false)


    pretrain(component1, data.map(x ⇒ Array(x(0))).toArray)
    Await.result(generateMetricsHistoryReport(log), Duration(1, TimeUnit.MINUTES))
    Await.result(regenerateReports(), Duration(1, TimeUnit.MINUTES))

    log.p("Pretraining complete")
    Thread.sleep(10000)
    component1.freeze()
    monitoringRoot.clearConstants()

    log.p("We trainCjGD using a the following strategy: ")
    buildTrainer(data, model).run()

    log.p("A summary of the training timeline: ")
    summarizeHistory(log)
    Await.result(regenerateReports(), Duration(1, TimeUnit.MINUTES))

    log.p("Validation Report")
    validation(log, model)

    //log.p("Parameter History Data Table")
    //log.p(dataTable.toHtmlTable)

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
    log.p("The accuracy, summarized over the entire validation setByCoord: ")
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