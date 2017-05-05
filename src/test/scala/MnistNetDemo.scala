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

import java.util.UUID
import java.util.concurrent.TimeUnit
import java.util.function.ToDoubleFunction
import java.{lang, util}

import com.simiacryptus.mindseye.net.activation.{AbsActivationLayer, SoftmaxActivationLayer}
import com.simiacryptus.mindseye.net.basic.BiasLayer
import com.simiacryptus.mindseye.net.dag._
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerJBLAS
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer
import com.simiacryptus.mindseye.net.media.{ConvolutionSynapseLayer, MaxSubsampleLayer}
import com.simiacryptus.mindseye.net.util.VerboseWrapper
import com.simiacryptus.mindseye.training.{IterativeTrainer, LbfgsTrainer, TrainingContext}
import com.simiacryptus.util.Util
import com.simiacryptus.util.ml.{Coordinate, Tensor}
import com.simiacryptus.util.test.MNIST
import com.simiacryptus.util.text.TableOutput
import guru.nidi.graphviz.attribute.RankDir
import guru.nidi.graphviz.engine.{Format, Graphviz}
import guru.nidi.graphviz.model._
import org.scalatest.{MustMatchers, WordSpec}
import smile.plot.{PlotCanvas, ScatterPlot}

import scala.collection.JavaConverters._

class MnistNetDemo extends WordSpec with MustMatchers with MarkdownReporter {

    val inputSize = Array[Int](28, 28, 1)
    val outputSize = Array[Int](10)
    var trainingTimeMinutes = 120

  "Train Digit Recognizer Network" should {

    "Flat Logistic Regression" in {
      report("simple", log ⇒ {
        test(log, log.eval {
          trainingTimeMinutes = 5
          var model: PipelineNetwork = new PipelineNetwork
          model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.0
          }))
          model.add(new BiasLayer(outputSize: _*))
          model.add(new SoftmaxActivationLayer)
          model
        })
      })
    }

    "Flat 2-Layer Abs" in {
      report("twolayerabs", log ⇒ {
        test(log, log.eval {
          trainingTimeMinutes = 120
          val middleSize = Array[Int](28, 28, 1)
          var model: PipelineNetwork = new PipelineNetwork
          model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), middleSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new BiasLayer(middleSize: _*))
          model.add(new AbsActivationLayer)
          model.add(new DenseSynapseLayerJBLAS(Tensor.dim(middleSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new BiasLayer(outputSize: _*))
          model.add(new SoftmaxActivationLayer)
          model
        })
      })
    }

    "Simple convolution-maxpool" in {
      report("simpleconv", log ⇒ {
        test(log, log.eval {
          trainingTimeMinutes = 120
          val middleSize = Array[Int](28, 28, 1)
          var model: PipelineNetwork = new PipelineNetwork
          model.add(new ConvolutionSynapseLayer(Array(2,2), 2).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new AbsActivationLayer)
          model.add(new MaxSubsampleLayer(2,2,1))
          model.add(new ConvolutionSynapseLayer(Array(2,2), 2).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new AbsActivationLayer)
          model.add(new MaxSubsampleLayer(2,2,1))

          def headDims = model.eval(new Tensor(inputSize:_*)).data(0).getDims
          model.add(new DenseSynapseLayerJBLAS(Tensor.dim(headDims: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new BiasLayer(headDims: _*))
          model.add(new SoftmaxActivationLayer)
          model
        })
      })
    }

  }

  def test(log: ScalaMarkdownPrintStream, model: PipelineNetwork) = {
    log.h2("Data")
    log.p("First, we load the training dataset: ")
    val data: Seq[Array[Tensor]] = log.code(() ⇒ {
      MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
        Array(labeledObj.data, toOutNDArray(toOut(labeledObj.label), 10))
      }).toList
    })

    log.p("We can visualize this network as a graph: ")
    networkGraph(log, model, 800)
    log.p("We encapsulate our model network within a supervisory network that applies a loss function: ")
    val trainingNetwork: SupervisedNetwork = log.eval {
      new SupervisedNetwork(model, new EntropyLossLayer)
    }
    log.p("With a the following component graph: ")
    networkGraph(log, trainingNetwork, 600)
    log.p("Note that this visualization does not expand DAGNetworks recursively")

    log.h2("Training")
    log.p("We train using a standard iterative L-BFGS strategy: ")
    val trainer = log.eval {
      val trainer: LbfgsTrainer = new LbfgsTrainer
      trainer.setVerbose(true)
      trainer.setTrainingSize(2000)
      trainer.setNet(trainingNetwork)
      trainer.setData(data.toArray)
      new IterativeTrainer(trainer) {
        override protected def onStep(step: IterativeTrainer.StepState): Unit = {
          System.err.println(s"${step.getIteration} - ${step.getFitness} in ${step.getEvaluationTime}s")
          println(s"${step.getIteration} - ${step.getFitness} in ${step.getEvaluationTime}s")
          super.onStep(step)
        }
      }
    }
    log.eval {
      val trainingContext = new TrainingContext
      trainingContext.terminalErr = 0.0
      trainingContext.setTimeout(trainingTimeMinutes, TimeUnit.MINUTES)
      val finalError = trainer.step(trainingContext).finalError
      System.out.println(s"Final Error = $finalError")
    }
    log.p("After training, we have the following parameterized model: ")
    log.eval {
      model.toString
    }
    log.p("A summary of the training timeline: ")
    summarizeHistory(log, trainer.history)

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
    log.out("Actual \\ Predicted | " + (0 to 9).mkString(" | "))
    log.out((0 to 10).map(_ ⇒ "---").mkString(" | "))
    (0 to 9).foreach(actual ⇒ {
      log.out(s" **$actual** | " + (0 to 9).map(prediction ⇒ {
        categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(prediction, 0)
      }).mkString(" | "))
    })
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

  private def summarizeHistory(log: ScalaMarkdownPrintStream, history: util.ArrayList[IterativeTrainer.StepState]) = {
    log.eval {
      val step = Math.max(Math.pow(10,Math.ceil(Math.log(history.size()) / Math.log(10))-2), 1).toInt
      TableOutput.create(history.asScala.filter(0==_.getIteration%step).map(state ⇒
        Map[String, AnyRef](
          "iteration" → state.getIteration.toInt.asInstanceOf[Integer],
          "time" → state.getEvaluationTime.toDouble.asInstanceOf[lang.Double],
          "fitness" → state.getFitness.toDouble.asInstanceOf[lang.Double]
        ).asJava
      ): _*)
    }
    log.eval {
      val plot: PlotCanvas = ScatterPlot.plot(history.asScala.map(item ⇒ Array[Double](
        item.getIteration, Math.log(item.getFitness)
      )).toArray: _*)
      plot.setTitle("Convergence Plot")
      plot.setAxisLabels("Iteration", "log(Fitness)")
      plot.setSize(600, 400)
      plot
    }
  }

  private def networkGraph(log: ScalaMarkdownPrintStream, dagNetwork: DAGNetwork, width: Int = 1200) = {
    log.eval {
      val nodes: List[DAGNode] = dagNetwork.getNodes.asScala.toList
      val graphNodes: Map[UUID, MutableNode] = nodes.map(node ⇒ {
        node.getId() → guru.nidi.graphviz.model.Factory.mutNode((node match {
          case n : InnerNode ⇒
            n.layer match {
              case _ if(n.layer.isInstanceOf[VerboseWrapper]) ⇒ n.layer.asInstanceOf[VerboseWrapper].inner.getClass.getSimpleName
              case _ ⇒ n.layer.getClass.getSimpleName
            }
          case _ ⇒ node.getClass.getSimpleName
        }) + "\n" + node.getId.toString)
      }).toMap
      val idMap: Map[UUID, List[UUID]] = nodes.flatMap((to: DAGNode) ⇒ {
        to.getInputs.map((from: DAGNode) ⇒ {
          from.getId → to.getId
        })
      }).groupBy(_._1).mapValues(_.map(_._2))
      nodes.foreach((to: DAGNode) ⇒ {
        graphNodes(to.getId).addLink(idMap.getOrElse(to.getId, List.empty).map(from ⇒ {
          Link.to(graphNodes(from))
        }): _*)
      })
      val nodeArray = graphNodes.values.map(_.asInstanceOf[LinkSource]).toArray
      val graph = guru.nidi.graphviz.model.Factory.graph().`with`(nodeArray: _*)
        .generalAttr.`with`(RankDir.TOP_TO_BOTTOM).directed()
      Graphviz.fromGraph(graph).width(width).render(Format.PNG).toImage
    }
  }

  def toOut(label: String): Int = {
    var i = 0
    while ( {
      i < 10
    }) {
      if (label == "[" + i + "]") return i

      {
        i += 1;
        i - 1
      }
    }
    throw new RuntimeException
  }

  def toOutNDArray(out: Int, max: Int): Tensor = {
    val ndArray = new Tensor(max)
    ndArray.set(out, 1)
    ndArray
  }

}