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

import java.util.concurrent.TimeUnit
import java.util.function.ToDoubleFunction
import java.{lang, util}

import com.simiacryptus.mindseye.net.{AutoencoderNetwork, PipelineNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.net.activation.AbsActivationLayer
import com.simiacryptus.mindseye.net.dag._
import com.simiacryptus.mindseye.net.dev.{DenseSynapseLayerJBLAS, ToeplitzSynapseLayerJBLAS}
import com.simiacryptus.mindseye.net.loss.SqLossLayer
import com.simiacryptus.mindseye.training.{IterativeTrainer, LbfgsTrainer, TrainingContext}
import com.simiacryptus.util.Util
import com.simiacryptus.util.ml.{Coordinate, Tensor}
import com.simiacryptus.util.test.MNIST
import com.simiacryptus.util.text.TableOutput
import guru.nidi.graphviz.engine.{Format, Graphviz}
import org.scalatest.{MustMatchers, WordSpec}
import smile.plot.{PlotCanvas, ScatterPlot}

import scala.collection.JavaConverters._

class AutoencoderDemo extends WordSpec with MustMatchers with MarkdownReporter {

    val inputSize = Array[Int](28, 28, 1)
    val middleSize = Array[Int](6, 6, 1)
    var trainingTimeMinutes = 5

  "Train Digit Autoencoder Network" should {

    "Linear" in {
      report("linear", log ⇒ {
        test(log, log.eval {
          trainingTimeMinutes = 60
          new AutoencoderNetwork({
            var model: PipelineNetwork = new PipelineNetwork
            model.add(new DenseSynapseLayerJBLAS(inputSize, middleSize).setWeights(new ToDoubleFunction[Coordinate] {
              override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
            }))
            model
          }, {
            var model: PipelineNetwork = new PipelineNetwork
            model.add(new DenseSynapseLayerJBLAS(middleSize, inputSize).setWeights(new ToDoubleFunction[Coordinate] {
              override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
            }))
            model
          })
        })
      })
    }

    "Flat 2-Layer Abs" in {
      report("twolayerabs", log ⇒ {
        test(log, log.eval {
          trainingTimeMinutes = 60
          new AutoencoderNetwork({
            var model: PipelineNetwork = new PipelineNetwork
            model.add(new DenseSynapseLayerJBLAS(inputSize, inputSize).setWeights(new ToDoubleFunction[Coordinate] {
              override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
            }))
            model.add(new AbsActivationLayer)
            model.add(new DenseSynapseLayerJBLAS(inputSize, middleSize).setWeights(new ToDoubleFunction[Coordinate] {
              override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
            }))
            model
          }, {
            var model: PipelineNetwork = new PipelineNetwork
            model.add(new DenseSynapseLayerJBLAS(middleSize, inputSize).setWeights(new ToDoubleFunction[Coordinate] {
              override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
            }))
            model.add(new AbsActivationLayer)
            model.add(new DenseSynapseLayerJBLAS(inputSize, inputSize).setWeights(new ToDoubleFunction[Coordinate] {
              override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
            }))
            model
          })
        })
      })
    }

    "Toeplitz 2-Layer Abs" in {
      report("twolayertoeplitz", log ⇒ {
        test(log, log.eval {
          trainingTimeMinutes = 60
          new AutoencoderNetwork({
            var model: PipelineNetwork = new PipelineNetwork
            model.add(new ToeplitzSynapseLayerJBLAS(inputSize, inputSize).setWeights(new ToDoubleFunction[Coordinate] {
              override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
            }))
            model.add(new AbsActivationLayer)
            model.add(new ToeplitzSynapseLayerJBLAS(inputSize, middleSize).setWeights(new ToDoubleFunction[Coordinate] {
              override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
            }))
            model
          }, {
            var model: PipelineNetwork = new PipelineNetwork
            model.add(new ToeplitzSynapseLayerJBLAS(middleSize, inputSize).setWeights(new ToDoubleFunction[Coordinate] {
              override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
            }))
            model.add(new AbsActivationLayer)
            model.add(new ToeplitzSynapseLayerJBLAS(inputSize, inputSize).setWeights(new ToDoubleFunction[Coordinate] {
              override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
            }))
            model
          })
        })
      })
    }

  }

  def test(log: ScalaMarkdownPrintStream, model: AutoencoderNetwork) = {
    log.h2("Data")
    log.p("First, we load the training dataset: ")
    val data: Seq[Array[Tensor]] = log.code(() ⇒ {
      MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
        Array(labeledObj.data, labeledObj.data)
      }).toList
    })

    log.p("We can visualize this network as a graph: ")
    networkGraph(log, model, 800)
    log.p("We encapsulate our model network within a supervisory network that applies a loss function: ")
    val trainingNetwork: SupervisedNetwork = log.eval {
      new SupervisedNetwork(model, new SqLossLayer)
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
    Graphviz.fromGraph(NetworkViz.toGraph(dagNetwork)).width(width).render(Format.PNG).toImage
  }

}