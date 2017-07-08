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

package report

import java.lang
import java.util.concurrent.TimeUnit
import java.util.function.{IntToDoubleFunction, ToDoubleFunction}

import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import util.Java8Util._
import util.{ReportNotebook, ScalaNotebookOutput}
import com.simiacryptus.mindseye.layers._
import com.simiacryptus.mindseye.layers.activation.{ReLuActivationLayer, SoftmaxActivationLayer}
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer
import com.simiacryptus.mindseye.layers.synapse.{BiasLayer, DenseSynapseLayer}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeConditions
import com.simiacryptus.mindseye.opt.trainable.{ConstL12Normalizer, StochasticArrayTrainable, Trainable}
import com.simiacryptus.util.Util
import com.simiacryptus.util.ml.{Coordinate, Tensor}
import com.simiacryptus.util.test.MNIST
import com.simiacryptus.util.text.TableOutput
import org.scalatest.{MustMatchers, WordSpec}
import smile.plot.{PlotCanvas, ScatterPlot}

import scala.collection.JavaConverters._

class OptimizerDemo extends WordSpec with MustMatchers with ReportNotebook {

  val schedule = List(
    TrainingStep(200, 5, 0.5), TrainingStep(1000, 5, 0.1)
  )
  val terminationThreshold = 0.01
  val inputSize = Array[Int](28, 28, 1)
  val outputSize = Array[Int](10) // Array[Int](28, 28)
  val middleSize = Array[Int](28, 28, 1)
  val lossLayer = new EntropyLossLayer // new MeanSqLossLayer

  def test(log: ScalaNotebookOutput, optimizer: Trainable ⇒ IterativeTrainer) = {
    log.h2("Model Problem: ")
    val (model, trainingData) = testProblem_category(log)

    log.h2("Training")
    val history = new scala.collection.mutable.ArrayBuffer[com.simiacryptus.mindseye.opt.Step]()
    val monitor = log.eval {
      val monitor = new TrainingMonitor {
        override def log(msg: String): Unit = {
          System.err.println(msg)
        }

        override def onStepComplete(currentPoint: Step): Unit = {
          history += currentPoint
        }
      }
      monitor
    }
    schedule.foreach(scheduledStep ⇒ {
      log.h3(scheduledStep.toString)
      log.eval {
        System.out.println(scheduledStep)
        val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, lossLayer)
        val trainable: StochasticArrayTrainable = new StochasticArrayTrainable(trainingData.toArray, trainingNetwork, scheduledStep.sampleSize)
        val trainer: IterativeTrainer = optimizer.apply(trainable)
        trainer.setMonitor(monitor)
        trainer.setTimeout(scheduledStep.timeoutMinutes, TimeUnit.MINUTES)
        trainer.setTerminateThreshold(scheduledStep.terminationThreshold)
        trainer.run()
      }
      log.eval {
        getBlankDeltaSet(model).map.asScala.map(ent ⇒ {
          val (layer, buffer) = ent
          Map(
            "layer" → layer.getClass.getSimpleName,
            "id" → layer.getId
          ) ++ summarize(buffer.target)
        }).mkString("\n")
      }
    })
    log.p("A summary of the training timeline: ")
    summarizeHistory(log, history.toList)
  }

  "Various Optimization Strategies" should {

    "LBFGS" in {
      report("lbfgs", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new LBFGS()) // This is the default
          trainer
        })
      })
    }

    "L1 Normalized SGD" in {
      report("l1sgd", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val normalized = new ConstL12Normalizer(trainable).setFactor_L1(-1000.0)
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(normalized)
          trainer.setOrientation(new GradientDescent())
          trainer
        })
      })
    }

    "L2 Normalized SGD" in {
      report("l2sgd", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val normalized = new ConstL12Normalizer(trainable).setFactor_L1(0.0).setFactor_L2(1.0)
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(normalized)
          trainer.setOrientation(new GradientDescent())
          trainer
        })
      })
    }

    "OWL-QN" in {
      report("owlqn", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new OwlQn())
          // Not needed, just for illustration:
          trainer.setLineSearchFactory(()⇒new ArmijoWolfeConditions().setC2(0.8).setAlpha(1e-6))
          trainer
        })
      })
    }


  }

  private def summarize(data: Array[Double]) = {
    val zeroTol = 1e-20
    val zeros = data.map(x ⇒ if (Math.abs(x) < zeroTol) 1.0 else 0.0).sum
    Map(
      "length" → data.length,
      "sparsity" → (zeros / data.length),
      "raw" → summarize2(data),
      "abs" → summarize2(data.map(x ⇒ Math.abs(x))),
      "log10" → summarize2(data.filter(x ⇒ Math.abs(x) > zeroTol).map(x ⇒ Math.log(Math.abs(x)) / Math.log(10)))
    )
  }

  private def summarize2(data: Array[Double]) = {
    if (data.isEmpty) {
      Map.empty
    } else {
      val sum = data.sum
      val sumSq = data.map(x ⇒ x * x).sum
      Map(
        "max" → data.max,
        "min" → data.min,
        "mean" → (sum / data.length),
        "stdDev" → Math.sqrt((sumSq / data.length) - Math.pow(sum / data.length, 2))
      )
    }
  }

  private def getBlankDeltaSet(model: PipelineNetwork) = {
    val set = new DeltaSet()
    model.eval(new Tensor(inputSize: _*)).accumulate(set, Array(new Tensor(outputSize: _*)))
    set
  }

  private def testProblem_category(log: ScalaNotebookOutput) = {
    log.eval {
      val data = MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
        Array(labeledObj.data, toOutNDArray(toOut(labeledObj.label), 10))
      }).toList

      val middleSize = Array[Int](28, 28, 1)
      var model: PipelineNetwork = new PipelineNetwork
      model.add(new DenseSynapseLayer(inputSize, middleSize)
        .setWeights(cvt((c: Coordinate) ⇒ Util.R.get.nextGaussian * 0.001)))
      model.add(new BiasLayer(middleSize: _*))
      model.add(new ReLuActivationLayer().freeze)
      model.add(new DenseSynapseLayer(middleSize, outputSize)
        .setWeights(cvt((c: Coordinate) ⇒ Util.R.get.nextGaussian * 0.001)))
      model.add(new BiasLayer(outputSize: _*))
      model.add(new SoftmaxActivationLayer)
      (model, data)
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

  private def summarizeHistory(log: ScalaNotebookOutput, history: List[com.simiacryptus.mindseye.opt.Step]) = {
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
    if (!history.isEmpty) log.eval {
      val plot: PlotCanvas = ScatterPlot.plot(history.map(item ⇒ Array[Double](
        item.iteration, Math.log(item.point.value)
      )).toArray: _*)
      plot.setTitle("Convergence Plot")
      plot.setAxisLabels("Iteration", "log(Fitness)")
      plot.setSize(600, 400)
      plot
    }
  }

  private def testProblem_autoencoder(log: ScalaNotebookOutput) = {
    log.eval {
      val data = MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
        Array(labeledObj.data, labeledObj.data)
      }).toList

      var model: PipelineNetwork = new PipelineNetwork
      model.add(new DenseSynapseLayer(inputSize, middleSize)
        .setWeights(cvt((c: Coordinate) ⇒ Util.R.get.nextGaussian * 0.001)))
      model.add(new BiasLayer(middleSize: _*))
      //model.add(new ReLuActivationLayer().freeze)
      model.add(new DenseSynapseLayer(middleSize, inputSize)
        .setWeights(cvt((c: Coordinate) ⇒ Util.R.get.nextGaussian * 0.001)))
      model.add(new BiasLayer(outputSize: _*))
      (model, data)
    }
  }

  case class TrainingStep(sampleSize: Int, timeoutMinutes: Int, terminationThreshold: Double)

}

object OptimizerDemo {

  implicit def cvt(fn: Int ⇒ Double): IntToDoubleFunction = {
    new IntToDoubleFunction {
      override def applyAsDouble(v: Int): Double = fn(v)
    }
  }

  implicit def cvt[T](fn: T ⇒ Double): ToDoubleFunction[T] = {
    new ToDoubleFunction[T] {
      override def applyAsDouble(v: T): Double = fn(v)
    }
  }

}