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

import java.io.File
import java.lang
import java.util.concurrent.TimeUnit
import java.util.function.ToDoubleFunction

import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.layers.activation.{AbsActivationLayer, SoftmaxActivationLayer}
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer
import com.simiacryptus.mindseye.layers.media.{ImgConvolutionSynapseLayer, MaxSubsampleLayer}
import com.simiacryptus.mindseye.layers.synapse.{BiasLayer, DenseSynapseLayer, ToeplitzSynapseLayer}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.orient.LBFGS
import com.simiacryptus.mindseye.opt.trainable.StochasticArrayTrainable
import com.simiacryptus.util.Util
import com.simiacryptus.util.io.IOUtil
import com.simiacryptus.util.ml.{Coordinate, Tensor}
import com.simiacryptus.util.test.MNIST
import com.simiacryptus.util.text.TableOutput
import org.scalatest.{MustMatchers, WordSpec}
import smile.plot.{PlotCanvas, ScatterPlot}
import util.{ReportNotebook, ScalaNotebookOutput}

import scala.collection.JavaConverters._

class MnistNetDemo extends WordSpec with MustMatchers with ReportNotebook {

  val terminationThreshold = 0.01
  val inputSize = Array[Int](28, 28, 1)
  val outputSize = Array[Int](10)
  var trainingTimeMinutes = 5
  "Train Digit Recognizer Network" should {

    "Flat Logistic Regression" in {
      report("simple", log ⇒ {
        test(log, log.eval {
          trainingTimeMinutes = 2
          var model: PipelineNetwork = new PipelineNetwork
          model.add(new DenseSynapseLayer(inputSize, outputSize).setWeights(new ToDoubleFunction[Coordinate] {
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
          trainingTimeMinutes = 10
          val middleSize = Array[Int](28, 28, 1)
          var model: PipelineNetwork = new PipelineNetwork
          model.add(new DenseSynapseLayer(inputSize, middleSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new BiasLayer(middleSize: _*))
          model.add(new AbsActivationLayer)
          model.add(new DenseSynapseLayer(middleSize, outputSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new BiasLayer(outputSize: _*))
          model.add(new SoftmaxActivationLayer)
          model
        })
      })
    }

    "Flat 2-Layer ReLu" in {
      report("twolayerrelu", log ⇒ {
        test(log, log.eval {
          trainingTimeMinutes = 10
          val middleSize = Array[Int](28, 28, 1)
          var model: PipelineNetwork = new PipelineNetwork
          model.add(new DenseSynapseLayer(inputSize, middleSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new BiasLayer(middleSize: _*))
          model.add(new AbsActivationLayer)
          model.add(new DenseSynapseLayer(middleSize, outputSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new BiasLayer(outputSize: _*))
          model.add(new SoftmaxActivationLayer)
          model
        })
      })
    }

    "Toeplitz 2-Layer Abs" in {
      report("twolayertoeplitz", log ⇒ {
        test(log, log.eval {
          trainingTimeMinutes = 60
          val middleSize = Array[Int](28, 28, 1)
          var model: PipelineNetwork = new PipelineNetwork
          model.add(new ToeplitzSynapseLayer(inputSize, middleSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new AbsActivationLayer)
          model.add(new DenseSynapseLayer(middleSize, outputSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new BiasLayer(outputSize: _*))
          model.add(new SoftmaxActivationLayer)
          model
        })
      })
    }

    "ToeplitzMax 2-Layer Abs" in {
      report("twolayertoeplitzmax", log ⇒ {
        test(log, log.eval {
          trainingTimeMinutes = 60
          val middleSize1 = Array[Int](28, 28, 4)
          val middleSize2 = Array[Int](14, 14, 4)
          var model: PipelineNetwork = new PipelineNetwork
          model.add(new ToeplitzSynapseLayer(inputSize, middleSize1).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new AbsActivationLayer)
          model.add(new MaxSubsampleLayer(2, 2, 1))
          model.add(new DenseSynapseLayer(middleSize2, outputSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new BiasLayer(outputSize: _*))
          model.add(new SoftmaxActivationLayer)
          model
        })
      })
    }

    "ToeplitzMax 3-Layer Abs" in {
      report("threelayertoeplitzmax", log ⇒ {
        test(log, log.eval {
          trainingTimeMinutes = 60
          val middleSize1 = Array[Int](28, 28, 4)
          val middleSize2 = Array[Int](14, 14, 4)
          val middleSize3 = Array[Int](14, 14, 16)
          val middleSize4 = Array[Int](7, 7, 16)
          var model: PipelineNetwork = new PipelineNetwork
          model.add(new ToeplitzSynapseLayer(inputSize, middleSize1).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new AbsActivationLayer)
          model.add(new MaxSubsampleLayer(2, 2, 1))
          model.add(new ToeplitzSynapseLayer(middleSize2, middleSize3).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new AbsActivationLayer)
          model.add(new MaxSubsampleLayer(2, 2, 1))
          model.add(new DenseSynapseLayer(middleSize4, outputSize).setWeights(new ToDoubleFunction[Coordinate] {
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
          trainingTimeMinutes = 60
          val middleSize = Array[Int](28, 28, 1)
          var model: PipelineNetwork = new PipelineNetwork
          model.add(new ImgConvolutionSynapseLayer(2, 2, 2).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new AbsActivationLayer)
          model.add(new MaxSubsampleLayer(2, 2, 1))
          model.add(new ImgConvolutionSynapseLayer(2, 2, 2).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new AbsActivationLayer)
          model.add(new MaxSubsampleLayer(2, 2, 1))

          def headDims = model.eval(new Tensor(inputSize: _*)).data(0).getDims

          model.add(new DenseSynapseLayer(headDims, outputSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.001
          }))
          model.add(new BiasLayer(headDims: _*))
          model.add(new SoftmaxActivationLayer)
          model
        })
      })
    }

  }

  def test(log: ScalaNotebookOutput, model: PipelineNetwork) = {
    log.h2("Data")
    log.p("First, we cache the training dataset: ")
    val trainingData: Seq[Array[Tensor]] = log.code(() ⇒ {
      MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
        Array(labeledObj.data, toOutNDArray(toOut(labeledObj.label), 10))
      }).toList
    })
    log.h2("Training")
    log.p("We encapsulate our model network within a supervisory network that applies a loss function, then newTrainer using a standard iterative L-BFGS strategy: ")
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
    log.p("First we pretrain the model on a very small dataset until it is at a reasonable starting point")
    log.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
      val trainable = new StochasticArrayTrainable(trainingData.toArray, trainingNetwork, 100)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
      trainer.setOrientation(new LBFGS())
      trainer.setMonitor(monitor)
      trainer.setTimeout(Math.min(trainingTimeMinutes, 10), TimeUnit.MINUTES)
      trainer.setTerminateThreshold(1.0)
      trainer.run()
    }
    log.p("The second phase of training uses more data")
    log.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
      val trainable = new StochasticArrayTrainable(trainingData.toArray, trainingNetwork, 2000)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
      trainer.setOrientation(new LBFGS())
      trainer.setMonitor(monitor)
      trainer.setTimeout(trainingTimeMinutes, TimeUnit.MINUTES)
      trainer.setTerminateThreshold(0.05)
      trainer.run()
    }

    log.p("After training, we save the following parameterized model: ")
    log.eval {
      IOUtil.writeKryo(model, log.file("network.kryo"))
      IOUtil.readKryo[PipelineNetwork](new File("reports/MnistNetDemo/etc/network.kryo"))
    }
    log.p("A summary of the training timeline: ")
    summarizeHistory(log, history.toList)

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