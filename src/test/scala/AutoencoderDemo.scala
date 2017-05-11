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

import java.awt.Color
import java.util.concurrent.TimeUnit
import java.util.function.{IntToDoubleFunction, ToDoubleFunction}
import java.{lang, util}

import com.simiacryptus.mindseye.net._
import com.simiacryptus.mindseye.net.activation.{AbsActivationLayer, ReLuActivationLayer, SigmoidActivationLayer, SoftmaxActivationLayer}
import com.simiacryptus.mindseye.net.basic.BiasLayer
import com.simiacryptus.mindseye.net.dag._
import com.simiacryptus.mindseye.net.dev.{DenseSynapseLayerJBLAS, ToeplitzSynapseLayerJBLAS}
import com.simiacryptus.mindseye.net.loss.MeanSqLossLayer
import com.simiacryptus.mindseye.opt.{OrientationStrategy, _}
import com.simiacryptus.util.{IO, Util}
import com.simiacryptus.util.ml.{Coordinate, Tensor}
import com.simiacryptus.util.test.MNIST
import com.simiacryptus.util.text.TableOutput
import guru.nidi.graphviz.engine.{Format, Graphviz}
import org.scalatest.{MustMatchers, WordSpec}
import smile.plot.{PlotCanvas, ScatterPlot}

import scala.collection.JavaConverters._
import scala.util.Random

object AutoencoderDemo {

  implicit def cvt(fn:Int⇒Double) : IntToDoubleFunction = {
    new IntToDoubleFunction {
      override def applyAsDouble(v : Int): Double = fn(v)
    }
  }

  implicit def cvt[T](fn:T⇒Double) : ToDoubleFunction[T] = {
    new ToDoubleFunction[T] {
      override def applyAsDouble(v : T): Double = fn(v)
    }
  }

}
import AutoencoderDemo._

class AutoencoderDemo extends WordSpec with MustMatchers with MarkdownReporter {

    val inputSize = Array[Int](28, 28, 1)
    case class TrainingStep(sampleSize: Int, timeoutMinutes: Int, endFitness : Double, orient : OrientationStrategy, step : LineSearchStrategy)
    var schedule = List(
      TrainingStep(5000, 30, 100.0,
        new LBFGS(),
        new ArmijoWolfeConditions().setC2(0.99)
      ),
      TrainingStep(10000, 30, 10.0,
        new LBFGS(),
        new ArmijoWolfeConditions()
      )
    )

  "Train Digit Autoencoder Network" should {

    "LinearBias" in {
      report("linearBias", log ⇒ {
        testAutoencoder(log, log.eval {
          val middleSize = Array[Int](10, 10, 1)
          new AutoencoderNetwork({
            var model: PipelineNetwork = new PipelineNetwork
            model.add(new DenseSynapseLayerJBLAS(inputSize, middleSize)
              .setWeights(cvt((c:Coordinate)⇒Util.R.get.nextGaussian * 0.001)))
            model.add(new BiasLayer(middleSize:_*).setWeights(cvt(i⇒0.0)))
            model
          }, {
            var model: PipelineNetwork = new PipelineNetwork
            model.add(new DenseSynapseLayerJBLAS(middleSize, inputSize)
              .setWeights(cvt((c:Coordinate)⇒Util.R.get.nextGaussian * 0.001)))
            model.add(new BiasLayer(inputSize:_*).setWeights(cvt(i⇒0.0)))
            model
          })
        })
      })
    }

    "ReLu" in {
      report("relu", log ⇒ {
        testAutoencoder(log, log.eval {
          val middleSize = Array[Int](10, 10, 1)
          new AutoencoderNetwork({
            var model: PipelineNetwork = new PipelineNetwork
            model.add(new DenseSynapseLayerJBLAS(inputSize, middleSize)
              .setWeights(cvt((c:Coordinate)⇒Util.R.get.nextGaussian * 0.001)))
            model.add(new BiasLayer(middleSize:_*).setWeights(cvt(i⇒0.0)))
            model.add(new ReLuActivationLayer)
            model
          }, {
            var model: PipelineNetwork = new PipelineNetwork
            model.add(new DenseSynapseLayerJBLAS(middleSize, inputSize)
              .setWeights(cvt((c:Coordinate)⇒Util.R.get.nextGaussian * 0.001)))
            model.add(new BiasLayer(inputSize:_*).setWeights(cvt(i⇒0.0)))
            model
          })
        })
      })
    }

    "SparseLinearBias" in {
      report("sparseLinearBias", log ⇒ {
        import scala.language.implicitConversions
        testSparse(log, log.eval {
          val middleSize = Array[Int](50, 50, 1)
          new SparseAutoencoderTrainer({
            var model: PipelineNetwork = new PipelineNetwork
            model.add(new DenseSynapseLayerJBLAS(inputSize, middleSize)
              .setWeights(cvt((c:Coordinate)⇒Util.R.get.nextGaussian * 0.0001)))
            model.add(new BiasLayer(middleSize:_*).setWeights(cvt(i⇒0.0)))
            model.add(new SigmoidActivationLayer().setBalanced(false))
            model
          }, {
            var model: PipelineNetwork = new PipelineNetwork
            model.add(new DenseSynapseLayerJBLAS(middleSize, inputSize)
              .setWeights(cvt((c:Coordinate)⇒Util.R.get.nextGaussian * 0.0001)))
            model.add(new BiasLayer(inputSize:_*).setWeights(cvt(i⇒0.0)))
            model
          })
        })
      })
    }


  }

  def testAutoencoder(log: ScalaMarkdownPrintStream, model: ⇒AutoencoderNetwork) = {
    def trainingNetwork = new SimpleLossNetwork(model, new MeanSqLossLayer())
    log.h2("Data")
    log.p("First, we load the training dataset: ")
    val data: Array[Array[Tensor]] = log.code(() ⇒ {
      MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
        Array(labeledObj.data, labeledObj.data)
      }).toList.map(x ⇒ Array(x(0),x(0))).toArray
    })
    val currentNetwork = train(log, trainingNetwork, data)
    val resultModel = currentNetwork.studentNode.getLayer.asInstanceOf[AutoencoderNetwork]
    report(log, data, resultModel.encoder, resultModel.decoder)
  }

  def testSparse(log: ScalaMarkdownPrintStream, trainingNetwork: ⇒SparseAutoencoderTrainer) = {
    log.h2("Data")
    log.p("First, we load the training dataset: ")
    val data: Array[Array[Tensor]] = log.code(() ⇒ {
      MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
        Array(labeledObj.data, labeledObj.data)
      }).toList.map(x ⇒ Array(x(0))).toArray
    })
    val currentNetwork = train(log, trainingNetwork, data)
    report(log, data, currentNetwork.encoder, currentNetwork.decoder)
  }

  private def report(log: ScalaMarkdownPrintStream, data: Array[Array[Tensor]], encoder: DAGNode, decoder: DAGNode) = {
    log.eval {
      TableOutput.create(MNIST.validationDataStream().iterator().asScala.toStream.take(100).map(testObj ⇒ {
        var evalModel: PipelineNetwork = new PipelineNetwork
        evalModel.add(encoder.getLayer)
        evalModel.add(decoder.getLayer)
        val result = evalModel.eval(testObj.data).data.head
        Map[String, AnyRef](
          "Input" → log.image(testObj.data.toGrayImage(), testObj.label),
          "Output" → log.image(result.toGrayImage(), "Autoencoder Output")
        ).asJava
      }): _*)
    }
    val encoded = encoder.getLayer.eval(data.head.head).data.head
    val width = encoded.getDims()(0)
    val height = encoded.getDims()(1)
    log.draw(gfx ⇒ {
      (0 until width).foreach(x ⇒ {
        (0 until height).foreach(y ⇒ {
          encoded.fill(cvt((i: Int) ⇒ 0.0))
          encoded.set(Array(x, y), 1.0)
          val image = decoder.getLayer.eval(encoded).data.head
          val sum = image.getData.sum
          val min = image.getData.min
          val max = image.getData.max
          (0 until inputSize(0)).foreach(xx ⇒
            (0 until inputSize(1)).foreach(yy ⇒ {
              val value: Double = 255 * (image.get(xx, yy) - min) / (max - min)
              gfx.setColor(new Color(value.toInt, value.toInt, value.toInt))
              gfx.drawRect((x * inputSize(0)) + xx, (y * inputSize(1)) + yy, 1, 1)
            }))
        })
      })
    }, width = inputSize(0) * width, height = inputSize(1) * height)
  }

  private def train[T <: SupervisedNetwork](log: ScalaMarkdownPrintStream, trainingNetwork: ⇒T, data: Array[Array[Tensor]]) : T = {
    val history = new scala.collection.mutable.ArrayBuffer[IterativeTrainer.Step]()
    log.h2("Pre-Training")
    var currentNetwork: Option[T] = None
    log.eval {
      var trainer : IterativeTrainer = null
      val pretrainingThreshold = 4000.0
      do {
        currentNetwork = Option(trainingNetwork)
        val trainable = new StochasticArrayTrainable(data.filter(_⇒Random.nextDouble() < (100.0 / data.length)), currentNetwork.get, Integer.MAX_VALUE)
        //val normalized = new L12Normalizer(trainable).setFactor_L1(0.0).setFactor_L2(1.0)
        trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
        trainer.setOrientation(new GradientDescent())
        trainer.setScaling(new ArmijoWolfeConditions().setAlpha(1e-7).setC2(0.99).setC1(1e-7))
        trainer.setMonitor(new TrainingMonitor {
          override def log(msg: String): Unit = {
            System.err.println(msg)
          }

          override def onStepComplete(currentPoint: IterativeTrainer.Step): Unit = {
            history += currentPoint
          }
        })
        trainer.setMaxIterations(10)
        trainer.setTimeout(1, TimeUnit.MINUTES)
        trainer.setTerminateThreshold(pretrainingThreshold)
      } while(trainer.run() > pretrainingThreshold)
    }
    log.h2("Training")
    schedule.foreach(scheduledStep ⇒ {
      log.h3(scheduledStep.toString)
      log.eval {
        val trainable = new StochasticArrayTrainable(data, currentNetwork.get, scheduledStep.sampleSize)
        //val normalized = new L12Normalizer(trainable).setFactor_L1(0.0).setFactor_L2(1.0)
        val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
        trainer.setOrientation(scheduledStep.orient)
        trainer.setScaling(scheduledStep.step)
        trainer.setMonitor(new TrainingMonitor {
          override def log(msg: String): Unit = {
            System.err.println(msg)
          }

          override def onStepComplete(currentPoint: IterativeTrainer.Step): Unit = {
            history += currentPoint
          }
        })
        trainer.setTimeout(scheduledStep.timeoutMinutes, TimeUnit.MINUTES)
        trainer.setTerminateThreshold(scheduledStep.endFitness)
        trainer.run()
      }
    })
    log.p("A summary of the training timeline: ")
    summarizeHistory(log, history.toList)
    IO.writeKryo(currentNetwork.get, log.newFile(MarkdownReporter.currentMethod + ".kryo.gz"))
    currentNetwork.get
  }

  private def summarizeHistory(log: ScalaMarkdownPrintStream, history: List[com.simiacryptus.mindseye.opt.IterativeTrainer.Step]) = {
    log.eval {
      val step = Math.max(Math.pow(10,Math.ceil(Math.log(history.size) / Math.log(10))-2), 1).toInt
      TableOutput.create(history.filter(0==_.iteration%step).map(state ⇒
        Map[String, AnyRef](
          "iteration" → state.iteration.toInt.asInstanceOf[Integer],
          "time" → state.time.toDouble.asInstanceOf[lang.Double],
          "fitness" → state.point.value.toDouble.asInstanceOf[lang.Double]
        ).asJava
      ): _*)
    }
    if(!history.isEmpty) log.eval {
      val plot: PlotCanvas = ScatterPlot.plot(history.map(item ⇒ Array[Double](
        item.iteration, Math.log(item.point.value)
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