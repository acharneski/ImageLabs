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
import com.simiacryptus.mindseye.net.activation.{AbsActivationLayer, SigmoidActivationLayer, SoftmaxActivationLayer}
import com.simiacryptus.mindseye.net.basic.BiasLayer
import com.simiacryptus.mindseye.net.dag._
import com.simiacryptus.mindseye.net.dev.{DenseSynapseLayerJBLAS, ToeplitzSynapseLayerJBLAS}
import com.simiacryptus.mindseye.net.loss.MeanSqLossLayer
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.util.Util
import com.simiacryptus.util.ml.{Coordinate, Tensor}
import com.simiacryptus.util.test.MNIST
import com.simiacryptus.util.text.TableOutput
import guru.nidi.graphviz.engine.{Format, Graphviz}
import org.scalatest.{MustMatchers, WordSpec}
import smile.plot.{PlotCanvas, ScatterPlot}

import scala.collection.JavaConverters._

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
    val middleSize = Array[Int](10, 10, 1)
    val minutesPerPhase = 10

  "Train Digit Autoencoder Network" should {

    "LinearBias" in {
      report("linearBias", log ⇒ {
        test(log, log.eval {
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

    "SparseLinearBias" in {
      report("sparseLinearBias", log ⇒ {
        import scala.language.implicitConversions
        test(log, log.eval {
          val middleSize = Array[Int](10, 10, 1)
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

  def test(log: ScalaMarkdownPrintStream, model: AutoencoderNetwork) = {
    val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new MeanSqLossLayer())
    log.h2("Data")
    log.p("First, we load the training dataset: ")
    val data: Array[Array[Tensor]] = log.code(() ⇒ {
      MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
        Array(labeledObj.data, labeledObj.data)
      }).toList.map(x ⇒ Array(x(0),x(0))).toArray
    })
    _test(log, trainingNetwork, data)
    log.eval {
      TableOutput.create(MNIST.validationDataStream().iterator().asScala.toStream.take(100).map(testObj ⇒ {
        val result = model.eval(testObj.data).data.head
        Map[String, AnyRef](
          "Input" → log.image(testObj.data.toGrayImage(), testObj.label),
          "Output" → log.image(result.toGrayImage(), "Autoencoder Output")
        ).asJava
      }): _*)
    }

    val encoded = model.encoder.getLayer.eval(data.head.head).data.head
    val width = encoded.getDims()(0)
    val height = encoded.getDims()(1)
    log.draw(gfx ⇒ {
      (0 until width).foreach(x⇒{
        (0 until height).foreach(y⇒{
          encoded.fill(cvt((i:Int)⇒0.0))
          encoded.set(Array(x,y),1.0)
          val image = model.decoder.getLayer.eval(encoded).data.head
          val sum = image.getData.sum
          val min = image.getData.min
          val max = image.getData.max
          (0 until inputSize(0)).foreach(xx ⇒
            (0 until inputSize(1)).foreach(yy ⇒ {
              val value: Double = 255 * (image.get(xx, yy)-min) / (max-min)
              gfx.setColor(new Color(value.toInt, value.toInt, value.toInt))
              gfx.drawRect((x * inputSize(0)) + xx, (y * inputSize(1)) + yy, 1, 1)
            }))
        })
      })
    }, width = inputSize(0) * width, height = inputSize(1) * height)
  }

  def test(log: ScalaMarkdownPrintStream, trainingNetwork: SparseAutoencoderTrainer) = {
    log.h2("Data")
    log.p("First, we load the training dataset: ")
    val data: Array[Array[Tensor]] = log.code(() ⇒ {
      MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
        Array(labeledObj.data, labeledObj.data)
      }).toList.map(x ⇒ Array(x(0))).toArray
    })
    _test(log, trainingNetwork, data)
    log.eval {
      TableOutput.create(MNIST.validationDataStream().iterator().asScala.toStream.take(100).map(testObj ⇒ {
        var evalModel: PipelineNetwork = new PipelineNetwork
        evalModel.add(trainingNetwork.encoder.getLayer)
        evalModel.add(trainingNetwork.decoder.getLayer)
        val result = evalModel.eval(testObj.data).data.head
        Map[String, AnyRef](
          "Input" → log.image(testObj.data.toGrayImage(), testObj.label),
          "Output" → log.image(result.toGrayImage(), "Autoencoder Output")
        ).asJava
      }): _*)
    }

  }

  private def _test(log: ScalaMarkdownPrintStream, trainingNetwork: SupervisedNetwork, data: Array[Array[Tensor]]) = {
    log.h2("Training")
    val history = new scala.collection.mutable.ArrayBuffer[IterativeTrainer.Step]()
    log.eval {
      case class TrainingStep(sampleSize: Int, timeoutMinutes: Int)
      List(TrainingStep(2500, minutesPerPhase), TrainingStep(5000, minutesPerPhase), TrainingStep(20000, minutesPerPhase)).foreach(scheduledStep ⇒ {
        val trainable = new StochasticArrayTrainable(data, trainingNetwork, scheduledStep.sampleSize)
        val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
        trainer.setOrientation(new L12NormalizedConst(new LBFGS()).setFactor_L1(0.001))
        trainer.setMonitor(new TrainingMonitor {
          override def log(msg: String): Unit = {
            System.err.println(msg)
          }

          override def onStepComplete(currentPoint: IterativeTrainer.Step): Unit = {
            history += currentPoint
          }
        })
        trainer.setTimeout(scheduledStep.timeoutMinutes, TimeUnit.MINUTES)
        trainer.setTerminateThreshold(0.0)
        trainer.run()
      })
    }
    log.p("A summary of the training timeline: ")
    summarizeHistory(log, history.toList)
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