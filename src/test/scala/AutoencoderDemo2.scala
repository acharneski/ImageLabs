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
import java.lang
import java.util.concurrent.TimeUnit
import java.util.function.{IntToDoubleFunction, ToDoubleBiFunction, ToDoubleFunction}
import javax.imageio.ImageIO

import com.simiacryptus.mindseye.net._
import com.simiacryptus.mindseye.net.activation._
import com.simiacryptus.mindseye.graph.dag._
import com.simiacryptus.mindseye.net.loss.MeanSqLossLayer
import com.simiacryptus.mindseye.net.synapse.{BiasLayer, DenseSynapseLayer, MappedSynapseLayer, TransposedSynapseLayer}
import com.simiacryptus.mindseye.opt.{OrientationStrategy, _}
import com.simiacryptus.util.ml.{Coordinate, Tensor}
import com.simiacryptus.util.test.MNIST
import com.simiacryptus.util.text.TableOutput
import com.simiacryptus.util.{IO, ImageTiles, Util}
import org.scalatest.{MustMatchers, WordSpec}
import smile.plot.{PlotCanvas, ScatterPlot}

import scala.collection.JavaConverters._
import scala.collection.mutable
import AutoencoderUtil._
import com.simiacryptus.mindseye.graph.{AutoencoderNetwork, PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}

class AutoencoderDemo2 extends WordSpec with MustMatchers with MarkdownReporter {

  val inputSize = Array[Int](28, 28, 1)
  val l1normalization = 0.0
  case class TrainingStep(sampleSize: Int, timeoutMinutes: Int, endFitness : Double, orient : OrientationStrategy, step : LineSearchStrategy)
  var schedule = List(
    TrainingStep(5000, 10, 100.0,
      new LBFGS(),
      new ArmijoWolfeConditions().setC2(0.99).setAlpha(1e-6)
    ),
    TrainingStep(10000, 5, 10.0,
      new LBFGS(),
      new ArmijoWolfeConditions()
    )
  )
  val history = new mutable.ArrayBuffer[com.simiacryptus.mindseye.opt.IterativeTrainer.Step]
  var monitor = new TrainingMonitor {
    override def log(msg: String): Unit = {
      System.err.println(msg)
    }
    override def onStepComplete(currentPoint: IterativeTrainer.Step): Unit = {
      history += currentPoint
    }
  }

  val data: Array[Array[Tensor]] = {
    MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
      Array(labeledObj.data, labeledObj.data)
    }).toList.map(x ⇒ Array(x(0),x(0))).toArray
  }

  "Train Digit Autoencoder Network" should {

    "Transfer" in {
      report("transfer2", log ⇒ {

        //ImageTiles.tilesRgb(ImageIO.read(getClass.getResourceAsStream("/monkey1.jpg")), 10, 10)



        log.h2("Data")
        val l1normalization = 1.0

        val mnist = MNIST.trainingDataStream().iterator().asScala.toStream
        val perLabel = mnist.groupBy(_.label).values.toList
        val smallDataset1 = perLabel.flatMap((_.take(10))).map(data ⇒ {
          Array(data.data, data.data)
        }).map(x ⇒ Array(x(0),x(0))).toArray
        val smallDataset2 = perLabel.flatMap((_.drop(10).take(10))).map(data ⇒ {
          Array(data.data, data.data)
        }).map(x ⇒ Array(x(0),x(0))).toArray

        val ae1 = new Object() {
          val middleSize = Array[Int](10, 10, 1)
          val inputNoise = new GaussianNoiseLayer().setValue(1.0)
          val encoderSynapse = new DenseSynapseLayer(inputSize, middleSize)
          val encoderBias = new BiasLayer(middleSize: _*).setWeights(cvt(i ⇒ 0.0))
          val encoderActivation = new ReLuActivationLayer().freeze()
          val encodedNoise = new DropoutNoiseLayer().setValue(0.2)
          val decoderBias = new BiasLayer(inputSize: _*).setWeights(cvt(i ⇒ 0.0))
          var decoderSynapse : NNLayer = new TransposedSynapseLayer(encoderSynapse)
          var decoderActivation : NNLayer = new ReLuActivationLayer().freeze()
        }

        monitor = new TrainingMonitor {
          override def log(msg: String): Unit = {
            System.err.println(msg)
          }
          override def onStepComplete(currentPoint: IterativeTrainer.Step): Unit = {
            ae1.inputNoise.shuffle()
            ae1.encodedNoise.shuffle()
            history += currentPoint
          }
        }

        var network = new AutoencoderNetwork({
          var model: PipelineNetwork = new PipelineNetwork
          model.add(ae1.inputNoise)
          model.add(ae1.encoderSynapse)
          model.add(ae1.encoderBias)
          model.add(ae1.encoderActivation)
          model.add(ae1.encodedNoise)
          model
        }, {
          var model: PipelineNetwork = new PipelineNetwork
          model.add(ae1.decoderSynapse)
          model.add(ae1.decoderBias)
          model.add(ae1.decoderActivation)
          model
        })

        log.p("Setting synapse weights with a localizing initializer")
        ae1.encoderSynapse.setWeights(cvt((c: Coordinate) ⇒ Util.R.get.nextGaussian * 0.001))

        log.eval {
          val stiffness = 3
          val radius = 0.5
          val peak = 0.001
          ae1.encoderSynapse.setWeights2(cvt((in: Coordinate, out: Coordinate) ⇒ {
            val doubleCoords = (0 until in.coords.length).map(d⇒{
              val from = in.coords(d) * 1.0 / ae1.encoderSynapse.inputDims(d)
              val to = out.coords(d) * 1.0 / ae1.encoderSynapse.outputDims(d)
              from - to
            }).toArray
            val dist = Math.sqrt(doubleCoords.map(x⇒x*x).sum)
            val factor = (1 + Math.tanh(stiffness * (radius - dist))) / 2
            peak * factor
          }))
        }

        if(ae1.decoderSynapse.isInstanceOf[TransposedSynapseLayer]) ae1.decoderSynapse = ae1.decoderSynapse.asInstanceOf[MappedSynapseLayer].asNewSynapseLayer()
        reportMatrix(log, smallDataset2, network.encoder, network.decoder)
        def trainingNetwork = new SimpleLossNetwork(network, new MeanSqLossLayer())
        schedule = List(
          TrainingStep(100, 10, 3000.0,
            new LBFGS().setMinHistory(5).setMaxHistory(35),
            new ArmijoWolfeConditions().setC2(0.9).setAlpha(1e-4)
          )
        )
        train(log, trainingNetwork, smallDataset1)
        summarizeHistory(log)
        IO.writeKryo(trainingNetwork, log.newFile(MarkdownReporter.currentMethod + "1.kryo.gz"))
        reportTable(log, smallDataset1, network.encoder, network.decoder)
        reportTable(log, smallDataset2, network.encoder, network.decoder)
        reportMatrix(log, smallDataset2, network.encoder, network.decoder)

        if(ae1.decoderSynapse.isInstanceOf[TransposedSynapseLayer]) ae1.decoderSynapse = ae1.decoderSynapse.asInstanceOf[MappedSynapseLayer].asNewSynapseLayer()
        network = new AutoencoderNetwork({
          var model: PipelineNetwork = new PipelineNetwork
          model.add(ae1.inputNoise)
          model.add(ae1.encoderSynapse)
          model.add(ae1.encoderBias)
          model.add(ae1.encoderActivation)
          model.add(ae1.encodedNoise)
          model
        }, {
          var model: PipelineNetwork = new PipelineNetwork
          model.add(ae1.decoderSynapse)
          model.add(ae1.decoderBias)
          model.add(ae1.decoderActivation)
          model
        })

        schedule = List(
          TrainingStep(1000, 10, 0.0,
            new LBFGS().setMinHistory(5).setMaxHistory(35),
            new ArmijoWolfeConditions().setC2(0.9).setAlpha(1e-4)
          )
        )
        train(log, trainingNetwork, data)
        summarizeHistory(log)
        IO.writeKryo(trainingNetwork, log.newFile(MarkdownReporter.currentMethod + "2.kryo.gz"))
        reportTable(log, smallDataset1, network.encoder, network.decoder)
        reportTable(log, smallDataset2, network.encoder, network.decoder)
        reportMatrix(log, smallDataset2, network.encoder, network.decoder)

      })
    }


  }


  private def reportMatrix(log: ScalaMarkdownPrintStream, data: Array[Array[Tensor]], encoder: DAGNode, decoder: DAGNode) = {
    val inputPrototype = data.head.head
    val encoded = encoder.getLayer.eval(inputPrototype).data.head
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

  private def reportTable(log: ScalaMarkdownPrintStream, data: Array[Array[Tensor]], encoder: DAGNode, decoder: DAGNode) = {
    log.eval {
      TableOutput.create(data.take(20).map(testObj ⇒ {
        var evalModel: PipelineNetwork = new PipelineNetwork
        evalModel.add(encoder.getLayer)
        evalModel.add(decoder.getLayer)
        val result = evalModel.eval(testObj(0)).data.head
        Map[String, AnyRef](
          "Input" → log.image(testObj(0).toGrayImage(), "Input"),
          "Output" → log.image(result.toGrayImage(), "Autoencoder Output")
        ).asJava
      }): _*)
    }
  }

  private def train[T <: SupervisedNetwork](log: ScalaMarkdownPrintStream, currentNetwork: T, data: Array[Array[Tensor]]) = {
    schedule.foreach(scheduledStep ⇒ {
      log.h3(scheduledStep.toString)
      log.eval {
        val trainable = new StochasticArrayTrainable(data, currentNetwork, scheduledStep.sampleSize)
        val normalized = new L12Normalizer(trainable).setFactor_L1(l1normalization).setFactor_L2(0.0)
        val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(normalized)
        trainer.setOrientation(scheduledStep.orient)
        trainer.setScaling(scheduledStep.step)
        trainer.setMonitor(monitor)
        trainer.setTimeout(scheduledStep.timeoutMinutes, TimeUnit.MINUTES)
        trainer.setTerminateThreshold(scheduledStep.endFitness)
        trainer.run()
      }
    })
  }

  private def summarizeHistory(log: ScalaMarkdownPrintStream) = {
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

}