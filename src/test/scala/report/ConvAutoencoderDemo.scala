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

import java.awt.Color
import java.lang
import java.util.concurrent.TimeUnit
import javax.imageio.ImageIO

import com.simiacryptus.mindseye.network._
import com.simiacryptus.mindseye.layers.NNLayer
import com.simiacryptus.mindseye.layers.activation.SoftmaxActivationLayer
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.trainable.StochasticArrayTrainable
import com.simiacryptus.util.io.IOUtil
import com.simiacryptus.util.ml.{Coordinate, Tensor}
import com.simiacryptus.util.test.{ImageTiles, MNIST}
import com.simiacryptus.util.text.TableOutput
import de.javakaffee.kryoserializers.KryoReflectionFactorySupport
import org.scalatest.{MustMatchers, WordSpec}
import smile.plot.{PlotCanvas, ScatterPlot}
import util.Java8Util._
import util.{ReportNotebook, ScalaNotebookOutput}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.Random

class ConvAutoencoderDemo extends WordSpec with MustMatchers with ReportNotebook {

  var data: Array[Tensor] = null
  val history = new mutable.ArrayBuffer[com.simiacryptus.mindseye.opt.IterativeTrainer.Step]
  var monitor = new TrainingMonitor {
    override def log(msg: String): Unit = {
      System.err.println(msg)
    }

    override def onStepComplete(currentPoint: IterativeTrainer.Step): Unit = {
      history += currentPoint
    }
  }
  val minutesPerStep = 5

  "Train Autoencoder Network" should {

    "MNIST" in {
      Thread.sleep(15*1000)
      report("mnist", log ⇒ {
        data = log.eval {
          MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ labeledObj.data).toArray
        }
        preview(log, 10, 10)

        val autoencoder = log.eval {
          new ConvAutoencoderNetwork.RecursiveBuilder(data) {
            override protected def configure(builder: ConvAutoencoderNetwork.Builder): ConvAutoencoderNetwork.Builder = {
              super.configure(builder
                .setNoise(0.1)
                .setDropout(0.1)
              )
            }

            override protected def configure(trainingParameters: ConvAutoencoderNetwork.TrainingParameters): ConvAutoencoderNetwork.TrainingParameters = {
              super.configure(trainingParameters
                .setMonitor(monitor)
                .setSampleSize(100)
                .setTimeoutMinutes(minutesPerStep)
              )
            }
          }
        }
        history.clear()
        log.eval {
          autoencoder.growLayer(14, 14, 4)
        }
        summarizeHistory(log)
        reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
        reportMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder)
        IOUtil.writeKryo(autoencoder.getEncoder, log.file(ReportNotebook.currentMethod + ".encoder.1.kryo.gz"))
        IOUtil.writeKryo(autoencoder.getDecoder, log.file(ReportNotebook.currentMethod + ".decoder.1.kryo.gz"))

        log.eval {
          autoencoder.growLayer(7, 7, 4)
        }
        summarizeHistory(log)
        reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
        reportMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder)
        IOUtil.writeKryo(autoencoder.getEncoder, log.file(ReportNotebook.currentMethod + ".encoder.2.kryo.gz"))
        IOUtil.writeKryo(autoencoder.getDecoder, log.file(ReportNotebook.currentMethod + ".decoder.2.kryo.gz"))

        log.eval {
          autoencoder.tune()
        }
        summarizeHistory(log)
        reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
        reportMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder)
        IOUtil.writeKryo(autoencoder.getEncoder, log.file(ReportNotebook.currentMethod + ".encoder.3.kryo.gz"))
        IOUtil.writeKryo(autoencoder.getDecoder, log.file(ReportNotebook.currentMethod + ".decoder.3.kryo.gz"))


        val trainingData: Seq[Array[Tensor]] = MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
          Array(labeledObj.data, toOutNDArray(toOut(labeledObj.label), 10))
        }).toList
        val categorizationAdapter = new DenseSynapseLayer(Array[Int](5, 5, 1), Array[Int](10))
        categorizationAdapter.setWeights(cvt((c:Coordinate)⇒Random.nextGaussian() * 0.001))
        var categorizationNetwork = log.eval {
          val kryo = new KryoReflectionFactorySupport()
          val categorizationNetwork = new PipelineNetwork()
          categorizationNetwork.add(kryo.copy(autoencoder.getEncoder).freeze())
          categorizationNetwork.add(categorizationAdapter)
          categorizationNetwork.add(new SoftmaxActivationLayer)
          val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(categorizationNetwork, new EntropyLossLayer)
          val trainable = new StochasticArrayTrainable(trainingData.toArray, trainingNetwork, 100)
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new LBFGS())
          trainer.setMonitor(monitor)
          trainer.setTimeout(minutesPerStep, TimeUnit.MINUTES)
          trainer.setTerminateThreshold(1.0)
          trainer.run()
          categorizationNetwork
        }
        mnistClassificationReport(log, categorizationNetwork)
        categorizationNetwork = log.eval {
          val kryo = new KryoReflectionFactorySupport()
          val categorizationNetwork = new PipelineNetwork()
          categorizationNetwork.add(kryo.copy(autoencoder.getEncoder))
          categorizationNetwork.add(categorizationAdapter)
          categorizationNetwork.add(new SoftmaxActivationLayer)
          val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(categorizationNetwork, new EntropyLossLayer)
          val trainable = new StochasticArrayTrainable(trainingData.toArray, trainingNetwork, 100)
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new LBFGS())
          trainer.setMonitor(monitor)
          trainer.setTimeout(minutesPerStep, TimeUnit.MINUTES)
          trainer.setTerminateThreshold(1.0)
          trainer.run()
          categorizationNetwork
        }
        mnistClassificationReport(log, categorizationNetwork)

      })
    }

    "Monkey" in {
      report("monkey", log ⇒ {
        data = log.eval {
          var data = ImageTiles.tilesRgb(ImageIO.read(getClass.getClassLoader.getResourceAsStream("monkey1.jpg")), 10, 10, 10, 10)
          data = Random.shuffle(data.toList).toArray
          data
        }
        preview(log, 100, 60)

        val autoencoder = log.eval {
          new ConvAutoencoderNetwork.RecursiveBuilder(data) {
            override protected def configure(builder: ConvAutoencoderNetwork.Builder): ConvAutoencoderNetwork.Builder = {
              super.configure(builder
                .setNoise(0.01)
                .setDropout(0.01)
              )
            }

            override protected def configure(trainingParameters: ConvAutoencoderNetwork.TrainingParameters): ConvAutoencoderNetwork.TrainingParameters = {
              super.configure(trainingParameters
                .setMonitor(monitor)
                .setSampleSize(1000)
                .setTimeoutMinutes(minutesPerStep)
              )
            }
          }
        }

        history.clear()
        log.eval {
          autoencoder.growLayer(5, 5, 3)
        }
        summarizeHistory(log)
        reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
        (0 until 3).foreach(band ⇒ reportMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder, band))
        IOUtil.writeKryo(autoencoder.getEncoder, log.file(ReportNotebook.currentMethod + ".encoder.1.kryo.gz"))
        IOUtil.writeKryo(autoencoder.getDecoder, log.file(ReportNotebook.currentMethod + ".decoder.1.kryo.gz"))

        history.clear()
        log.eval {
          autoencoder.growLayer(3, 3, 3)
        }
        summarizeHistory(log)
        reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
        (0 until 3).foreach(band ⇒ reportMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder, band))
        IOUtil.writeKryo(autoencoder.getEncoder, log.file(ReportNotebook.currentMethod + ".encoder.2.kryo.gz"))
        IOUtil.writeKryo(autoencoder.getDecoder, log.file(ReportNotebook.currentMethod + ".decoder.2.kryo.gz"))

        history.clear()
        log.eval {
          autoencoder.tune()
        }
        summarizeHistory(log)
        reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
        (0 until 3).foreach(band ⇒ reportMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder, band))
        IOUtil.writeKryo(autoencoder.getEncoder, log.file(ReportNotebook.currentMethod + ".encoder.3.kryo.gz"))
        IOUtil.writeKryo(autoencoder.getDecoder, log.file(ReportNotebook.currentMethod + ".decoder.3.kryo.gz"))
      })
    }


  }


  private def mnistClassificationReport(log: ScalaNotebookOutput, categorizationNetwork : PipelineNetwork) = {
    log.eval {
      log.p("The (mis)categorization matrix displays a count matrix for every actual/predicted category: ")
      val categorizationMatrix: Map[Int, Map[Int, Int]] = log.eval {
        MNIST.validationDataStream().iterator().asScala.toStream.map(testObj ⇒ {
          val result = categorizationNetwork.eval(testObj.data).data.head
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
  }

  private def reportMatrix(log: ScalaNotebookOutput, encoder: NNLayer, decoder: NNLayer, band: Int = 0) = {
    val inputPrototype = data.head
    val dims = inputPrototype.getDims()
    val encoded = encoder.eval(inputPrototype).data.head
    val width = encoded.getDims()(0)
    val height = encoded.getDims()(1)
    log.draw(gfx ⇒ {
      (0 until width).foreach(x ⇒ {
        (0 until height).foreach(y ⇒ {
          encoded.fill(cvt((i: Int) ⇒ 0.0))
          encoded.set(Array(x, y, band), 1.0)
          val tensor = decoder.eval(encoded).data.head
          val sum = tensor.getData.sum
          val min = tensor.getData.min
          val max = tensor.getData.max
          var getPixel: (Int, Int) ⇒ Color = null
          val dims = tensor.getDims
          if (3 == dims.length) {
            if (3 == dims(2)) {
              getPixel = (xx: Int, yy: Int) ⇒ {
                val red: Double = 255 * (tensor.get(xx, yy, 0) - min) / (max - min)
                val blue: Double = 255 * (tensor.get(xx, yy, 1) - min) / (max - min)
                val green: Double = 255 * (tensor.get(xx, yy, 2) - min) / (max - min)
                new Color(red.toInt, blue.toInt, green.toInt)
              }
            } else {
              assert(1 == dims(2))
              getPixel = (xx: Int, yy: Int) ⇒ {
                val value: Double = 255 * (tensor.get(xx, yy) - min) / (max - min)
                new Color(value.toInt, value.toInt, value.toInt)
              }
            }
          } else {
            assert(2 == dims.length)
            getPixel = (xx: Int, yy: Int) ⇒ {
              val value: Double = 255 * (tensor.get(xx, yy) - min) / (max - min)
              new Color(value.toInt, value.toInt, value.toInt)
            }
          }
          (0 until dims(0)).foreach(xx ⇒
            (0 until dims(1)).foreach(yy ⇒ {
              gfx.setColor(getPixel(xx, yy))
              gfx.drawRect((x * dims(0)) + xx, (y * dims(1)) + yy, 1, 1)
            }))
        })
      })
    }, width = dims(0) * width, height = dims(1) * height)
  }

  private def preview(log: ScalaNotebookOutput, width: Int, height: Int) = {
    val inputPrototype = data.head
    val dims = inputPrototype.getDims
    log.draw(gfx ⇒ {
      (0 until width).foreach(x ⇒ {
        (0 until height).foreach(y ⇒ {
          val tensor = data((y * width + x) % data.length)
          val min = 0 // tensor.getData.min
          val max = 255 // tensor.getData.max
          var getPixel: (Int, Int) ⇒ Color = null
          if (3 == dims.length) {
            if (3 == dims(2)) {
              getPixel = (xx: Int, yy: Int) ⇒ {
                val red: Double = 255 * (tensor.get(xx, yy, 0) - min) / (max - min)
                val green: Double = 255 * (tensor.get(xx, yy, 1) - min) / (max - min)
                val blue: Double = 255 * (tensor.get(xx, yy, 2) - min) / (max - min)
                new Color(red.toInt, green.toInt, blue.toInt)
              }
            } else {
              assert(1 == dims(2))
              getPixel = (xx: Int, yy: Int) ⇒ {
                val value: Double = 255 * (tensor.get(xx, yy) - min) / (max - min)
                new Color(value.toInt, value.toInt, value.toInt)
              }
            }
          } else {
            assert(2 == dims.length)
            getPixel = (xx: Int, yy: Int) ⇒ {
              val value: Double = 255 * (tensor.get(xx, yy) - min) / (max - min)
              new Color(value.toInt, value.toInt, value.toInt)
            }
          }
          (0 until dims(0)).foreach(xx ⇒
            (0 until dims(1)).foreach(yy ⇒ {
              gfx.setColor(getPixel(xx, yy))
              gfx.drawRect((x * dims(0)) + xx, (y * dims(1)) + yy, 1, 1)
            }))
        })
      })
    }, width = dims(0) * width, height = dims(1) * height)
  }

  private def reportTable(log: ScalaNotebookOutput, encoder: NNLayer, decoder: NNLayer) = {
    log.eval {
      TableOutput.create(data.take(20).map(testObj ⇒ {
        var evalModel: PipelineNetwork = new PipelineNetwork
        evalModel.add(encoder)
        evalModel.add(decoder)
        val result = evalModel.eval(testObj).data.head
        Map[String, AnyRef](
          "Input" → log.image(testObj.toImage(), "Input"),
          "Output" → log.image(result.toImage(), "Autoencoder Output")
        ).asJava
      }): _*)
    }
  }

  private def summarizeHistory(log: ScalaNotebookOutput) = {
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