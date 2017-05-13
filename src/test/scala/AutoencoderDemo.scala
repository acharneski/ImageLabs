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
import javax.imageio.ImageIO

import AutoencoderUtil._
import com.simiacryptus.mindseye.graph._
import com.simiacryptus.mindseye.graph.dag._
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.util.ml.Tensor
import com.simiacryptus.util.test.MNIST
import com.simiacryptus.util.text.TableOutput
import com.simiacryptus.util.{IO, ImageTiles}
import org.scalatest.{MustMatchers, WordSpec}
import smile.plot.{PlotCanvas, ScatterPlot}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.Random

class AutoencoderDemo extends WordSpec with MustMatchers with MarkdownReporter {

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

  "Train Autoencoder Network" should {

    "MNIST" in {
      report("mnist", log ⇒ {
        data = log.eval {
          MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ labeledObj.data).toArray
        }
        preview(log, 10, 10)

        val autoencoder = log.eval {
          new AutoencoderNetwork.RecursiveBuilder(data) {
            override protected def configure(builder: AutoencoderNetwork.Builder): AutoencoderNetwork.Builder = {
              super.configure(builder
                .setNoise(0.1)
                .setDropout(0.1)
              )
            }

            override protected def configure(trainingParameters: AutoencoderNetwork.TrainingParameters): AutoencoderNetwork.TrainingParameters = {
              super.configure(trainingParameters
                .setMonitor(monitor)
                .setSampleSize(100)
                .setTimeoutMinutes(10)
              )
            }
          }
        }
        history.clear()
        log.eval {
          autoencoder.growLayer(10, 10, 1)
        }
        summarizeHistory(log)
        reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
        reportMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder)
        IO.writeKryo(autoencoder.getEncoder, log.newFile(MarkdownReporter.currentMethod + ".encoder.1.kryo.gz"))
        IO.writeKryo(autoencoder.getDecoder, log.newFile(MarkdownReporter.currentMethod + ".decoder.1.kryo.gz"))

        log.eval {
          autoencoder.growLayer(5, 5, 1)
        }
        summarizeHistory(log)
        reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
        reportMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder)
        IO.writeKryo(autoencoder.getEncoder, log.newFile(MarkdownReporter.currentMethod + ".encoder.2.kryo.gz"))
        IO.writeKryo(autoencoder.getDecoder, log.newFile(MarkdownReporter.currentMethod + ".decoder.2.kryo.gz"))

        log.eval {
          autoencoder.tune()
        }
        summarizeHistory(log)
        reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
        reportMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder)
        IO.writeKryo(autoencoder.getEncoder, log.newFile(MarkdownReporter.currentMethod + ".encoder.3.kryo.gz"))
        IO.writeKryo(autoencoder.getDecoder, log.newFile(MarkdownReporter.currentMethod + ".decoder.3.kryo.gz"))
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
          new AutoencoderNetwork.RecursiveBuilder(data) {
            override protected def configure(builder: AutoencoderNetwork.Builder): AutoencoderNetwork.Builder = {
              super.configure(builder
                .setNoise(0.01)
                .setDropout(0.01)
              )
            }

            override protected def configure(trainingParameters: AutoencoderNetwork.TrainingParameters): AutoencoderNetwork.TrainingParameters = {
              super.configure(trainingParameters
                .setMonitor(monitor)
                .setSampleSize(1000)
                .setTimeoutMinutes(10)
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
        reportMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder)
        IO.writeKryo(autoencoder.getEncoder, log.newFile(MarkdownReporter.currentMethod + ".encoder.1.kryo.gz"))
        IO.writeKryo(autoencoder.getDecoder, log.newFile(MarkdownReporter.currentMethod + ".decoder.1.kryo.gz"))

        history.clear()
        log.eval {
          autoencoder.growLayer(3, 3, 3)
        }
        summarizeHistory(log)
        reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
        reportMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder)
        IO.writeKryo(autoencoder.getEncoder, log.newFile(MarkdownReporter.currentMethod + ".encoder.2.kryo.gz"))
        IO.writeKryo(autoencoder.getDecoder, log.newFile(MarkdownReporter.currentMethod + ".decoder.2.kryo.gz"))

        history.clear()
        log.eval {
          autoencoder.tune()
        }
        summarizeHistory(log)
        reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
        reportMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder)
        IO.writeKryo(autoencoder.getEncoder, log.newFile(MarkdownReporter.currentMethod + ".encoder.3.kryo.gz"))
        IO.writeKryo(autoencoder.getDecoder, log.newFile(MarkdownReporter.currentMethod + ".decoder.3.kryo.gz"))
      })
    }


  }


  private def reportMatrix(log: ScalaMarkdownPrintStream, encoder: DAGNode, decoder: DAGNode) = {
    val inputPrototype = data.head
    val dims = inputPrototype.getDims()
    val encoded = encoder.getLayer.eval(inputPrototype).data.head
    val width = encoded.getDims()(0)
    val height = encoded.getDims()(1)
    log.draw(gfx ⇒ {
      (0 until width).foreach(x ⇒ {
        (0 until height).foreach(y ⇒ {
          encoded.fill(cvt((i: Int) ⇒ 0.0))
          encoded.set(Array(x, y), 1.0)
          val tensor = decoder.getLayer.eval(encoded).data.head
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

  private def preview(log: ScalaMarkdownPrintStream, width: Int, height: Int) = {
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

  private def reportTable(log: ScalaMarkdownPrintStream, encoder: DAGNode, decoder: DAGNode) = {
    log.eval {
      TableOutput.create(data.take(20).map(testObj ⇒ {
        var evalModel: PipelineNetwork = new PipelineNetwork
        evalModel.add(encoder.getLayer)
        evalModel.add(decoder.getLayer)
        val result = evalModel.eval(testObj).data.head
        Map[String, AnyRef](
          "Input" → log.image(testObj.toImage(), "Input"),
          "Output" → log.image(result.toImage(), "Autoencoder Output")
        ).asJava
      }): _*)
    }
  }

  private def summarizeHistory(log: ScalaMarkdownPrintStream) = {
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

}