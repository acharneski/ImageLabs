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

import java.awt.Color
import java.io.PrintStream
import java.lang
import java.util.concurrent.{Semaphore, TimeUnit}

import com.simiacryptus.mindseye.data.MNIST
import com.simiacryptus.mindseye.eval.StochasticArrayTrainable
import com.simiacryptus.mindseye.lang._
import com.simiacryptus.mindseye.layers.java.{DenseSynapseLayer, EntropyLossLayer, SoftmaxActivationLayer}
import com.simiacryptus.mindseye.network.{AutoencoderNetwork, PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt.orient.LBFGS
import com.simiacryptus.mindseye.opt.{Step, TrainingMonitor}
import com.simiacryptus.text.TableOutput
import com.simiacryptus.util.StreamNanoHTTPD
import com.simiacryptus.util.io.{HtmlNotebookOutput, KryoUtil, TeeOutputStream}
import fi.iki.elonen.NanoHTTPD
import fi.iki.elonen.NanoHTTPD.IHTTPSession
import smile.plot.{PlotCanvas, ScatterPlot}
import util.Java8Util.cvt
import util._

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.Random

object MnistAutoencoder extends Report {

  def main(args: Array[String]): Unit = {
    report((server,log)⇒new MnistAutoencoder(server,log).run())
    System.exit(0)
  }
}

private class MnistAutoencoder(server: StreamNanoHTTPD, log: HtmlNotebookOutput with ScalaNotebookOutput) {
  def kryo = KryoUtil.kryo()
  val originalStdOut = System.out

  var data: Array[Tensor] = null
  val history = new mutable.ArrayBuffer[com.simiacryptus.mindseye.opt.Step]
  val minutesPerStep = 10
  val logOut = new TeeOutputStream(log.file("log.txt"), true)
  val monitor = new TrainingMonitor {
    val logPrintStream = new PrintStream(logOut)
    override def log(msg: String): Unit = {
      //if(!msg.trim.isEmpty) {}
      logPrintStream.println(msg)
      originalStdOut.println(msg)
    }

    override def onStepComplete(currentPoint: Step): Unit = {
      history += currentPoint
    }
  }

  def run(): Unit = {

    log.p("View the convergence history: <a href='/history.html'>/history.html</a>")
    server.addAsyncHandler("history.html", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        summarizeHistory(log, history.toList.toArray)
      })
    }), false)

    log.p("View the log: <a href='/log'>/log</a>")
    server.addSessionHandler("log", Java8Util.cvt((session : IHTTPSession)⇒{
      NanoHTTPD.newChunkedResponse(NanoHTTPD.Response.Status.OK, "text/plain", logOut.newInputStream())
    }))
    log.out("<hr/>");

    data = log.eval {
      MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ labeledObj.data).toArray
    }

    preview(log, 10, 10)

    var noise = 10.0
    var dropout = 0.5
    val autoencoder = log.eval {
      new AutoencoderNetwork.RecursiveBuilder(new TensorArray(data:_*)) {
        override protected def configure(builder: AutoencoderNetwork.Builder): AutoencoderNetwork.Builder = {
          super.configure(builder
            .setNoise(noise)
            .setDropout(dropout)
          )
        }

        override protected def configure(trainingParameters: AutoencoderNetwork.TrainingParameters): AutoencoderNetwork.TrainingParameters = {
          super.configure(trainingParameters
            .setMonitor(monitor)
            .setSampleSize(1000)
            .setTimeoutMinutes(minutesPerStep)
          )
        }
      }
    }


    log.p("<a href='/reportTable.html'>Autorecognition Sample</a>")
    server.addAsyncHandler("reportTable.html", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
      })
    }), false)

    log.p("<a href='/representationMatrix.html'>Representation Matrix</a>")
    server.addAsyncHandler("representationMatrix.html", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        representationMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder)
      })
    }), false)


    log.eval {
      autoencoder.growLayer(20, 20, 1)
    }
    summarizeHistory(log, history.toArray)
    reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
    representationMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder)

    log.eval {
      noise = 0
      autoencoder.growLayer(10, 10, 1)
    }
    summarizeHistory(log, history.toArray)
    reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
    representationMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder)

    log.eval {
      autoencoder.growLayer(5, 5, 1)
    }
    summarizeHistory(log, history.toArray)
    reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
    representationMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder)

    log.eval {
      autoencoder.tune()
    }
    summarizeHistory(log, history.toArray)
    reportTable(log, autoencoder.getEncoder, autoencoder.getDecoder)
    representationMatrix(log, autoencoder.getEncoder, autoencoder.getDecoder)


    val trainingData: Seq[Array[Tensor]] = MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
      Array(labeledObj.data, toOutNDArray(toOut(labeledObj.label), 10))
    }).toList
    val categorizationAdapter = new DenseSynapseLayer(Array[Int](5, 5, 1), Array[Int](10))
    categorizationAdapter.setWeights(cvt((c:Coordinate)⇒Random.nextGaussian() * 0.001))
    var categorizationNetwork = log.eval {
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

    logOut.close()
    val onExit = new Semaphore(0)
    log.p("To exit the sever: <a href='/exit'>/exit</a>")
    server.addAsyncHandler("exit", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        log.h1("OK")
        onExit.release(1)
      })
    }), false)
    onExit.acquire()
  }

  private def mnistClassificationReport(log: ScalaNotebookOutput, categorizationNetwork : PipelineNetwork) = {
    log.eval {
      log.p("The (mis)categorization matrix displays a count matrix for every actual/predicted category: ")
      val categorizationMatrix: Map[Int, Map[Int, Int]] = log.eval {
        MNIST.validationDataStream().iterator().asScala.toStream.map(testObj ⇒ {
          val result = categorizationNetwork.eval(new NNExecutionContext() {}, testObj.data).getData.get(0)
          val prediction: Int = (0 to 9).maxBy(i ⇒ result.get(i))
          val actual: Int = toOut(testObj.label)
          actual → prediction
        }).groupBy(_._1).mapValues(_.groupBy(_._2).mapValues(_.size))
      }
      writeMislassificationMatrix(log.asInstanceOf[HtmlNotebookOutput], categorizationMatrix)
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

  private def representationMatrix(log: ScalaNotebookOutput, encoder: NNLayer, decoder: NNLayer, band: Int = 0, probeIntensity : Double = 255.0) = {
    val inputPrototype = data.head
    val dims = inputPrototype.getDimensions()
    val encoded: Tensor = encoder.eval(new NNExecutionContext() {}, inputPrototype).getData.get(0)
    val width = encoded.getDimensions()(0)
    val height = encoded.getDimensions()(1)
    log.draw(gfx ⇒ {
      (0 until width).foreach(x ⇒ {
        (0 until height).foreach(y ⇒ {
          encoded.fill(cvt((i: Int) ⇒ 0.0))
          encoded.set(Array(x, y, band), probeIntensity)
          val tensor: Tensor = decoder.eval(new NNExecutionContext() {}, encoded).getData.get(0)
          val min: Double = tensor.getData.min
          val max: Double = tensor.getData.max
          if(min != max) {
            var getPixel: (Int, Int) ⇒ Color = null
            val dims = tensor.getDimensions
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
                  val value: Double = 255 * (tensor.get(xx, yy, 0) - min) / (max - min)
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
                val pixel = getPixel(xx, yy)
                gfx.setColor(pixel)
                gfx.drawRect((x * dims(0)) + xx, (y * dims(1)) + yy, 1, 1)
              }))
          }
        })
      })
    }, width = dims(0) * width, height = dims(1) * height)
  }

  private def preview(log: ScalaNotebookOutput, width: Int, height: Int) = {
    val inputPrototype = data.head
    val dims = inputPrototype.getDimensions
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
        val result = evalModel.eval(new NNExecutionContext() {}, testObj).getData.get(0)
        Map[String, AnyRef](
          "Input" → log.image(testObj.toImage(), "Input"),
          "Output" → log.image(result.toImage(), "Autoencoder Output")
        ).asJava
      }): _*)
    }
  }


  private def writeMislassificationMatrix(log: HtmlNotebookOutput, categorizationMatrix: Map[Int, Map[Int, Int]]) = {
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

  private def toOut(label: String): Int = {
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

  private def toOutNDArray(out: Int, max: Int): Tensor = {
    val ndArray = new Tensor(max)
    ndArray.set(out, 1)
    ndArray
  }

  private def summarizeHistory(log: ScalaNotebookOutput, history: Array[com.simiacryptus.mindseye.opt.Step]) = {
    if(!history.isEmpty) {
      log.eval {
        val plot: PlotCanvas = ScatterPlot.plot(history.map(item ⇒ Array[Double](
          item.iteration, Math.log(item.point.sum)
        )).toArray: _*)
        plot.setTitle("Convergence Plot")
        plot.setAxisLabels("Iteration", "log(Fitness)")
        plot.setSize(600, 400)
        plot
      }
      log.eval {
        val step = Math.max(Math.pow(10, Math.ceil(Math.log(history.size) / Math.log(10)) - 2), 1).toInt
        TableOutput.create(history.filter(0 == _.iteration % step).map(state ⇒
          Map[String, AnyRef](
            "iteration" → state.iteration.toInt.asInstanceOf[Integer],
            "time" → state.time.toDouble.asInstanceOf[lang.Double],
            "fitness" → state.point.sum.toDouble.asInstanceOf[lang.Double]
          ).asJava
        ): _*)
      }
    }
  }

}