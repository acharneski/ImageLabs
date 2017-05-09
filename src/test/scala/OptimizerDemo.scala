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

import java.lang
import java.util.concurrent.TimeUnit
import java.util.function.{IntToDoubleFunction, ToDoubleFunction}

import AutoencoderDemo.cvt
import com.simiacryptus.mindseye.net._
import com.simiacryptus.mindseye.net.activation.{ReLuActivationLayer, SoftmaxActivationLayer}
import com.simiacryptus.mindseye.net.basic.BiasLayer
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerJBLAS
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.util.Util
import com.simiacryptus.util.ml.{Coordinate, Tensor}
import com.simiacryptus.util.test.MNIST
import com.simiacryptus.util.text.TableOutput
import org.scalatest.{MustMatchers, WordSpec}
import smile.plot.{PlotCanvas, ScatterPlot}

import scala.collection.JavaConverters._

object OptimizerDemo {

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

class OptimizerDemo extends WordSpec with MustMatchers with MarkdownReporter {

  case class TrainingStep(sampleSize: Int, timeoutMinutes: Int)
  val schedule = List(
    TrainingStep(200, minutesPerPhase),
    TrainingStep(1000, minutesPerPhase),
    TrainingStep(10000, minutesPerPhase)
  )
  val terminationThreshold = 0.01
  val inputSize = Array[Int](28, 28, 1)
  val outputSize = Array[Int](10)
  var minutesPerPhase = 5

  "Various Optimization Strategies" should {

    "Neutral" in {
      report("neutral", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new LBFGS())
          trainer
        })
      })
    }

    "Normalized L1 a" in {
      report("normal_l1_a", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new L12Normalized(new LBFGS()).setFactor_L1(0.01))
          trainer
        })
      })
    }

    "Normalized L1 b" in {
      report("normal_l1_b", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new L12Normalized(new LBFGS()) {
            override def directionFilter(unitDotProduct: Double): Double = {
              if(unitDotProduct>0) 0
              else unitDotProduct
            }
          }.setFactor_L1(0.01))
          trainer
        })
      })
    }

    "Normalized L1 c" in {
      report("normal_l1_c", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new L12NormalizedConst(new LBFGS()).setFactor_L1(0.01))
          trainer
        })
      })
    }

    "Normalized L1 a neg" in {
      report("normal_l1_a_neg", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new L12Normalized(new LBFGS()).setFactor_L1(-0.01))
          trainer
        })
      })
    }

    "Normalized L1 b neg" in {
      report("normal_l1_b_neg", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new L12Normalized(new LBFGS()) {
            override def directionFilter(unitDotProduct: Double): Double = {
              if(unitDotProduct>0) 0
              else unitDotProduct
            }
          }.setFactor_L1(-0.01))
          trainer
        })
      })
    }

    "Normalized L1 c neg" in {
      report("normal_l1_c_neg", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new L12NormalizedConst(new LBFGS()).setFactor_L1(-0.01))
          trainer
        })
      })
    }

    "Normalized L2 a" in {
      report("normal_l2_a", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new L12Normalized(new LBFGS()).setFactor_L1(0.0).setFactor_L2(0.01))
          trainer
        })
      })
    }

    "Normalized L2 b" in {
      report("normal_l2_b", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new L12Normalized(new LBFGS()) {
            override def directionFilter(unitDotProduct: Double): Double = {
              if(unitDotProduct>0) 0
              else unitDotProduct
            }
          }.setFactor_L1(0.0).setFactor_L2(0.01))
          trainer
        })
      })
    }

    "Normalized L2 c" in {
      report("normal_l2_c", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new L12NormalizedConst(new LBFGS()).setFactor_L1(0.0).setFactor_L2(0.01))
          trainer
        })
      })
    }

    "Normalized L2 a neg" in {
      report("normal_l2_a_neg", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new L12Normalized(new LBFGS()).setFactor_L1(0.0).setFactor_L2(-0.01))
          trainer
        })
      })
    }

    "Normalized L2 b neg" in {
      report("normal_l2_b_neg", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new L12Normalized(new LBFGS()) {
            override def directionFilter(unitDotProduct: Double): Double = {
              if(unitDotProduct>0) 0
              else unitDotProduct
            }
          }.setFactor_L1(0.0).setFactor_L2(-0.01))
          trainer
        })
      })
    }

    "Normalized L2 c neg" in {
      report("normal_l2_c_neg", log ⇒ {
        test(log, trainable ⇒ log.eval {
          val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
          trainer.setOrientation(new L12NormalizedConst(new LBFGS()).setFactor_L1(0.0).setFactor_L2(-0.01))
          trainer
        })
      })
    }

  }

  def test(log: ScalaMarkdownPrintStream, optimizer: Trainable⇒IterativeTrainer) = {
    val model = log.eval {
      val middleSize = Array[Int](28, 28, 1)
      var model: PipelineNetwork = new PipelineNetwork
      model.add(new DenseSynapseLayerJBLAS(inputSize, middleSize)
        .setWeights(cvt((c:Coordinate)⇒Util.R.get.nextGaussian * 0.001)))
      model.add(new BiasLayer(middleSize: _*))
      model.add(new ReLuActivationLayer().freeze)
      model.add(new DenseSynapseLayerJBLAS(middleSize, outputSize)
        .setWeights(cvt((c:Coordinate)⇒Util.R.get.nextGaussian * 0.001)))
      model.add(new BiasLayer(outputSize: _*))
      model.add(new SoftmaxActivationLayer)
      model
    }
    log.h2("Data")
    log.p("First, we load the training dataset: ")
    val trainingData: Seq[Array[Tensor]] = log.code(() ⇒ {
      MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
        Array(labeledObj.data, toOutNDArray(toOut(labeledObj.label), 10))
      }).toList
    })
    log.h2("Training")
    log.p("We encapsulate our model network within a supervisory network that applies a loss function, then train using a standard iterative L-BFGS strategy: ")
    val history = new scala.collection.mutable.ArrayBuffer[com.simiacryptus.mindseye.opt.IterativeTrainer.Step]()
    val monitor = log.eval {
      val monitor = new TrainingMonitor {
        override def log(msg: String): Unit = {
          System.err.println(msg)
        }

        override def onStepComplete(currentPoint: IterativeTrainer.Step): Unit = {
          history += currentPoint
        }
      }
      monitor
    }
    log.p("First we pretrain the model on a very small dataset until it is at a reasonable starting point")

    schedule.foreach(scheduledStep ⇒ {
      log.eval {
        System.out.println(scheduledStep)
        val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
        val trainable: StochasticArrayTrainable = new StochasticArrayTrainable(trainingData.toArray, trainingNetwork, scheduledStep.sampleSize)
        val trainer: IterativeTrainer = optimizer.apply(trainable)
        trainer.setMonitor(monitor)
        trainer.setTimeout(Math.min(scheduledStep.timeoutMinutes, 10), TimeUnit.MINUTES)
        trainer.setTerminateThreshold(1.0)
        trainer.run()
      }

      log.eval {
        val set = new DeltaSet()
        model.eval(new Tensor(inputSize:_*)).accumulate(set,Array(new Tensor(10)))
        set.map.asScala.map(ent⇒{
          val (layer, buffer) = ent
          val data = buffer.target
          val length = data.length
          val sum = data.sum
          val sumSq = data.map(x ⇒ x * x).sum
          val meanAbs = data.map(x ⇒ Math.abs(x)).sum / length
          Map(
            "layer" → layer.getClass.getSimpleName,
            "id" → layer.getId,
            "max" → data.max,
            "min" → data.min,
            "length" → length,
            "sum" → sum,
            "sumSq" → sumSq,
            "mean" → (sum / length),
            "meanAbs" → meanAbs,
            "stdDev" → Math.sqrt((sumSq / length) - Math.pow(sum / length, 2))
          )
        }).mkString("\n")
      }
    })
    log.p("After training, we have the following parameterized model: ")
    log.eval {
      model
    }
    log.p("A summary of the training timeline: ")
    summarizeHistory(log, history.toList)

    log.h2("Validation")

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