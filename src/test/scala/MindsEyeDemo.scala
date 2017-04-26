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
import java.util
import java.util.function.ToDoubleFunction

import com.simiacryptus.mindseye.Util
import com.simiacryptus.mindseye.core.TrainingContext
import com.simiacryptus.mindseye.net.DAGNetwork
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer
import com.simiacryptus.mindseye.net.basic.BiasLayer
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerJBLAS
import com.simiacryptus.mindseye.net.loss.EntropyLossLayer
import com.simiacryptus.mindseye.training.{DynamicRateTrainer, GradientDescentTrainer}
import com.simiacryptus.util.ml.{Coordinate, Tensor}
import com.simiacryptus.util.test.MNIST
import com.simiacryptus.util.text.TableOutput
import org.scalatest.{MustMatchers, WordSpec}

import scala.collection.JavaConverters._

class MindsEyeDemo extends WordSpec with MustMatchers with MarkdownReporter {


  "MindsEye Demo" should {

    "Access MNIST dataset" in {
      report("mnist_data", log ⇒ {
        val rows = 100
        val cols = 50
        val size = 28
        log.draw(gfx ⇒ {
          var n = 0
          MNIST.trainingDataStream().iterator().asScala.toStream.take(rows * cols).foreach(item ⇒ {
            val (x, y) = ((n % cols) * size, (n / cols) * size)
            (0 until size).foreach(xx ⇒
              (0 until size).foreach(yy ⇒ {
                val value: Double = item.data.get(xx, yy)
                gfx.setColor(new Color(value.toInt, value.toInt, value.toInt))
                gfx.drawRect(x + xx, y + yy, 1, 1)
              }))
            n = n + 1
          })
        }, width = size * cols, height = size * rows)
      })
    }

    "Train Simple Digit Recognizer" in {
      report("mnist_simple", log ⇒ {
        val inputSize = Array[Int](28, 28, 1)
        val outputSize = Array[Int](10)

        var model: DAGNetwork = log.code(()⇒{
          var model: DAGNetwork = new DAGNetwork
          model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize:_*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.0
          }))
          model = model.add(new BiasLayer(outputSize:_*))
          // model = model.add(new MinMaxFilterLayer());
          model = model.add(new SoftmaxActivationLayer)
          model
        })

        val data: Seq[Array[Tensor]] = log.code(()⇒{
          MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
            Array(labeledObj.data, toOutNDArray(toOut(labeledObj.label), 10))
          })
        })

        log.code(()⇒ {
          val previewTable = new TableOutput()
          data.take(10).map(testObj⇒{
            val row = new util.LinkedHashMap[String,AnyRef]()
            row.put("Input1 (as Image)",log.image(testObj(0).toGrayImage(), testObj(0).toString))
            row.put("Input2 (as String)",testObj(1).toString)
            row.put("Input1 (as String)",testObj(0).toString)
            row
          }).foreach(previewTable.putRow(_))
          previewTable
        })

        val trainer = log.code(()⇒{
          val trainingNetwork: DAGNetwork = new DAGNetwork
          trainingNetwork.add(model)
          trainingNetwork.addLossComponent(new EntropyLossLayer)
          val gradientTrainer: GradientDescentTrainer = new GradientDescentTrainer
          gradientTrainer.setNet(trainingNetwork)
          gradientTrainer.setData(data.toArray)
          new DynamicRateTrainer(gradientTrainer)
        })

        log.code(()⇒{
          val trainingContext = new TrainingContext
          trainingContext.terminalErr = 0.05
          trainer.step(trainingContext)
          val finalError = trainer.step(trainingContext).finalError
          System.out.println(s"Final Error = $finalError")
          model
        })

        log.code(()⇒ {
          val validationTable = new TableOutput()
          MNIST.validationDataStream().iterator().asScala.toStream.take(10).map(testObj⇒{
            val row = new util.LinkedHashMap[String,AnyRef]()
            row.put("Input",log.image(testObj.data.toGrayImage(), testObj.label))
            val result = model.eval(testObj.data).data.head
            val prediction: Int = (0 to 9).maxBy(i⇒result.get(i))
            row.put("Predicted Label", prediction.asInstanceOf[java.lang.Integer])
            row.put("Actual Label",testObj.label)
            row.put("Network Output",result)
            row
          }).foreach(validationTable.putRow(_))
          validationTable
        })


        val categorizationMatrix: Map[Int, Map[Int, Int]] = log.code(()⇒ {
          MNIST.validationDataStream().iterator().asScala.toStream.map(testObj⇒{
            val result = model.eval(testObj.data).data.head
            val prediction: Int = (0 to 9).maxBy(i⇒result.get(i))
            val actual: Int = toOut(testObj.label)
            actual → prediction
          }).groupBy(_._1).mapValues(_.groupBy(_._2).mapValues(_.size))
        })
        log.out(" | Actual \\ Predicted | " + (0 to 9).mkString(" | "))
        log.out((0 to 10).map(_⇒"---").mkString(" | "))
        (0 to 9).foreach(actual⇒{
          log.out(s" **$actual** | " + (0 to 9).map(prediction⇒{
            categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(prediction,0)
          }).mkString(" | "))
        })
        log.code(()⇒ {
          (0 to 9).map(actual⇒{
            actual → (categorizationMatrix(actual)(actual) * 100.0 / categorizationMatrix(actual).values.sum)
          }).toMap
        })
        log.code(()⇒ {
          (0 to 9).map(actual⇒{
            categorizationMatrix(actual)(actual)
          }).sum.toDouble * 100.0 / categorizationMatrix.values.flatMap(_.values).sum
        })


        log.code(()⇒ {
          val validationTable = new TableOutput()
          MNIST.validationDataStream().iterator().asScala.toStream.filterNot(testObj⇒{
            val result = model.eval(testObj.data).data.head
            val prediction: Int = (0 to 9).maxBy(i⇒result.get(i))
            val actual = toOut(testObj.label)
            prediction == actual
          }).take(10).map(testObj⇒{
            val result = model.eval(testObj.data).data.head
            val prediction: Int = (0 to 9).maxBy(i⇒result.get(i))
            val row = new util.LinkedHashMap[String,AnyRef]()
            row.put("Input",log.image(testObj.data.toGrayImage(), testObj.label))
            row.put("Predicted Label", prediction.asInstanceOf[java.lang.Integer])
            row.put("Actual Label", testObj.label)
            row.put("Network Output", result)
            row
          }).foreach(validationTable.putRow(_))
          validationTable
        })

      })
    }





  }

  def toOut(label: String): Int = {
    var i = 0
    while ( {
      i < 10
    }) {
      if (label == "[" + i + "]") return i

      {
        i += 1; i - 1
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