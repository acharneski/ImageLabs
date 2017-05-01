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

import java.awt.{Color, Graphics2D}
import java.awt.geom.AffineTransform
import java.awt.image.{AffineTransformOp, BufferedImage}
import java.util.UUID
import java.util.concurrent.TimeUnit
import java.util.function.{DoubleSupplier, ToDoubleFunction}
import java.{lang, util}
import javax.imageio.ImageIO

import com.simiacryptus.mindseye.net.activation.{AbsActivationLayer, L1NormalizationLayer, LinearActivationLayer, SoftmaxActivationLayer}
import com.simiacryptus.mindseye.net.basic.BiasLayer
import com.simiacryptus.mindseye.net.dag.{DAGNetwork, DAGNode, InnerNode}
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerJBLAS
import com.simiacryptus.mindseye.net.loss.{EntropyLossLayer, SqLossLayer}
import com.simiacryptus.mindseye.net.media.{ConvolutionSynapseLayer, EntropyLayer}
import com.simiacryptus.mindseye.net.reducers.SumInputsLayer
import com.simiacryptus.mindseye.net.util.VerboseWrapper
import com.simiacryptus.mindseye.training.{IterativeTrainer, LbfgsTrainer, TrainingContext}
import com.simiacryptus.util.Util
import com.simiacryptus.util.ml.{Coordinate, Tensor}
import com.simiacryptus.util.test.MNIST
import com.simiacryptus.util.text.TableOutput
import guru.nidi.graphviz.attribute.RankDir
import guru.nidi.graphviz.engine.{Format, Graphviz}
import guru.nidi.graphviz.model._
import org.scalatest.{MustMatchers, WordSpec}
import smile.plot.{PlotCanvas, ScatterPlot}

import scala.collection.JavaConverters._
import scala.util.Random

class MindsEyeDemo extends WordSpec with MustMatchers with MarkdownReporter {


  "MindsEye Demo" should {

    "Train Simple Digit Recognizer" in {
      report("mnist_simple", log ⇒ {
        val inputSize = Array[Int](28, 28, 1)
        val outputSize = Array[Int](10)
        log.p("In this demo we train a simple neural network against the MNIST handwritten digit dataset")

        log.h2("Data")
        log.p("First, we load the training dataset: ")
        val data: Seq[Array[Tensor]] = log.code(() ⇒ {
          MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
            Array(labeledObj.data, toOutNDArray(toOut(labeledObj.label), 10))
          })
        })
        log.p("And preview a few rows: ")
        log.eval {
          TableOutput.create(data.take(10).map(testObj ⇒ Map[String,AnyRef](
            "Input1 (as Image)" → log.image(testObj(0).toGrayImage(), testObj(0).toString),
            "Input2 (as String)" → testObj(1).toString,
            "Input1 (as String)" → testObj(0).toString
          ).asJava):_*)
        }

        log.h2("Model")
        log.p("Here we define the logic network that we are about to train: ")
        var model: DAGNetwork = log.eval {
          var model: DAGNetwork = new DAGNetwork
          model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.0
          }))
          model = model.add(new BiasLayer(outputSize: _*))
          model = model.add(new SoftmaxActivationLayer)
          model
        }
        log.p("We can visualize this network as a graph: ")
        networkGraph(log, model, 800)
        log.p("We encapsulate our model network within a supervisory network that applies a loss function: ")
        val trainingNetwork: DAGNetwork = log.eval {
          val trainingNetwork: DAGNetwork = new DAGNetwork
          trainingNetwork.add(model)
          trainingNetwork.addLossComponent(new EntropyLossLayer)
          trainingNetwork
        }
        log.p("With a the following component graph: ")
        networkGraph(log, trainingNetwork, 600)
        log.p("Note that this visualization does not expand DAGNetworks recursively")

        log.h2("Training")
        log.p("We train using a standard iterative L-BFGS strategy: ")
        val trainer = log.eval {
          val gradientTrainer: LbfgsTrainer = new LbfgsTrainer
          gradientTrainer.setNet(trainingNetwork)
          gradientTrainer.setData(data.toArray)
          new IterativeTrainer(gradientTrainer)
        }
        log.eval {
          val trainingContext = new TrainingContext
          trainingContext.terminalErr = 0.0
          trainingContext.setTimeout(5, TimeUnit.SECONDS)
          trainer.step(trainingContext)
          val finalError = trainer.step(trainingContext).finalError
          System.out.println(s"Final Error = $finalError")
        }
        log.p("After training, we have the following parameterized model: ")
        log.eval {
          model.toString
        }
        log.p("A summary of the training timeline: ")
        summarizeHistory(log, trainer.history)

        log.h2("Validation")
        log.p("Here we examine a sample of validation rows, randomly selected: ")
        log.eval {
          TableOutput.create(MNIST.validationDataStream().iterator().asScala.toStream.take(10).map(testObj ⇒ {
            val result = model.eval(testObj.data).data.head
            Map[String, AnyRef](
              "Input" → log.image(testObj.data.toGrayImage(), testObj.label),
              "Predicted Label" → (0 to 9).maxBy(i ⇒ result.get(i)).asInstanceOf[java.lang.Integer],
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
              "Predicted Label" → (0 to 9).maxBy(i ⇒ result.get(i)).asInstanceOf[java.lang.Integer],
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
            actual → (categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0) * 100.0 / categorizationMatrix.getOrElse(actual,Map.empty).values.sum)
          }).toMap
        }
        log.p("The accuracy, summarized over the entire validation set: ")
        log.eval {
          (0 to 9).map(actual ⇒ {
            categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0)
          }).sum.toDouble * 100.0 / categorizationMatrix.values.flatMap(_.values).sum
        }

      })
    }

    "Learns Simple 2d Functions" in {
      report("2d_simple", log ⇒ {

        val inputSize = Array[Int](2)
        val outputSize = Array[Int](2)
        val MAX: Int = 2

        def runTest(function: (Double, Double) ⇒ Int, model: DAGNetwork) = {

          val trainingData: Seq[Array[Tensor]] = Stream.continually({
            val x = Random.nextDouble() * 2.0 - 1.0
            val y = Random.nextDouble() * 2.0 - 1.0
            Array(new Tensor(Array(2), Array(x, y)), toOutNDArray(function(x, y), 2))
          }).take(100)
          val validationData: Seq[Array[Tensor]] = Stream.continually({
            val x = Random.nextDouble() * 2.0 - 1.0
            val y = Random.nextDouble() * 2.0 - 1.0
            Array(new Tensor(Array(2), Array(x, y)), toOutNDArray(function(x, y), 2))
          }).take(100)

          val trainer = {
            val trainingNetwork: DAGNetwork = new DAGNetwork
            trainingNetwork.add(model)
            trainingNetwork.addLossComponent(new EntropyLossLayer)
            val gradientTrainer: LbfgsTrainer = new LbfgsTrainer
            gradientTrainer.setNet(trainingNetwork)
            gradientTrainer.setData(trainingData.toArray)
            new IterativeTrainer(gradientTrainer)
          }

          {
            val trainingContext = new TrainingContext
            trainingContext.terminalErr = 0.05
            trainer.step(trainingContext)
            val finalError = trainer.step(trainingContext).finalError
            System.out.println(s"Final Error = $finalError")
            model
          }

          def plotXY(gfx : Graphics2D) = {
            (0 to 400).foreach(x ⇒ (0 to 400).foreach(y ⇒ {
              function((x / 200.0) - 1.0, (y / 200.0) - 1.0) match {
                case 0 ⇒ gfx.setColor(Color.RED)
                case 1 ⇒ gfx.setColor(Color.GREEN)
              }
              gfx.drawRect(x, y, 1, 1)
            }))
            validationData.foreach(testObj ⇒ {
              val row = new util.LinkedHashMap[String, AnyRef]()
              val result = model.eval(testObj(0)).data.head
              (0 until MAX).maxBy(i ⇒ result.get(i)) match {
                case 0 ⇒ gfx.setColor(Color.PINK)
                case 1 ⇒ gfx.setColor(Color.BLUE)
              }
              val xx = testObj(0).get(0) * 200.0 + 200.0
              val yy = testObj(0).get(1) * 200.0 + 200.0
              gfx.drawRect(xx.toInt - 1, yy.toInt - 1, 3, 3)
            })
          }
          log.draw(gfx ⇒ {
            plotXY(gfx)
          }, width = 600, height = 600)
          val categorizationMatrix: Map[Int, Map[Int, Int]] = {
            validationData.map(testObj ⇒ {
              val result = model.eval(testObj(0)).data.head
              val prediction: Int = (0 until MAX).maxBy(i ⇒ result.get(i))
              val actual: Int = (0 until MAX).maxBy(i ⇒ testObj(1).get(i))
              actual → prediction
            }).groupBy(_._1).mapValues(_.groupBy(_._2).mapValues(_.size))
          }
          val byCategory = (0 until MAX).map(actual ⇒ {
            actual → (categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0) * 100.0 / categorizationMatrix.getOrElse(actual, Map.empty).values.sum)
          }).toMap
          val overall = (0 until MAX).map(actual ⇒ {
            categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0)
          }).sum.toDouble * 100.0 / categorizationMatrix.values.flatMap(_.values).sum
          log.eval {
            overall → byCategory
          }
          //summarizeHistory(log, trainer.history)
        }

        log.h2("Linear")
        runTest(log.eval {
          (x: Double, y: Double) ⇒ if (x < y) 0 else 1
        }, log.eval {
          var model: DAGNetwork = new DAGNetwork
          model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.1
          }))
          model = model.add(new BiasLayer(outputSize: _*))
          model = model.add(new SoftmaxActivationLayer)
          model
        })

        log.h2("XOR")
        val xor_fn = log.eval {
          (x: Double, y: Double) ⇒ if ((x < 0) ^ (y < 0)) 0 else 1
        }
        runTest(xor_fn, log.eval {
          var model: DAGNetwork = new DAGNetwork
          model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.2
          }))
          model = model.add(new BiasLayer(outputSize: _*))
          model = model.add(new SoftmaxActivationLayer)
          model
        })
        runTest(xor_fn, log.eval {
          var model: DAGNetwork = new DAGNetwork
          val middleSize = Array[Int](15)
          model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), middleSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 1
          }))
          model = model.add(new BiasLayer(middleSize: _*))
          model = model.add(new AbsActivationLayer())
          model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(middleSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 1
          }))
          model = model.add(new BiasLayer(outputSize: _*))
          model = model.add(new SoftmaxActivationLayer)
          model
        })

        log.h2("Circle")
        val circle_fn = log.eval {
          (x: Double, y: Double) ⇒ if ((x * x) + (y * y) < 0.5) 0 else 1
        }
        runTest(circle_fn, log.eval {
          var model: DAGNetwork = new DAGNetwork
          model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 0.2
          }))
          model = model.add(new BiasLayer(outputSize: _*))
          model = model.add(new SoftmaxActivationLayer)
          model
        })
        runTest(circle_fn, log.eval {
          var model: DAGNetwork = new DAGNetwork
          val middleSize = Array[Int](15)
          model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(inputSize: _*), middleSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 1
          }))
          model = model.add(new BiasLayer(middleSize: _*))
          model = model.add(new AbsActivationLayer())
          model = model.add(new DenseSynapseLayerJBLAS(Tensor.dim(middleSize: _*), outputSize).setWeights(new ToDoubleFunction[Coordinate] {
            override def applyAsDouble(value: Coordinate): Double = Util.R.get.nextGaussian * 1
          }))
          model = model.add(new BiasLayer(outputSize: _*))
          model = model.add(new SoftmaxActivationLayer)
          model
        })

      })
    }

    "Reverse Blur Filter" in {
      report("deconvolution", log ⇒ {

        log.p("First we define a forward filter, in this case a simple convolution filter emulating motion blur")
        val blurFilter = log.eval {
          def singleConvolution: ConvolutionSynapseLayer = {
            val convolution = new ConvolutionSynapseLayer(Array[Int](3, 3), 9)
            (0 until 3).foreach(ii⇒{
              val i = ii + ii * 3
              convolution.kernel.set(Array[Int](0, 2, i), 0.333)
              convolution.kernel.set(Array[Int](1, 1, i), 0.333)
              convolution.kernel.set(Array[Int](2, 0, i), 0.333)
            })
            convolution.freeze
            convolution
          }
          val net = new DAGNetwork
          net.add(singleConvolution)
          net.add(singleConvolution)
          net.add(singleConvolution)
          net
        }

        log.p("We load an ideal training image, which we will try to reconstruct: ")
        val idealImage = log.eval {
          val read = ImageIO.read(getClass.getResourceAsStream("/monkey1.jpg"))
          def scale(img: BufferedImage, scale: Double) = {
            val w = img.getWidth
            val h = img.getHeight
            val after = new BufferedImage((w * scale).toInt, (h * scale).toInt, BufferedImage.TYPE_INT_ARGB)
            val at = new AffineTransform
            at.scale(scale, scale)
            new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR).filter(img, after)
          }
          scale(read, 0.5)
        }

        log.p("Next we run this ideal image through our constructed filter to create a blurred image: ")
        val idealImageTensor: Tensor = Tensor.fromRGB(idealImage)
        val blurredImage: Tensor = log.eval {
          blurFilter.eval(Array(Array(idealImageTensor))).data.head
        }
        log.eval {
          blurredImage.toRgbImage()
        }

        val inputSize: Array[Int] = idealImageTensor.getDims
        val zeroInput = new Tensor(inputSize:_*)

        log.p("Now we define a reconstruction network, which adapts a bias layer to find the source image" +
          " given a post-filter image while also considering normalization factors including image entropy: ")
        val (bias, dagNetwork) = log.eval {
          val net = new DAGNetwork
          val bias = new BiasLayer(inputSize: _*)
          val modeledImageNode = net.add(bias).getHead
          net.add(blurFilter)
          net.addLossComponent(new SqLossLayer)
          val imageRMS: DAGNode = net.add(new VerboseWrapper("rms", new BiasLayer().freeze)).getHead
          net.add(new AbsActivationLayer, modeledImageNode)
          net.add(new L1NormalizationLayer)
          net.add(new EntropyLayer)
          net.add(new SumInputsLayer)
          val image_entropy: DAGNode = net.add(new VerboseWrapper("entropy", new BiasLayer().freeze)).getHead
          val scaledRms: DAGNode = net.add(new LinearActivationLayer().setWeight(1.0).freeze, imageRMS).getHead
          val scaledEntropy: DAGNode = net.add(new LinearActivationLayer().setWeight(0.001).freeze, image_entropy).getHead
          net.add(new VerboseWrapper("composite", new SumInputsLayer), scaledRms, scaledEntropy)
          (bias, net)
        }
        networkGraph(log, dagNetwork)

        log.p("Now we define a standard L-BFGS trainer to optimize the reconstruction")
        val trainer = log.eval {
          val gradientTrainer: LbfgsTrainer = new LbfgsTrainer
          gradientTrainer.setNet(dagNetwork)
          gradientTrainer.setData(Seq(Array(zeroInput, blurredImage)).toArray)
          new IterativeTrainer(gradientTrainer)
        }
        log.eval {
          bias.addWeights(new DoubleSupplier {
            override def getAsDouble: Double = Util.R.get.nextGaussian * 1e-5
          })
          val trainingContext = new TrainingContext
          trainingContext.terminalErr = 0.05
          trainer.step(trainingContext)
          val finalError = trainer.step(trainingContext).finalError
          System.out.println(s"Final Error = $finalError")
        }
        log.p("Which results in the convergence timeline: ")
        summarizeHistory(log, trainer.history)

        log.p("Now we query the reconstruction model for the source image: ")
        log.eval {
          dagNetwork.getChild(bias.getId).asInstanceOf[BiasLayer].eval(zeroInput).data(0).toRgbImage()
        }

      })
    }

  }

  private def summarizeHistory(log: ScalaMarkdownPrintStream, history: util.ArrayList[IterativeTrainer.StepState]) = {
    log.eval {
      val step = Math.max(Math.pow(10,Math.ceil(Math.log(history.size()) / Math.log(10))-2), 1).toInt
      TableOutput.create(history.asScala.filter(0==_.getIteration%step).map(state ⇒
        Map[String, AnyRef](
          "iteration" → state.getIteration.toInt.asInstanceOf[Integer],
          "time" → state.getEvaluationTime.toDouble.asInstanceOf[lang.Double],
          "fitness" → state.getFitness.toDouble.asInstanceOf[lang.Double]
        ).asJava
      ): _*)
    }
    log.eval {
      val plot: PlotCanvas = ScatterPlot.plot(history.asScala.map(item ⇒ Array[Double](
        item.getIteration, Math.log(item.getFitness)
      )).toArray: _*)
      plot.setTitle("Convergence Plot")
      plot.setAxisLabels("Iteration", "log(Fitness)")
      plot.setSize(600, 400)
      plot
    }
  }

  private def networkGraph(log: ScalaMarkdownPrintStream, dagNetwork: DAGNetwork, width: Int = 1200) = {
    log.eval {
      val nodes: List[DAGNode] = dagNetwork.getNodes.asScala.toList
      val graphNodes: Map[UUID, MutableNode] = nodes.map(node ⇒ {
        node.getId() → guru.nidi.graphviz.model.Factory.mutNode((node match {
          case n : InnerNode ⇒
            n.nnlayer match {
              case _ if(n.nnlayer.isInstanceOf[VerboseWrapper]) ⇒ n.nnlayer.asInstanceOf[VerboseWrapper].inner.getClass.getSimpleName
              case _ ⇒ n.nnlayer.getClass.getSimpleName
            }
          case _ ⇒ node.getClass.getSimpleName
        }) + "\n" + node.getId.toString)
      }).toMap
      val idMap: Map[UUID, List[UUID]] = nodes.flatMap((to: DAGNode) ⇒ {
        to.getInputs.map((from: DAGNode) ⇒ {
          from.getId → to.getId
        })
      }).groupBy(_._1).mapValues(_.map(_._2))
      nodes.foreach((to: DAGNode) ⇒ {
        graphNodes(to.getId).addLink(idMap.getOrElse(to.getId, List.empty).map(from ⇒ {
          Link.to(graphNodes(from))
        }): _*)
      })
      val nodeArray = graphNodes.values.map(_.asInstanceOf[LinkSource]).toArray
      val graph = guru.nidi.graphviz.model.Factory.graph().`with`(nodeArray: _*)
        .generalAttr.`with`(RankDir.TOP_TO_BOTTOM).directed()
      Graphviz.fromGraph(graph).width(width).render(Format.PNG).toImage
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