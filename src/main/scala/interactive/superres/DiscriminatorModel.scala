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

package interactive.superres

import java.awt.image.BufferedImage
import java.awt.{Graphics2D, RenderingHints}
import java.io._
import java.lang
import java.util.concurrent.TimeUnit
import java.util.function.{DoubleSupplier, IntToDoubleFunction}
import java.util.stream.Collectors

import _root_.util.Java8Util.cvt
import _root_.util._
import com.google.gson.{GsonBuilder, JsonObject}
import com.simiacryptus.mindseye.data.ImageTiles.ImageTensorLoader
import com.simiacryptus.mindseye.eval.{ArrayTrainable, LinkedExampleArrayTrainable, Trainable}
import com.simiacryptus.mindseye.lang.{NNExecutionContext, NNLayer, NNResult, Tensor}
import com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer
import com.simiacryptus.mindseye.layers.java._
import com.simiacryptus.mindseye.network.graph.DAGNode
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line._
import com.simiacryptus.mindseye.opt.orient._
import com.simiacryptus.text.TableOutput
import com.simiacryptus.util.io.{HtmlNotebookOutput, KryoUtil}
import com.simiacryptus.util.test.LabeledObject
import com.simiacryptus.util.{MonitoredObject, StreamNanoHTTPD, Util}
import org.apache.commons.io.IOUtils
import util.NNLayerUtil._

import scala.collection.JavaConverters._
import scala.util.Random

case class DeepNetworkDescriminator(
                                weight1 : Double,
                                weight2 : Double,
                                weight3 : Double,
                                weight4 : Double
                              ) {

  def getNetwork(monitor: TrainingMonitor,
                 monitoringRoot : MonitoredObject,
                 fitness : Boolean = false) : NNLayer = {
    val parameters = this
    var network: PipelineNetwork = if(fitness) {
      new PipelineNetwork(2)
    } else {
      new PipelineNetwork(1)
    }
    val zeroSeed : IntToDoubleFunction = Java8Util.cvt(_ ⇒ 0.0)
    def buildLayer(from: Int,
                   to: Int,
                   layerNumber: String,
                   weights: Double,
                   layerRadius: Int = 5,
                   simpleBorder: Boolean = false,
                   activationLayer: NNLayer = new ReLuActivationLayer()) = {
      def weightSeed : DoubleSupplier = Java8Util.cvt(() ⇒ {
        val r = Util.R.get.nextDouble() * 2 - 1
        r * weights
      })
      network.add(new ImgBandBiasLayer(from).setWeights(zeroSeed).setName("bias_" + layerNumber).addTo(monitoringRoot))
      if (null != activationLayer) {
        network.add(activationLayer.setName("activation_" + layerNumber).freeze.addTo(monitoringRoot))
      }
      network.add(new ConvolutionLayer(layerRadius, layerRadius, from * to, simpleBorder)
        .setWeights(weightSeed).setName("conv_" + layerNumber).addTo(monitoringRoot));
      //network.fn(new MonitoringSynapse().addTo(monitoringRoot).setName("output_" + layerNumber))
    }

    // 64 x 64 x 3
    val l1 = buildLayer(3, 5, "0", weights = Math.pow(10, parameters.weight1), activationLayer = null)
    network.add(new AvgSubsampleLayer(2, 2, 1).setName("avg0").addTo(monitoringRoot))
    // 30
    val l2 = buildLayer(5, 10, "1", layerRadius = 5, weights = Math.pow(10, parameters.weight2), activationLayer = new HyperbolicActivationLayer().setScale(5).freeze())
    network.add(new AvgSubsampleLayer(2, 2, 1).setName("avg1").addTo(monitoringRoot))
    // 14
    val l3 = buildLayer(10, 20, "2", layerRadius = 4, weights = Math.pow(10, parameters.weight3), activationLayer = new HyperbolicActivationLayer().setScale(5).freeze())
    network.add(new AvgSubsampleLayer(2, 2, 1).setName("avg3").addTo(monitoringRoot))
    // 5
    val l4 = buildLayer(20, 3, "3", layerRadius = 3, weights = Math.pow(10, parameters.weight4), activationLayer = new HyperbolicActivationLayer().setScale(5).freeze())

    network.add(new AvgImageBandLayer().setName("avgFinal").addTo(monitoringRoot))
    network.add(new SoftmaxActivationLayer)

    //network.fn(new ImgReshapeLayer(2,2,true))
    //network.fn(new HyperbolicActivationLayer().setScale(5).freeze().setName("activation4").addTo(monitoringRoot))

    if(fitness) {
      val output = network.getHead
      def normalizeStdDev(layer:DAGNode, target:Double) = network.add(new AbsActivationLayer(), network.add(new SumInputsLayer(),
                network.add(new AvgReducerLayer(), network.add(new StdDevMetaLayer(), layer)),
                network.add(new ConstNNLayer(new Tensor(1).set(0,-target)))
              ))
      network.add(new ProductInputsLayer(), network.add(new EntropyLossLayer(), output, network.getInput(1)), network.add(new SumInputsLayer(),
                network.add(new ConstNNLayer(new Tensor(1).set(0,0.1))),
                normalizeStdDev(l1,1),
                normalizeStdDev(l2,1),
                normalizeStdDev(l3,1),
                normalizeStdDev(l4,1)
              ))
    }

    network
  }

  def fitness(monitor: TrainingMonitor, monitoringRoot : MonitoredObject, data: Array[Array[Tensor]], n: Int = 3) : Double = {
    val values = (1 to n).map(i ⇒ {
      val network = getNetwork(monitor, monitoringRoot, fitness = true)
      val measure = new ArrayTrainable(data, network).measure(false, monitor)
      measure.sum
    }).toList
    val avg = values.sum / n
    monitor.log(s"Numeric Opt: $this => $avg ($values)")
    avg
  }

}

class DiscriminatorModel(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  val modelName = System.getProperty("modelName","descriminator_1")
  val tileSize = 64
  val scaleFactor: Double = (64 * 64.0) / (tileSize * tileSize)
  val sampleTiles = 1000

  def run(awaitExit:Boolean=true): Unit = {
    defineHeader()
    declareTestHandler()
    out.out("<hr/>")
    if(findFile(modelName).isEmpty || System.getProperties.containsKey("rebuild")) step_Generate()
    step_LBFGS((50 * scaleFactor).toInt, 30, 50)
    step_SGD((100 * scaleFactor).toInt, 30, reshufflePeriod = 5)
    step_LBFGS((500 * scaleFactor).toInt, 60, 50)
//    var rates = step_diagnostics_layerRates().mapCoords(e⇒e._1.getName→e._2.rate)
//    step_SGD((500 * scaleFactor).toInt, 60, reshufflePeriod = 1, rates = rates)
    if(null != forwardNetwork) for(i ← 1 to 10) step_Adversarial((10 * scaleFactor).toInt, 60, reshufflePeriod = 1)
    out.out("<hr/>")
    if(awaitExit) waitForExit()
  }

  def resize(source: BufferedImage, size: Int) = {
    val image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB)
    val graphics = image.getGraphics.asInstanceOf[Graphics2D]
    graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
    graphics.drawImage(source, 0, 0, size, size, null)
    image
  }

  def step_Generate() = {
    phase({
      val optTraining: Array[Array[Tensor]] = Random.shuffle(data.toStream.flatten).take((10 * scaleFactor).ceil.toInt).toArray
      SimplexOptimizer[DeepNetworkDescriminator](
        DeepNetworkDescriminator(-3.1962815165239653,0.5,-2.5,-7.5),//(-2.1962815165239653,1.0,-2.0,-6.0),
        x ⇒ x.fitness(monitor, monitoringRoot, optTraining, n=3), relativeTolerance=0.01
      ).getNetwork(monitor, monitoringRoot)
    }, (model: NNLayer) ⇒ {
      out.h1("Model Initialization")
      val trainer = out.eval {
        val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer())
        var inner: Trainable = new LinkedExampleArrayTrainable(data, trainingNetwork, (50 * scaleFactor).toInt)
        //inner = new ConstL12Normalizer(inner).setFactor_L1(0.001)
        val trainer = new IterativeTrainer(inner)
        trainer.setMonitor(monitor)
        trainer.setTimeout(5, TimeUnit.MINUTES)
        trainer.setIterationsPerSample(1)
        trainer.setOrientation(new GradientDescent())
        trainer.setLineSearchFactory(Java8Util.cvt((s) ⇒ new QuadraticSearch))
        trainer.setTerminateThreshold(1.0)
        trainer
      }
      trainer.run()
    }: Unit, modelName)
  }

  def step_diagnostics_layerRates(sampleSize : Int = (100 * scaleFactor).toInt) = phase[Map[NNLayer, LayerRateDiagnosticTrainer.LayerStats]](
    modelName, (model: NNLayer) ⇒ {
    val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer())
    out.h1("Diagnostics - Layer Rates")
    out.eval {
      var inner: Trainable = new LinkedExampleArrayTrainable(data, trainingNetwork, sampleSize)
      val trainer = new LayerRateDiagnosticTrainer(inner).setStrict(true).setMaxIterations(1)
      trainer.setMonitor(monitor)
      trainer.run()
      trainer.getLayerRates().asScala.toMap
    }
  }, modelName)

  def step_SGD(sampleSize: Int, timeoutMin: Int, termValue: Double = 0.0, momentum: Double = 0.2, maxIterations: Int = Integer.MAX_VALUE, reshufflePeriod: Int = 1,rates: Map[String, Double] = Map.empty) = phase(modelName, (model: NNLayer) ⇒ {
    monitor.clear()
    out.h1(s"SGD(sampleSize=$sampleSize,timeoutMin=$timeoutMin)")
    val trainer = out.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer())
      var inner: Trainable = new LinkedExampleArrayTrainable(data, trainingNetwork, sampleSize)
      //inner = new ConstL12Normalizer(inner).setFactor_L1(0.001)
      val trainer = new IterativeTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(timeoutMin, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(reshufflePeriod)
      val momentumStrategy = new MomentumStrategy(new GradientDescent()).setCarryOver(momentum)
      val reweight = new LayerReweightingStrategy(momentumStrategy) {
        override def getRegionPolicy(layer: NNLayer): lang.Double = layer.getName match {
          case key if rates.contains(key) ⇒ rates(key)
          case _ ⇒ 1.0
        }
        override def reset(): Unit = {}
      }
      trainer.setOrientation(reweight)
      trainer.setLineSearchFactory(Java8Util.cvt((s)⇒new ArmijoWolfeSearch().setAlpha(1e-12).setC1(0).setC2(1)))
      trainer.setTerminateThreshold(termValue)
      trainer.setMaxIterations(maxIterations)
      trainer
    }
    trainer.run()
  }, modelName)

  lazy val forwardNetwork = loadModel("downsample_1")

  def step_Adversarial(sampleSize: Int, timeoutMin: Int, termValue: Double = 0.0, momentum: Double = 0.2, maxIterations: Int = Integer.MAX_VALUE, reshufflePeriod: Int = 1,rates: Map[String, Double] = Map.empty) = phase(modelName, (model: NNLayer) ⇒ {
    monitor.clear()
    out.h1(s"Adversarial(sampleSize=$sampleSize,timeoutMin=$timeoutMin)")
    lazy val startModel = KryoUtil.kryo().copy(model)
    lazy val adversarialData = Random.shuffle(data.toList).take(sampleSize).toStream.map((x: Array[Array[Tensor]]) ⇒{
      assert(3 == x.length)
      assert(2 == x(0).length)
      val original = x(0)(0)
      val downsampled = Tensor.fromRGB(UpsamplingOptimizer.resize(original.toRgbImage, tileSize / 4))
      val reconstruct = UpsamplingOptimizer.reconstructImage(forwardNetwork, startModel, downsampled, monitor)
      x ++ Array(Array(reconstruct, new Tensor(3).set(2,1)))
    })
    out.eval {
      TableOutput.create(adversarialData.map((data: Array[Array[Tensor]]) ⇒ {
        assert(4 == data.length)
        assert(2 == data(0).length)
        assert(data.forall(2 == _.length))
        assert(data.forall(3 == _(1).dim()))
        Map[String, AnyRef](
          "Original Image" → out.image(data(0)(0).toRgbImage, ""),
          "Reconsructed" → out.image(data(3)(0).toRgbImage, "")
        ).asJava
      }): _*)
    }
    summarizeHistory()
    monitor.clear()
    val trainer = out.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer())
      var inner: Trainable = new ArrayTrainable(adversarialData.toList.flatten.toArray, trainingNetwork)
      //inner = new ConstL12Normalizer(inner).setFactor_L1(0.001)
      val trainer = new IterativeTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(timeoutMin, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(reshufflePeriod)
      val momentumStrategy = new MomentumStrategy(new GradientDescent()).setCarryOver(momentum)
      val reweight = new LayerReweightingStrategy(momentumStrategy) {
        override def getRegionPolicy(layer: NNLayer): lang.Double = layer.getName match {
          case key if rates.contains(key) ⇒ rates(key)
          case _ ⇒ 1.0
        }
        override def reset(): Unit = {}
      }
      trainer.setOrientation(reweight)
      trainer.setLineSearchFactory(Java8Util.cvt((s)⇒new ArmijoWolfeSearch().setAlpha(1e-12).setC1(0).setC2(1)))
      trainer.setTerminateThreshold(termValue)
      trainer.setMaxIterations(maxIterations)
      trainer
    }
    trainer.run()
  }, modelName)

  def step_LBFGS(sampleSize: Int, timeoutMin: Int, iterationSize: Int): Unit = phase(modelName, (model: NNLayer) ⇒ {
    monitor.clear()
    out.h1(s"LBFGS(sampleSize=$sampleSize,timeoutMin=$timeoutMin)")
    val trainer = out.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer())
      val inner = new LinkedExampleArrayTrainable(data, trainingNetwork, sampleSize)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(timeoutMin, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(iterationSize)
      val lbfgs = new LBFGS().setMaxHistory(35).setMinHistory(4)
      trainer.setOrientation(lbfgs)
      trainer.setLineSearchFactory(Java8Util.cvt((s:String)⇒(s match {
        case s if s.contains("LBFGS") ⇒ new StaticLearningRate().setRate(1.0)
        case _ ⇒ new ArmijoWolfeSearch().setAlpha(1e-5)
      })))
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    trainer.run()
  }, modelName)

  def declareTestHandler() = {
    out.p("<a href='testCat.html'>Test Categorization</a><br/>")
    server.addSyncHandler("testCat.html", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(out ⇒ {
        testCategorization(out, getModelCheckpoint)
      })
    }), false)
  }

  def testCategorization(out: HtmlNotebookOutput with ScalaNotebookOutput, model : NNLayer) = {
    try {
      out.eval {
        TableOutput.create(Random.shuffle(data.flatten.toList).take(100).map(testObj ⇒ Map[String, AnyRef](
          "Image" → out.image(testObj(0).toRgbImage(), ""),
          "Categorization" → categories.toList.sortBy(_._2).map(_._1)
            .zip(model.eval(new NNExecutionContext() {}, testObj(0)).getData.get(0).getData.map(_ * 100.0))
        ).asJava): _*)
      }
    } catch {
      case e: Throwable ⇒ e.printStackTrace()
    }
  }

  val corruptors = List[(String, Tensor ⇒ Tensor)](
    "noise" → (imgTensor ⇒ {
      imgTensor.map(Java8Util.cvt((x:Double)⇒Math.min(Math.max(x+(50.0*(Random.nextDouble()-0.5)), 0.0), 256.0)))
    }), "resample" → (imgTensor ⇒ {
      Tensor.fromRGB(resize(resize(imgTensor.toRgbImage, tileSize/4), tileSize))
    })
  )


  lazy val (categories: Map[String, Int], data: Array[Array[Array[Tensor]]]) = {
    def toOut(label: String, max: Int): Int = {
      (0 until max).find(label == "[" + _ + "]").get
    }
    def toOutNDArray(out: Int, max: Int): Tensor = {
      val ndArray = new Tensor(max)
      ndArray.set(out, 1)
      ndArray
    }

    val categories: Map[String, Int] = (List("original") ++ corruptors.map(_._1)).zipWithIndex.toMap

    val filename = "filterNetwork.json"
    val preFilter : Seq[Tensor] ⇒ Seq[Tensor] = if(new File(filename).exists()) {
      val filterNetwork = NNLayer.fromJson(new GsonBuilder().create().fromJson(IOUtils.toString(new FileInputStream(filename), "UTF-8"), classOf[JsonObject]))
      (obj:Seq[Tensor]) ⇒ {
        import scala.collection.JavaConverters._
        obj.grouped(1000).toStream.flatMap(obj ⇒ {
          filterNetwork.eval(new NNExecutionContext() {}, NNResult.batchResultArray(obj.map(y ⇒ Array(y)).toArray)).getData.stream().collect(Collectors.toList()).asScala
        })
          .zip(obj).sortBy(-_._1.get(categories("noise"))).take(1000).map(_._2)
      }
    } else {
      x⇒x
    }

    out.p("Loading data from " + source)
    val loader = new ImageTensorLoader(new File(source), tileSize, tileSize, tileSize, tileSize, 10, 10)
    val unfilteredData = loader.stream().iterator().asScala.take(sampleTiles).toArray
    loader.stop()
    out.p("Preparing training dataset")

    val rawData: List[List[LabeledObject[Tensor]]] = preFilter(unfilteredData).map(tile ⇒ List(
      new LabeledObject[Tensor](tile, "original")
    ) ++ corruptors.map(e ⇒ {
      new LabeledObject[Tensor](e._2(tile), e._1)
    })).take(sampleTiles).toList
    out.p("<ol>" + categories.toList.sortBy(_._2).map(x ⇒ "<li>" + x + "</li>").mkString("\n") + "</ol>")
    val data: Array[Array[Array[Tensor]]] = rawData.map(rawData⇒rawData.map((labeledObj: LabeledObject[Tensor]) ⇒ {
      Array(labeledObj.data, toOutNDArray(categories(labeledObj.label), categories.size))
    }).toArray).toArray
    out.eval {
      TableOutput.create(rawData.flatten.take(100).map(testObj ⇒ {
        val checkpoint = getModelCheckpoint
        if(null != checkpoint) {
          Map[String, AnyRef](
            "Image" → out.image(testObj.data.toRgbImage(), testObj.data.toString),
            "Label" → testObj.label,
            "Categorization" → categories.toList.sortBy(_._2).map(_._1)
              .zip(checkpoint.eval(new NNExecutionContext() {}, testObj.data).getData.get(0).getData.map(_ * 100.0)).mkString(", ")
          ).asJava
        } else {
          Map[String, AnyRef](
            "Image" → out.image(testObj.data.toRgbImage(), testObj.data.toString),
            "Label" → testObj.label
          ).asJava
        }
      }): _*)
    }
    out.p("Loading data complete")
    (categories, data)
  }

}

object DiscriminatorModel extends Report {

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒ new DiscriminatorModel(source, server, out).run()
      case _ ⇒ new DiscriminatorModel("E:\\testImages\\256_ObjectCategories", server, out).run()
    })

  }

}
