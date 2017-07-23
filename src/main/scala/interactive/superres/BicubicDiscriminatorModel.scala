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

import _root_.util.Java8Util.cvt
import _root_.util._
import com.google.gson.{GsonBuilder, JsonObject}
import com.simiacryptus.mindseye.layers.{NNLayer, NNResult}
import com.simiacryptus.mindseye.layers.activation._
import com.simiacryptus.mindseye.layers.loss.{EntropyLossLayer, MeanSqLossLayer}
import com.simiacryptus.mindseye.layers.media.{ImgBandBiasLayer, ImgConvolutionSynapseLayer, ImgReshapeLayer, MaxSubsampleLayer}
import com.simiacryptus.mindseye.layers.meta.StdDevMetaLayer
import com.simiacryptus.mindseye.layers.reducers.{AvgReducerLayer, ProductInputsLayer, SumInputsLayer, SumReducerLayer}
import com.simiacryptus.mindseye.layers.util.{ConstNNLayer, MonitoringWrapper}
import com.simiacryptus.mindseye.network.graph.DAGNode
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line._
import com.simiacryptus.mindseye.opt.orient._
import com.simiacryptus.mindseye.opt.trainable._
import com.simiacryptus.util.{MonitoredObject, StreamNanoHTTPD, Util}
import com.simiacryptus.util.io.HtmlNotebookOutput
import com.simiacryptus.util.ml.{Coordinate, Tensor}
import com.simiacryptus.util.test.ImageTiles.ImageTensorLoader
import com.simiacryptus.util.test.LabeledObject
import com.simiacryptus.util.text.TableOutput
import org.apache.commons.io.IOUtils

import scala.collection.JavaConverters._
import scala.util.Random
import NNLayerUtil._
import com.simiacryptus.mindseye.layers.synapse.BiasLayer
import com.simiacryptus.mindseye.opt.region.{StaticConstraint, TrustRegion}

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
                   activationLayer: NNLayer = new ReLuActivationLayer()) = {
      def weightSeed : DoubleSupplier = Java8Util.cvt(() ⇒ {
        val r = Util.R.get.nextDouble() * 2 - 1
        r * weights
      })
      network.add(new ImgBandBiasLayer(from).setWeights(zeroSeed).setName("bias_" + layerNumber).addTo(monitoringRoot))
      if (null != activationLayer) {
        network.add(activationLayer.setName("activation_" + layerNumber).freeze.addTo(monitoringRoot))
      }
      network.add(new ImgConvolutionSynapseLayer(layerRadius, layerRadius, from * to).setWeights(weightSeed).setName("conv_" + layerNumber).addTo(monitoringRoot));
      //network.add(new MonitoringSynapse().addTo(monitoringRoot).setName("output_" + layerNumber))
    }

    // 64 x 64 x 3
    val l1 = buildLayer(3, 5, "0", weights = Math.pow(10, parameters.weight1), activationLayer = null)
    network.add(new MaxSubsampleLayer(2, 2, 1).setName("max0").addTo(monitoringRoot))
    // 32 x 32 x 64
    val l2 = buildLayer(5, 10, "1", weights = Math.pow(10, parameters.weight2), activationLayer = new HyperbolicActivationLayer().setScale(5).freeze())
    network.add(new MaxSubsampleLayer(2, 2, 1).setName("max1").addTo(monitoringRoot))
    // 16 x 16 x 64
    val l3 = buildLayer(10, 20, "2", weights = Math.pow(10, parameters.weight3), activationLayer = new HyperbolicActivationLayer().setScale(5).freeze())
    network.add(new MaxSubsampleLayer(2, 2, 1).setName("max3").addTo(monitoringRoot))
    // 8 x 8 x 32
    val l4 = buildLayer(20, 3, "3", weights = Math.pow(10, parameters.weight4), activationLayer = new HyperbolicActivationLayer().setScale(5).freeze())
    network.add(new MaxSubsampleLayer(8, 8, 1).setName("max4").addTo(monitoringRoot))
    // 1 x 1 x 2
    network.add(new SoftmaxActivationLayer)

    //network.add(new ImgReshapeLayer(2,2,true))
    //network.add(new HyperbolicActivationLayer().setScale(5).freeze().setName("activation4").addTo(monitoringRoot))

    if(fitness) {
      val output = network.getHead
      def normalizeStdDev(layer:DAGNode, target:Double) = network.add(new AbsActivationLayer(),
        network.add(new SumInputsLayer(),
          network.add(new AvgReducerLayer(), network.add(new StdDevMetaLayer(), layer)),
          network.add(new ConstNNLayer(new Tensor(1).set(0,-target)))
        )
      )
      network.add(new ProductInputsLayer(),
        network.add(new EntropyLossLayer(), output, network.getInput(1)),
        network.add(new SumInputsLayer(),
          network.add(new ConstNNLayer(new Tensor(1).set(0,0.1))),
          normalizeStdDev(l1,1),
          normalizeStdDev(l2,1),
          normalizeStdDev(l3,1),
          normalizeStdDev(l4,1)
        )
      )
    }

    network
  }

  def fitness(monitor: TrainingMonitor, monitoringRoot : MonitoredObject, data: Array[Array[Tensor]], n: Int = 3) : Double = {
    val values = (1 to n).map(i ⇒ {
      val network = getNetwork(monitor, monitoringRoot, fitness = true)
      val measure = new ArrayTrainable(data, network).measure()
      measure.value
    }).toList
    val avg = values.sum / n
    monitor.log(s"Numeric Opt: $this => $avg ($values)")
    avg
  }

}

class BicubicDiscriminatorModel(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  val modelName = System.getProperty("modelName","descriminator_1")
  val tileSize = 64
  val fitnessBorderPadding = 8
  val scaleFactor: Double = (64 * 64.0) / (tileSize * tileSize)
  val sampleTiles = 10000

  def run(): Unit = {
    defineHeader()
    declareTestHandler()
    out.out("<hr/>")
    if(findFile(modelName).isEmpty || System.getProperties.containsKey("rebuild")) step_Generate()
    //step_LBFGS((50 * scaleFactor).toInt, 30, 50)
    //step_SGD((100 * scaleFactor).toInt, 30, reshufflePeriod = 5)
    step_LBFGS((500 * scaleFactor).toInt, 60, 50)
    var rates = step_diagnostics_layerRates().map(e⇒e._1.getName→e._2.rate)
    step_SGD((500 * scaleFactor).toInt, 60, reshufflePeriod = 1, rates = rates)
    summarizeHistory()
    out.out("<hr/>")
    waitForExit()
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
      out.h1("Step 1")
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
      trainer.setLineSearchFactory(Java8Util.cvt((s)⇒new ArmijoWolfeSearch().setAlpha(1e-12).setC1(0).setC2(0)))
      trainer.setTerminateThreshold(termValue)
      trainer.setMaxIterations(maxIterations)
      trainer
    }
    trainer.run()
  }, modelName)

  def step_LBFGS(sampleSize: Int, timeoutMin: Int, iterationSize: Int): Unit = phase(modelName, (model: NNLayer) ⇒ {
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
            .zip(model.eval(testObj(0)).data.head.getData.map(_ * 100.0))
        ).asJava): _*)
      }
    } catch {
      case e: Throwable ⇒ e.printStackTrace()
    }
  }

  val corruptors = Map[String, Tensor ⇒ Tensor](
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

    val labels = List("original") ++ corruptors.keys.toList.sorted
    val categories: Map[String, Int] = labels.zipWithIndex.toMap

    val filename = "filterNetwork.json"
    val preFilter : Seq[Tensor] ⇒ Seq[Tensor] = if(new File(filename).exists()) {
      val filterNetwork = NNLayer.fromJson(new GsonBuilder().create().fromJson(IOUtils.toString(new FileInputStream(filename), "UTF-8"), classOf[JsonObject]))
      (obj:Seq[Tensor]) ⇒ {
        obj.grouped(1000).toStream.flatMap(obj ⇒ filterNetwork.eval(NNResult.batchResultArray(obj.map(y ⇒ Array(y)).toArray): _*).data)
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
              .zip(checkpoint.eval(testObj.data).data.head.getData.map(_ * 100.0)).mkString(", ")
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




object BicubicDiscriminatorModel extends Report {

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒ new UpsamplingOptimizer(source, server, out).run()
      case _ ⇒ new UpsamplingOptimizer("E:\\testImages\\256_ObjectCategories", server, out).run()
    })

  }

}
