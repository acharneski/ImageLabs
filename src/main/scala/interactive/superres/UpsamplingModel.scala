/*
 * Copyright (c) 2018 by Andrew Charneski.
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

///*
// * Copyright (c) 2017 by Andrew Charneski.
// *
// * The author licenses this file to you under the
// * Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance
// * with the License.  You may obtain a copy
// * of the License at
// *
// *   http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing,
// * software distributed under the License is distributed on an
// * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// * KIND, either express or implied.  See the License for the
// * specific language governing permissions and limitations
// * under the License.
// */
//
//package interactive.superres
//
//import java.awt.image.BufferedImage
//import java.awt.{Graphics2D, RenderingHints}
//import java.io._
//import java.lang
//import java.util.concurrent.TimeUnit
//import java.util.function.{DoubleSupplier, IntToDoubleFunction}
//
//import _root_.util.Java8Util.cvt
//import _root_.util._
//import com.simiacryptus.mindseye.eval._
//import com.simiacryptus.mindseye.lang.{Coordinate, NNExecutionContext, LayerBase, Tensor}
//import com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer
//import com.simiacryptus.mindseye.layers.java._
//import com.simiacryptus.mindseye.network.{DAGNode, PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
//import com.simiacryptus.mindseye.opt._
//import com.simiacryptus.mindseye.opt.line._
//import com.simiacryptus.mindseye.opt.orient._
//import com.simiacryptus.mindseye.eval.data.ImageTiles.ImageTensorLoader
//import com.simiacryptus.util.io.HtmlNotebookOutput
//import com.simiacryptus.util.{MonitoredObject, StreamNanoHTTPD, TableOutput, Util}
//import util.NNLayerUtil._
//
//import scala.collection.JavaConverters._
//import scala.util.Random
//
//case class DeepNetworkUpsample(
//                                     weight1 : Double,
//                                     weight2 : Double,
//                                     weight3 : Double,
//                                     weight4 : Double
//                                   ) {
//
//  def getNetwork(monitor: TrainingMonitor,
//                 monitoringRoot : MonitoredObject,
//                 fitness : Boolean = false) : LayerBase = {
//    val parameters = this
//    var network: PipelineNetwork = if(fitness) {
//      new PipelineNetwork(2)
//    } else {
//      new PipelineNetwork(1)
//    }
//    val zeroSeed : IntToDoubleFunction = Java8Util.cvt(_ ⇒ 0.0)
//    def buildLayer(from: Int,
//                   to: Int,
//                   layerNumber: String,
//                   weights: Double,
//                   layerRadius: Int = 3,
//                   activationLayer: LayerBase = new ReLuActivationLayer()) = {
//      def weightSeed : DoubleSupplier = Java8Util.cvt(() ⇒ {
//        val r = Util.R.get.nextDouble() * 2 - 1
//        r * weights
//      })
//      network.add(new ImgBandBiasLayer(from).setWeights(zeroSeed).setName("bias_" + layerNumber).addTo(monitoringRoot))
//      if (null != activationLayer) {
//        network.add(activationLayer.setName("activation_" + layerNumber).freeze.addTo(monitoringRoot))
//      }
//      network.add(new ConvolutionLayer(layerRadius, layerRadius, from * to).setWeights(weightSeed).setName("conv_" + layerNumber).addTo(monitoringRoot));
//      //network.fn(new MonitoringSynapse().addTo(monitoringRoot).setName("output_" + layerNumber))
//    }
//
//    val l1 = buildLayer(3, 64, "0", weights = Math.pow(10, parameters.weight1), activationLayer = null)
//    val l2 = buildLayer(64, 64, "1", weights = Math.pow(10, parameters.weight2), activationLayer = new HyperbolicActivationLayer().setScale(5).freeze())
//    network.add(new ImgReshapeLayer(2,2,true))
//    val l3 = buildLayer(16, 12, "2", weights = Math.pow(10, parameters.weight3), activationLayer = new HyperbolicActivationLayer().setScale(5).freeze())
//    buildLayer(12, 12, "3", weights = Math.pow(10, parameters.weight4), activationLayer = new HyperbolicActivationLayer().setScale(5).freeze())
//    network.add(new ImgReshapeLayer(2,2,true))
//
//    if(fitness) {
//      val output = network.getHead
//      def normalizeStdDev(layer:DAGNode, target:Double) = network.add(new AbsActivationLayer(), network.add(new SumInputsLayer(),
//                network.add(new AvgReducerLayer(), network.add(new StdDevMetaLayer(), layer)),
//                network.add(new ConstLayer(new Tensor(1).setBytes(0,-target)))
//              ))
//      network.add(new ProductInputsLayer(), network.add(new MeanSqLossLayer(), output, network.getInput(1)), network.add(new SumInputsLayer(),
//                network.add(new ConstLayer(new Tensor(1).setBytes(0,1))),
//                normalizeStdDev(l1,16),
//                normalizeStdDev(l2,16),
//                normalizeStdDev(l3,16)
//              ))
//    }
//
//    network
//  }
//
//  def fitness(monitor: TrainingMonitor, monitoringRoot : MonitoredObject, data: Array[Array[Tensor]], n: Int = 2) : Double = {
//    val values = (1 to n).map(i ⇒ {
//      val network = getNetwork(monitor, monitoringRoot, fitness = true)
//      val measure = new ArrayTrainable(data, network).measure(monitor)
//      measure.sum
//    }).toList
//    val avg = values.sum / n
//    monitor.log(s"Numeric Opt: $this => $avg ($values)")
//    avg
//  }
//
//}
//
//
//class UpsamplingModel(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {
//
//  val modelName = System.getProperty("modelName","upsample_1")
//  val tileSize = 128
//  val fitnessBorderPadding = 8
//  val scaleFactor: Double = (64 * 64.0) / (tileSize * tileSize)
//
//  def eval(awaitExit:Boolean=true): Unit = {
//    defineHeader()
//    defineTestHandler()
//    out.out("<hr/>")
//    if(findFile(modelName).isEmpty || System.getProperties.containsKey("rebuild")) step_Generate()
//    //step_SGD((100 * scaleFactor).toInt, 30, reshufflePeriod = 5)
//    def rates = step_diagnostics_layerRates().map(e⇒e._1.getName→e._2.rate)
//    step_LBFGS((100 * scaleFactor).toInt, 30, 50)
//    step_SGD((100 * scaleFactor).toInt, 30, reshufflePeriod = 5) // , rates = rates)
//    step_LBFGS((500 * scaleFactor).toInt, 60, 50)
//    step_SGD((500 * scaleFactor).toInt, 60, reshufflePeriod = 5) // , rates = rates)
//    out.out("<hr/>")
//    if(awaitExit) waitForExit()
//  }
//
//  def resize(source: BufferedImage, size: Int) = {
//    val image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB)
//    val graphics = image.getGraphics.asInstanceOf[Graphics2D]
//    graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
//    graphics.drawImage(source, 0, 0, size, size, null)
//    image
//  }
//
//  lazy val data : List[Array[Tensor]] = {
//    out.p("Loading data from " + source)
//    val rawList: List[Tensor] = rawData
//    System.gc()
//    val data: List[Array[Tensor]] = rawList.map(tile ⇒ Array(Tensor.fromRGB(resize(tile.toRgbImage, tileSize/4)), tile))
//    out.eval {
//      TableOutput.create(Random.shuffle(data).take(100).map(testObj ⇒ Map[String, AnyRef](
//        "Source" → out.image(testObj(1).toRgbImage(), ""),
//        "Resized" → out.image(testObj(0).toRgbImage(), "")
//      ).asJava): _*)
//    }
//    out.p("Loading data complete")
//    data
//  }
//
//  private def rawData() = {
//    val loader = new ImageTensorLoader(new File(source), tileSize, tileSize, tileSize, tileSize, 10, 10)
//    val rawList = loader.stream().iterator().asScala.take((10000 * scaleFactor).toInt).toList
//    loader.stop()
//    rawList
//  }
//
//
//  def step_Generate() = {
//    phase({
//      val optTraining: Array[Array[Tensor]] = Random.shuffle(data).take((10 * scaleFactor).ceil.toInt).toArray
//      SimplexOptimizer[DeepNetworkUpsample](
//        DeepNetworkUpsample(-0.19628151652396514,-1.120332072478063,-1.5337950986957058,-1.5337950986957058),
//        x ⇒ x.fitness(monitor, monitoringRoot, optTraining, n=2), relativeTolerance=0.15
//      ).getNetwork(monitor, monitoringRoot)
//    }, (model: LayerBase) ⇒ {
//      out.h1("Step 1")
//      monitor.clear()
//      val trainer = out.eval {
//        val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, lossNetwork)
//        val dataArray = data.toArray
//        var heapCopy: Trainable = new SampledArrayTrainable(dataArray, trainingNetwork, (50 * scaleFactor).toInt)
//        //heapCopy = new ConstL12Normalizer(heapCopy).setFactor_L1(0.001)
//        val trainer = new IterativeTrainer(heapCopy)
//        trainer.setMonitor(monitor)
//        trainer.setTimeout(15, TimeUnit.MINUTES)
//        trainer.setIterationsPerSample(1)
//        val momentum = new GradientDescent()
//        trainer.setOrientation(momentum)
//        trainer.setLineSearchFactory(Java8Util.cvt((s) ⇒ new QuadraticSearch))
//        trainer.setTerminateThreshold(2500.0)
//        trainer
//      }
//      trainer.eval()
//    }: Unit, modelName)
//  }
//
//  def step_diagnostics_layerRates(sampleSize : Int = (100 * scaleFactor).toInt) = phase[Map[LayerBase, LayerRateDiagnosticTrainer.LayerStats]](modelName, (model: LayerBase) ⇒ {
//    monitor.clear()
//    val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, lossNetwork)
//    val dataArray = data.toArray
//    out.h1("Diagnostics - LayerBase Rates")
//    val result = out.eval {
//      var heapCopy: Trainable = new SampledArrayTrainable(dataArray, trainingNetwork, sampleSize)
//      val trainer = new LayerRateDiagnosticTrainer(heapCopy).setStrict(true).setMaxIterations(1)
//      trainer.setMonitor(monitor)
//      trainer.eval()
//      trainer.getLayerRates().asScala.toMap
//    }
//    result
//  }, modelName)
//
//  def step_SGD(sampleSize: Int, timeoutMin: Int, termValue: Double = 0.0, momentum: Double = 0.2, maxIterations: Int = Integer.MAX_VALUE, reshufflePeriod: Int = 1,rates: Map[String, Double] = Map.empty) = phase(modelName, (model: LayerBase) ⇒ {
//    monitor.clear()
//    out.h1(s"SGD(sampleSize=$sampleSize,timeoutMin=$timeoutMin)")
//    val trainer = out.eval {
//      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, lossNetwork)
//      val dataArray = data.toArray
//      var heapCopy: Trainable = new SampledArrayTrainable(dataArray, trainingNetwork, sampleSize)
//      //heapCopy = new ConstL12Normalizer(heapCopy).setFactor_L1(0.001)
//      val trainer = new IterativeTrainer(heapCopy)
//      trainer.setMonitor(monitor)
//      trainer.setTimeout(timeoutMin, TimeUnit.MINUTES)
//      trainer.setIterationsPerSample(reshufflePeriod)
//      val momentumStrategy = new MomentumStrategy(new GradientDescent()).setCarryOver(momentum)
//      val reweight = new LayerReweightingStrategy(momentumStrategy) {
//        override def getRegionPolicy(layer: LayerBase): lang.Double = layer.getName match {
//          case key if rates.contains(key) ⇒ rates(key)
//          case _ ⇒ 1.0
//        }
//        override def reset(): Unit = {}
//      }
//      trainer.setOrientation(reweight)
//      trainer.setLineSearchFactory(Java8Util.cvt((s)⇒new ArmijoWolfeSearch().setAlpha(1e-12).setC1(0).setC2(1)))
//      //trainer.setLineSearchFactory(Java8Util.cvt((s)⇒new BisectionSearch()))
//      trainer.setTerminateThreshold(termValue)
//      trainer.setMaxIterations(maxIterations)
//      trainer
//    }
//    val result = trainer.eval()
//    result
//  }, modelName)
//
//  def step_LBFGS(sampleSize: Int, timeoutMin: Int, iterationSize: Int): Unit = phase(modelName, (model: LayerBase) ⇒ {
//    monitor.clear()
//    out.h1(s"LBFGS(sampleSize=$sampleSize,timeoutMin=$timeoutMin)")
//    val trainer = out.eval {
//      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, lossNetwork)
//      val heapCopy = new SampledArrayTrainable(data.toArray, trainingNetwork, sampleSize)
//      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(heapCopy)
//      trainer.setMonitor(monitor)
//      trainer.setTimeout(timeoutMin, TimeUnit.MINUTES)
//      trainer.setIterationsPerSample(iterationSize)
//      val lbfgs = new LBFGS().setMaxHistory(35).setMinHistory(4)
//      trainer.setOrientation(lbfgs)
//      trainer.setLineSearchFactory(Java8Util.cvt((s:String)⇒(s match {
//        case s if s.contains("LBFGS") ⇒ new StaticLearningRate().setRate(1.0)
//        case _ ⇒ new ArmijoWolfeSearch().setAlpha(1e-5)
//      })))
//      trainer.setTerminateThreshold(0.0)
//      trainer
//    }
//    trainer.eval()
//  }, modelName)
//
//  def lossNetwork = {
//    val mask: Tensor = new Tensor(tileSize, tileSize, 3).mapCoords(Java8Util.cvt((c: Coordinate) ⇒ {
//      if (c.coords(0) < fitnessBorderPadding || c.coords(0) >= (tileSize - fitnessBorderPadding)) {
//        0.0
//      } else if (c.coords(1) < fitnessBorderPadding || c.coords(1) >= (tileSize - fitnessBorderPadding)) {
//        0.0
//      } else {
//        1.0
//      }
//    }))
//    val lossNetwork = new PipelineNetwork(2)
//    val maskNode = lossNetwork.add(new ConstLayer(mask).freeze())
//    lossNetwork.add(new MeanSqLossLayer(), lossNetwork.add(new ProductInputsLayer(), lossNetwork.getInput(0), maskNode), lossNetwork.add(new ProductInputsLayer(), lossNetwork.getInput(1), maskNode))
//    lossNetwork
//  }
//
//  def defineTestHandler() = {
//    out.p("<a href='eval.html'>Test Reconstruction</a>")
//    server.addSyncHandler("eval.html", "text/html", cvt(o ⇒ {
//      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(out ⇒ {
//        try {
//          out.eval {
//            TableOutput.create(Random.shuffle(data).take(100).map(testObj ⇒ Map[String, AnyRef](
//              "Source Truth" → out.image(testObj(1).toRgbImage(), ""),
//              "Corrupted" → out.image(testObj(0).toRgbImage(), ""),
//              "Reconstruction" → out.image(getModelCheckpoint.eval(new NNExecutionContext() {}, testObj(0)).getData.get(0).toRgbImage(), "")
//            ).asJava): _*)
//          }
//        } catch {
//          case e: Throwable ⇒ e.printStackTrace()
//        }
//      })
//    }), false)
//  }
//
//}
//
//object UpsamplingModel extends Report {
//
//  def main(args: Array[String]): Unit = {
//
//    report((server, out) ⇒ args match {
//      case Array(source) ⇒ new UpsamplingModel(source, server, out).eval()
//      case _ ⇒ new UpsamplingModel("E:\\testImages\\256_ObjectCategories", server, out).eval()
//    })
//
//  }
//
//}
