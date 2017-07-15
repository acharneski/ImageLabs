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

import java.awt.image.BufferedImage
import java.awt.{Graphics2D, RenderingHints}
import java.io.{File, FileInputStream}
import java.nio.file.Path
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

import _root_.util._
import com.google.gson.{GsonBuilder, JsonObject}
import com.simiacryptus.mindseye.layers.{NNLayer, NNResult}
import com.simiacryptus.mindseye.layers.activation.{HyperbolicActivationLayer, LinearActivationLayer, SoftmaxActivationLayer, SqActivationLayer}
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer
import com.simiacryptus.mindseye.layers.media.{AvgSubsampleLayer, ImgBandBiasLayer, ImgConvolutionSynapseLayer, MaxSubsampleLayer}
import com.simiacryptus.mindseye.layers.reducers.{SumInputsLayer, SumReducerLayer}
import com.simiacryptus.mindseye.layers.synapse.{BiasLayer, DenseSynapseLayer}
import com.simiacryptus.mindseye.layers.util.{MonitoringSynapse, MonitoringWrapper}
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch
import com.simiacryptus.mindseye.opt.orient.{GradientDescent, LBFGS, MomentumStrategy, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region._
import com.simiacryptus.mindseye.opt.trainable.{ArrayTrainable, LinkedExampleArrayTrainable, ScheduledSampleTrainable}
import com.simiacryptus.util.StreamNanoHTTPD
import com.simiacryptus.util.io.HtmlNotebookOutput
import com.simiacryptus.util.ml.Tensor
import com.simiacryptus.util.test.ImageTiles.ImageTensorLoader
import com.simiacryptus.util.test.LabeledObject
import com.simiacryptus.util.text.TableOutput
import org.apache.commons.io.IOUtils
import org.apache.spark.sql.SparkSession
import util.Java8Util._

import scala.collection.JavaConverters._
import scala.util.Random


object ImageCorruptionModeler extends ServiceNotebook {

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒ new ImageCorruptionModeler(source, server, out).run()
      case _ ⇒ new ImageCorruptionModeler("E:\\testImages\\256_ObjectCategories", server, out).run()
    })

  }

  def runSpark(masterUrl: String): Unit = {
    var builder = SparkSession.builder
      .appName("Spark MindsEye Demo")
    builder = masterUrl match {
      case "auto" ⇒ builder
      case _ ⇒ builder.master(masterUrl)
    }
    val sparkSession = builder.getOrCreate()
    sparkSession.stop()
  }


}

class ImageCorruptionModeler(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  val corruptors = {
    def resize(source: BufferedImage, size: Int) = {
      val image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB)
      val graphics = image.getGraphics.asInstanceOf[Graphics2D]
      graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
      graphics.drawImage(source, 0, 0, size, size, null)
      image
    }
    Map[String, Tensor ⇒ Tensor](
      "noise" → (imgTensor ⇒ {
        imgTensor.map(Java8Util.cvt((x:Double)⇒Math.min(Math.max(x+(50.0*(Random.nextDouble()-0.5)), 0.0), 256.0)))
      }), "resample" → (imgTensor ⇒ {
        Tensor.fromRGB(resize(resize(imgTensor.toRgbImage, 8), 32))
      })
    )
  }

  val outputSize = Array[Int](3)
  val sampleTiles = 10000
  val iterationCounter = new AtomicInteger(0)
  val original = "original"

  lazy val (categories: Map[String, Int], data: Array[Array[Array[Tensor]]]) = {
    def toOut(label: String, max: Int): Int = {
      (0 until max).find(label == "[" + _ + "]").get
    }
    def toOutNDArray(out: Int, max: Int): Tensor = {
      val ndArray = new Tensor(max)
      ndArray.set(out, 1)
      ndArray
    }

    val labels = List(original) ++ corruptors.keys.toList.sorted
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
    val loader = new ImageTensorLoader(new File(source), 32, 32, 32, 32, 10, 10)
    val unfilteredData = loader.stream().iterator().asScala.take(sampleTiles).toArray
    loader.stop()
    out.p("Preparing training dataset")

    val rawData: List[List[LabeledObject[Tensor]]] = preFilter(unfilteredData).map(tile ⇒ List(
      new LabeledObject[Tensor](tile, original)
    ) ++ corruptors.map(e ⇒ {
      new LabeledObject[Tensor](e._2(tile), e._1)
    })).take(sampleTiles).toList
    out.p("<ol>" + categories.toList.sortBy(_._2).map(x ⇒ "<li>" + x + "</li>").mkString("\n") + "</ol>")
    val data: Array[Array[Array[Tensor]]] = rawData.map(rawData⇒rawData.map((labeledObj: LabeledObject[Tensor]) ⇒ {
      Array(labeledObj.data, toOutNDArray(categories(labeledObj.label), categories.size))
    }).toArray).toArray
    out.eval {
      TableOutput.create(rawData.flatten.take(100).map(testObj ⇒ Map[String, AnyRef](
        "Image" → out.image(testObj.data.toRgbImage(), testObj.data.toString),
        "Label" → testObj.label,
        "Categorization" → categories.toList.sortBy(_._2).map(_._1)
          .zip(getModelCheckpoint.eval(testObj.data).data.head.getData.map(_ * 100.0)).mkString(", ")
      ).asJava): _*)
    }
    out.p("Loading data complete")
    (categories, data)
  }

  def run(): Unit = {
    defineHeader()
    declareTestHandler()
    out.out("<hr/>")
    if(findFile("classifier_initialized").isEmpty) {
      step1()
    }
    if(findFile("classifier_trained").isEmpty) {
      step2()
    }
    step3()
    adversarialTraining()
    profit()
    waitForExit()
  }

  def step1() = phase({
    var network: PipelineNetwork = new PipelineNetwork
    network.add(new MonitoringWrapper(new ImgBandBiasLayer(3).setName("inbias")).addTo(monitoringRoot))
    network.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(5, 5, 36)
      .setWeights(Java8Util.cvt(() ⇒ 0.1 * (Random.nextDouble() - 0.5))).setName("synapse1")).addTo(monitoringRoot))
    network.add(new MonitoringWrapper(new HyperbolicActivationLayer().setScale(0.01).setName("hypr1")).addTo(monitoringRoot))
    network.add(new MonitoringWrapper(new MaxSubsampleLayer(2, 2, 1).setName("max1")).addTo(monitoringRoot))
    //network.add(NetworkMetaNormalizers.scaleNormalizer2)
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "output1"))

    // 16 * 16 * 12
    network.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(5, 5, 240)
      .setWeights(Java8Util.cvt(() ⇒ 0.05 * (Random.nextDouble() - 0.5))).setName("synapse2")).addTo(monitoringRoot))
    network.add(new MonitoringWrapper(new HyperbolicActivationLayer().setScale(0.01).setName("hypr2")).addTo(monitoringRoot))
    network.add(new MonitoringWrapper(new MaxSubsampleLayer(2, 2, 1).setName("max2")).addTo(monitoringRoot))
    //network.add(NetworkMetaNormalizers.scaleNormalizer2)
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "output2"))

    // 8 * 8 * 20
    network.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(5, 5, 400)
      .setWeights(Java8Util.cvt(() ⇒ 0.05 * (Random.nextDouble() - 0.5))).setName("synapse3")).addTo(monitoringRoot))
    network.add(new MonitoringWrapper(new HyperbolicActivationLayer().setScale(0.01).setName("hypr3")).addTo(monitoringRoot))
    network.add(new MonitoringWrapper(new MaxSubsampleLayer(2, 2, 1).setName("max3")).addTo(monitoringRoot))
    //network.add(NetworkMetaNormalizers.scaleNormalizer2)
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "output3"))

    // 4 * 4 * 20
    network.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(3, 3, 400)
      .setWeights(Java8Util.cvt(() ⇒ 0.05 * (Random.nextDouble() - 0.5))).setName("synapse4")).addTo(monitoringRoot))
    network.add(new MonitoringWrapper(new HyperbolicActivationLayer().setScale(0.01).setName("hypr4")).addTo(monitoringRoot))
    network.add(new MonitoringWrapper(new MaxSubsampleLayer(2, 2, 1).setName("max4")).addTo(monitoringRoot))
    //network.add(NetworkMetaNormalizers.positionNormalizer2)
    //network.add(NetworkMetaNormalizers.scaleNormalizer2)
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "output4"))

    // 2 * 2 * 20
    network.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(3, 3, 800)
      .setWeights(Java8Util.cvt(() ⇒ 0.05 * (Random.nextDouble() - 0.5))).setName("synapse5")).addTo(monitoringRoot))
    network.add(new MonitoringWrapper(new HyperbolicActivationLayer().setScale(0.01).setName("hypr5")).addTo(monitoringRoot))
    network.add(new MonitoringWrapper(new MaxSubsampleLayer(2, 2, 1).setName("max5")).addTo(monitoringRoot))
    //network.add(NetworkMetaNormalizers.scaleNormalizer2)
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "output5"))

    // 1 * 1 * 40
    network.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(1, 1, 40 * outputSize(0))
      .setWeights(Java8Util.cvt(() ⇒ 0.1 * (Random.nextDouble() - 0.5))).setName("synapseF")).addTo(monitoringRoot))
    network.add(new MonitoringWrapper(new HyperbolicActivationLayer().setScale(0.01).setName("hyprF")).addTo(monitoringRoot))
    val outputBias = new BiasLayer(outputSize: _*)
    network.add(new MonitoringWrapper(outputBias.setName("biasF")).addTo(monitoringRoot))

    network.add(new MonitoringWrapper(new LinearActivationLayer().setScale(0.001).setName("OutputLinear")).addTo(monitoringRoot))
    network.add(new MonitoringSynapse().addTo(monitoringRoot, "outputF"))
    network.add(new SoftmaxActivationLayer)
    network.asInstanceOf[NNLayer]
  }, (model: NNLayer) ⇒ {
    out.h2("Step 1")
    val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
    val executorFunction = new LinkedExampleArrayTrainable(data, trainingNetwork, 50)
    val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(executorFunction)
      .setCurrentIteration(iterationCounter)
      .setIterationsPerSample(1)
    trainer.setOrientation(new TrustRegionStrategy(new GradientDescent) {
      override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
        case _: DenseSynapseLayer ⇒ new MeanVarianceGradient
        case _: ImgConvolutionSynapseLayer ⇒ new MeanVarianceGradient
        case _ ⇒ null
      }
    })
    trainer.setMonitor(monitor)
    trainer.setTimeout(1, TimeUnit.HOURS)
    trainer.setTerminateThreshold(0.0)
    trainer.setMaxIterations(10)
    require(trainer.run() < 2.0)
  }: Unit, "classifier_initialized")

  def step2() = phase("classifier_initialized", (model: NNLayer) ⇒ {
    out.h2("Step 2")
    val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
    val executorFunction = new LinkedExampleArrayTrainable(data, trainingNetwork, 100)
    val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(executorFunction)
      .setIterationsPerSample(1)
    trainer.setLineSearchFactory(Java8Util.cvt(()⇒new ArmijoWolfeSearch().setC1(1e-4).setC2(0.7)))
    trainer.setOrientation(new TrustRegionStrategy(new MomentumStrategy(new GradientDescent()).setCarryOver(0.2)) {
      override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
        case _: DenseSynapseLayer ⇒ new LinearSumConstraint
        case _: ImgConvolutionSynapseLayer ⇒ null
        case _ ⇒ new StaticConstraint
      }
    })
    trainer.setMonitor(monitor)
    trainer.setTimeout(2, TimeUnit.HOURS)
    trainer.setTerminateThreshold(0.0)
    trainer.setMaxIterations(5000)
    trainer.run()
  }: Unit, "classifier_trained")

  def step3() = phase("classifier_trained", (model: NNLayer) ⇒ {
    out.h2("Step 3")
    val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
    val executorFunction = new LinkedExampleArrayTrainable(data, trainingNetwork, 500)
    val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(executorFunction)
      .setIterationsPerSample(50)
    trainer.setLineSearchFactory(Java8Util.cvt(()⇒new ArmijoWolfeSearch().setC1(1e-6).setC2(0.9)))
    trainer.setOrientation(new TrustRegionStrategy(new LBFGS().setMaxHistory(30)) {
      override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
        case _ ⇒ null
      }
    })
    trainer.setMonitor(monitor)
    trainer.setTimeout(2, TimeUnit.HOURS)
    trainer.setTerminateThreshold(0.0)
    trainer.setMaxIterations(5000)
    trainer.run()
  }: Unit, "classifier_trained")

  def adversarialTraining() = phase("classifier_trained", (model: NNLayer) ⇒ {
    Random.shuffle(data.map(_.head.toList.toArray).toList).grouped(10).zipWithIndex.take(20).foreach(x⇒{
      val (srcimages,i) = x
      out.h2(s"Adversarial Training $i")
      val adversarialTrainingSet: Array[Array[Array[Tensor]]] = srcimages.map((testObj: Array[Tensor]) ⇒ {
        (corruptors.map(x ⇒{
          val (name,fn) = x
          val image = fn.apply(testObj(0))
          val adversarial = buildAdversarialImage(model, image, original, 0.8)
          Array(adversarial._2, new Tensor(categories.size).set(categories(name), 1.0))
        }) ++ corruptors.map(x ⇒{
          val (name,fn) = x
          val image = testObj(0)
          val adversarial = buildAdversarialImage(model, image, name, 0.8)
          Array(adversarial._2, new Tensor(categories.size).set(categories(original), 1.0))
        })).toArray.filterNot((tensors: Array[Tensor]) ⇒ tensors.isEmpty)
      }).toArray.filterNot((tensors: Array[Array[Tensor]]) ⇒ tensors.isEmpty)
      out.eval {
        TableOutput.create(adversarialTrainingSet.flatMap(
          _.map((testObj: Array[Tensor]) ⇒ {
            Map[String, AnyRef](
              "Image" → out.image(testObj(0).toRgbImage(), "Image"),
              "Class" → testObj(1),
              "Predicted" → categories.toList.sortBy(_._2).map(_._1)
                .zip(getModelCheckpoint.eval(testObj(0)).data.head.getData.map(_ * 100.0))
            ).asJava
          })
        ).take(20): _*)
      }
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
      val executorFunction = new ArrayTrainable(adversarialTrainingSet.flatten, trainingNetwork)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(executorFunction)
      trainer.setLineSearchFactory(Java8Util.cvt(()⇒new ArmijoWolfeSearch().setC1(1e-6).setC2(0.99)))
      trainer.setOrientation(new TrustRegionStrategy(new LBFGS) {
        override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
          case _: DenseSynapseLayer ⇒ new LinearSumConstraint
          case _: ImgConvolutionSynapseLayer ⇒ null
          case _: LinearActivationLayer ⇒ new StaticConstraint
          case _: BiasLayer ⇒ new StaticConstraint
          case _ ⇒ new StaticConstraint
        }
      })
      trainer.setMonitor(monitor)
      trainer.setTimeout(1, TimeUnit.HOURS)
      trainer.setTerminateThreshold(1.0)
      trainer.setMaxIterations(100)
      trainer.run()
    })
  }: Unit, "classifier_hardened")

  def resize(source: BufferedImage, size: Int) = {
    val image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB)
    val graphics = image.getGraphics.asInstanceOf[Graphics2D]
    graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
    graphics.drawImage(source, 0, 0, size, size, null)
    image
  }

  def profit() = phase("classifier_hardened", (model: NNLayer) ⇒ testReconstruction(out, model): Unit)

  def testReconstruction(out: HtmlNotebookOutput with ScalaNotebookOutput, model: NNLayer) = {
    out.h2("Test Reconstruction")
    out.eval {
      TableOutput.create(Random.shuffle(data.map(_.head).toList).take(10).map(testObj ⇒ {
        val corruptedImage = resize(resize(testObj(0).toRgbImage, 8), 32)
        val corruptedData = Tensor.fromRGB(corruptedImage)
        val reconstructed = reconstructImage(model, corruptedData, original).toRgbImage
        Map[String, AnyRef](
          "Original Image" → out.image(testObj(0).toRgbImage(), "Original Image"),
          "Corrupted" → out.image(corruptedImage, "Corrupted Image"),
          "Reconstructed" → out.image(reconstructed, "Reconstructed Image")
        ).asJava
      }): _*)
    }
  }

  def buildAdversarialImage(model: NNLayer, data: Tensor, targetCategory: String = original, certianty: Double): (Double, Tensor) = {
    val trainableNet = new PipelineNetwork()
    val imageCorrections = new BiasLayer(32, 32, 3)
    val result = trainableNet.add(model)
    val tensor = new Tensor(categories.size)
    tensor.setAll((1-certianty)/(categories.size-1))
    tensor.set(categories(targetCategory), certianty)
    val goal = trainableNet.constValue(tensor)
    val entropyNode = trainableNet.add(new EntropyLossLayer(), result, goal)
    trainableNet.add(imageCorrections, trainableNet.constValue(new Tensor(32,32,3)))
    trainableNet.add(new SqActivationLayer())
    trainableNet.add(new SumReducerLayer())
    trainableNet.add(new LinearActivationLayer().setScale(0.1))
    trainableNet.add(new SumInputsLayer(), trainableNet.getHead, entropyNode)
    val executorFunction = new ArrayTrainable(Array(Array(data)), trainableNet)
    val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(executorFunction)
    trainer.setLineSearchFactory(Java8Util.cvt(() ⇒ new ArmijoWolfeSearch().setC1(1e-6).setC2(0.9)))
    trainer.setOrientation(new TrustRegionStrategy(new GradientDescent) {
      override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
        case l: BiasLayer if l.id.equals(imageCorrections.getId) ⇒ new TrustRegion() {
          override def project(history: Array[Array[Double]], point: Array[Double]): Array[Double] = {
            point.zipWithIndex.map(x⇒{
              val v = data.get(x._2)
              Math.min(Math.max(x._1, - v), 256 - v)
            }).toList.toArray
          }
        }
        case _ ⇒ new StaticConstraint
      }
    })
    trainer.setMonitor(new TrainingMonitor{
      override def log(msg: String): Unit = println(msg)
    })
    trainer.setTimeout(1, TimeUnit.MINUTES)
    trainer.setTerminateThreshold(0.5)
    trainer.setMaxIterations(20)
    trainer.run() → imageCorrections.eval(data).data.head
  }

  def reconstructImage(model: NNLayer, data: Tensor, targetCategory: String): Tensor = {
    val trainableNet = new PipelineNetwork()
    val imageCorrections = new BiasLayer(32, 32, 3)
    trainableNet.add(model)
    val tensor = new Tensor(categories.size)
    tensor.set(categories(targetCategory), 1)
    val goal = trainableNet.constValue(tensor)
    val entropyNode = trainableNet.add(new EntropyLossLayer(), trainableNet.getHead, goal)
    trainableNet.add(imageCorrections, trainableNet.constValue(new Tensor(32,32,3)))
    trainableNet.add(new SqActivationLayer())
    trainableNet.add(new SumReducerLayer())
    trainableNet.add(new LinearActivationLayer().setScale(0.1))
    trainableNet.add(new SumInputsLayer(), trainableNet.getHead, entropyNode)
    val executorFunction = new ArrayTrainable(Array(Array(data)), trainableNet)
    val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(executorFunction)
    trainer.setLineSearchFactory(Java8Util.cvt(() ⇒ new ArmijoWolfeSearch().setC1(1e-6).setC2(0.9)))
    trainer.setOrientation(new TrustRegionStrategy(new LBFGS) {
      override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
        case l: BiasLayer if l.id.equals(imageCorrections.getId) ⇒ new TrustRegion() {
          override def project(history: Array[Array[Double]], point: Array[Double]): Array[Double] = {
            point.zipWithIndex.map(x⇒{
              val v = data.get(x._2)
              Math.min(Math.max(x._1, - v), 256 - v)
            }).toList.toArray
          }
        }
        case _ ⇒ new StaticConstraint
      }
    })
    trainer.setMonitor(new TrainingMonitor{
      override def log(msg: String): Unit = println(msg)
    })
    trainer.setTimeout(1, TimeUnit.MINUTES)
    trainer.setTerminateThreshold(0.0)
    trainer.setMaxIterations(100)
    trainer.run()
    imageCorrections.eval(data).data.head
  }

  def declareTestHandler() = {
    out.p("<a href='testCat.html'>Test Categorization</a><br/>")
    server.addSyncHandler("testCat.html", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(out ⇒ {
        testCategorization(out, getModelCheckpoint)
      })
    }), false)
    out.p("<a href='testRes.html'>Test Restoration</a><br/>")
    server.addSyncHandler("testRes.html", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(out ⇒ {
        testReconstruction(out, getModelCheckpoint)
      })
    }), false)
  }

  private def testCategorization(out: HtmlNotebookOutput with ScalaNotebookOutput, model : NNLayer) = {
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
}