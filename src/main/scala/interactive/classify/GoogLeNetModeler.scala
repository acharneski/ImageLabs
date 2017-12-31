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

package interactive.classify

import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage
import java.awt.{Graphics2D, RenderingHints}
import java.io._
import java.util.concurrent.TimeUnit
import java.util.function.Supplier
import javax.imageio.ImageIO

import _root_.util.Java8Util.cvt
import _root_.util._
import com.simiacryptus.mindseye.eval.{ArrayTrainable, SampledArrayTrainable, Trainable}
import com.simiacryptus.mindseye.lang._
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer.PoolingMode
import com.simiacryptus.mindseye.layers.cudnn._
import com.simiacryptus.mindseye.layers.java.{ImgBandBiasLayer => _, ImgConcatLayer => _, ProductLayer => _, _}
import com.simiacryptus.mindseye.network.{DAGNetwork, DAGNode, PipelineNetwork, SimpleLossNetwork}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line._
import com.simiacryptus.mindseye.opt.orient._
import com.simiacryptus.util.function.{SoftCachedSupplier, WeakCachedSupplier}
import com.simiacryptus.util.io.{HtmlNotebookOutput, KryoUtil}
import com.simiacryptus.util.{MonitoredObject, StreamNanoHTTPD, TableOutput}
import interactive.classify.GoogLeNetModeler.tileSize

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.Random

object GoogLeNetModeler extends Report {
  val modelName = System.getProperty("modelName", "googlenet_1")
  val tileSize = 224
  val categoryWhitelist = Set[String]()
  //("greyhound", "soccer-ball", "telephone-box", "windmill")
  val numberOfCategories = 256 // categoryWhitelist.size
  val imagesPerCategory = 500
  val fuzz = 1e-4
  val artificialVariants = 10

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒ new GoogLeNetModeler(source, server, out).run()
      case _ ⇒ new GoogLeNetModeler("D:\\testImages\\256_ObjectCategories", server, out).run()
    })

  }

}

import interactive.classify.GoogLeNetModeler._

case class GoogLeNet(
                      layer1: Double = -2.17,
                      layer2: Double = -1.8,
                      layer3: Double = -2.76,
                      layer4: Double = 0,
                      layer5: Double = -0.125,
                      layer6: Double = -0.398,
                      layer7: Double = -0.426,
                      layer8: Double = -0.426,
                      layer9: Double = -0.426,
                      layer10: Double = -0.439,
                      layer11: Double = -0.64,
                      layer12: Double = -0.64,
                      layer13: Double = -4.7,
                      conv1a: Double = -2.28,
                      conv3a: Double = -2.28,
                      conv3b: Double = -2.94,
                      conv5a: Double = -2.28,
                      conv5b: Double = -2.60,
                      conv1b: Double = -2.28,
                      trainingShunt: Double = -2
                    ) {
import NNLayerUtil._
  def getNetwork(monitor: TrainingMonitor,
                 monitoringRoot: MonitoredObject,
                 fitness: Boolean = false): NNLayer = {
    val network = new PipelineNetwork(2)

    def newInceptionLayer(layerName : String, head: DAGNode = network.getHead, inputBands: Int, bands1x1: Int, bands3x1: Int, bands1x3: Int, bands5x1: Int, bands1x5: Int, bandsPooling: Int): DAGNode = {
      network.add(new ImgConcatLayer(),
        network.addAll(head,
          new ConvolutionLayer(1, 1, inputBands, bands1x1).setWeightsLog(layer11 + conv1a).setName("conv_1x1_" + layerName).addTo(monitoringRoot),
          new ImgBandBiasLayer(bands1x1).setName("bias_1x1_" + layerName).addTo(monitoringRoot),
          new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_1x1_" + layerName).addTo(monitoringRoot)),
        network.addAll(head,
          new ConvolutionLayer(1, 1, inputBands, bands3x1).setWeightsLog(layer11 + conv3a).setName("conv_3x1_" + layerName).addTo(monitoringRoot),
          new ImgBandBiasLayer(bands3x1).setName("bias_3x1_" + layerName).addTo(monitoringRoot),
          new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_3x1_" + layerName).addTo(monitoringRoot),
          new ConvolutionLayer(3, 3, bands3x1, bands1x3).setWeightsLog(layer11 + conv3b).setName("conv_1x3_" + layerName).addTo(monitoringRoot),
          new ImgBandBiasLayer(bands1x3).setName("bias_1x3_" + layerName).addTo(monitoringRoot),
          new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_1x3_" + layerName).addTo(monitoringRoot)),
        network.addAll(head,
          new ConvolutionLayer(1, 1, inputBands, bands5x1).setWeightsLog(layer11 + conv5a).setName("conv_5x1_" + layerName).addTo(monitoringRoot),
          new ImgBandBiasLayer(bands5x1).setName("bias_5x1_" + layerName).addTo(monitoringRoot),
          new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_5x1_" + layerName).addTo(monitoringRoot),
          new ConvolutionLayer(5, 5, bands5x1, bands1x5).setWeightsLog(layer11 + conv5b).setName("conv_1x5_" + layerName).addTo(monitoringRoot),
          new ImgBandBiasLayer(bands1x5).setName("bias_1x5_" + layerName).addTo(monitoringRoot),
          new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_1x5_" + layerName).addTo(monitoringRoot)),
        network.addAll(head,
          new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1).setName("pool_" + layerName).addTo(monitoringRoot),
          new ConvolutionLayer(1, 1, inputBands, bandsPooling).setWeightsLog(layer11 + conv1b).setName("conv_pool_" + layerName).addTo(monitoringRoot),
          new ImgBandBiasLayer(bandsPooling).setName("bias_pool_" + layerName).addTo(monitoringRoot),
          new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_pool_" + layerName).addTo(monitoringRoot)))
    }

    network.add(new AssertDimensionsLayer(224, 224, 3), network.getInput(0))
    network.addAll(
      new ConvolutionLayer(7, 7, 3, 64).setWeightsLog(layer1).setStrideXY(2, 2).setName("conv_1").addTo(monitoringRoot),
      new ImgBandBiasLayer(64).setName("bias_1").addTo(monitoringRoot),
      new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_1").addTo(monitoringRoot),
      new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1).setName("pool_1").addTo(monitoringRoot),
      new ConvolutionLayer(1, 1, 64, 64).setWeightsLog(layer2).setName("conv_2").addTo(monitoringRoot),
      new ImgBandBiasLayer(64).setName("bias_2").addTo(monitoringRoot),
      new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_2").addTo(monitoringRoot),
      new ConvolutionLayer(3, 3, 64, 192).setWeightsLog(layer3).setName("conv_3").addTo(monitoringRoot),
      new ImgBandBiasLayer(192).setName("bias_3").addTo(monitoringRoot),
      new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_3").addTo(monitoringRoot),
      new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1).setName("pool_3").addTo(monitoringRoot))
    val inception_3a = newInceptionLayer(layerName = "3a", inputBands = 192, bands1x1 = 64, bands3x1 = 96, bands1x3 = 128, bands5x1 = 16, bands1x5 = 32, bandsPooling = 32)
    val inception_3b = newInceptionLayer(layerName = "3b", inputBands = 256, bands1x1 = 128, bands3x1 = 128, bands1x3 = 192, bands5x1 = 32, bands1x5 = 96, bandsPooling = 64)
    network.add(new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1).setName("pool_4").addTo(monitoringRoot))
    val inception_4a = newInceptionLayer(layerName = "4a", inputBands = 480, bands1x1 = 192, bands3x1 = 96, bands1x3 = 208, bands5x1 = 16, bands1x5 = 48, bandsPooling = 64)
    val inception_4b = newInceptionLayer(layerName = "4b", inputBands = 512, bands1x1 = 160, bands3x1 = 112, bands1x3 = 224, bands5x1 = 24, bands1x5 = 64, bandsPooling = 64)
    val inception_4c = newInceptionLayer(layerName = "4c", inputBands = 512, bands1x1 = 128, bands3x1 = 128, bands1x3 = 256, bands5x1 = 24, bands1x5 = 64, bandsPooling = 64)
    val inception_4d = newInceptionLayer(layerName = "4d", inputBands = 512, bands1x1 = 112, bands3x1 = 144, bands1x3 = 288, bands5x1 = 32, bands1x5 = 64, bandsPooling = 64)
    val inception_4e = newInceptionLayer(layerName = "4e", inputBands = 528, bands1x1 = 256, bands3x1 = 160, bands1x3 = 320, bands5x1 = 32, bands1x5 = 128, bandsPooling = 128)
    network.add(new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1).setName("pool_5").addTo(monitoringRoot))
    val inception_5a = newInceptionLayer(layerName = "5a", inputBands = 832, bands1x1 = 256, bands3x1 = 160, bands1x3 = 320, bands5x1 = 32, bands1x5 = 128, bandsPooling = 128)
    val inception_5b = newInceptionLayer(layerName = "5b", inputBands = 832, bands1x1 = 384, bands3x1 = 192, bands1x3 = 384, bands5x1 = 48, bands1x5 = 128, bandsPooling = 128)
    val rawCategorization = network.addAll(
      new PoolingLayer().setWindowXY(7, 7).setStrideXY(1, 1).setPaddingXY(0, 0).setMode(PoolingMode.Avg).setName("pool_6").addTo(monitoringRoot),
      new DropoutNoiseLayer().setValue(0.4).setName("dropout_6").addTo(monitoringRoot),
      new FullyConnectedLayer(Array(1024), Array(1024)).setName("syn_6").addTo(monitoringRoot),
      new BiasLayer(1024).setName("bias_6").addTo(monitoringRoot))

    val entropy = network.add(new SumInputsLayer(),
      network.add(new EntropyLossLayer(),
        network.add("classify", new SoftmaxActivationLayer(), rawCategorization),
        network.getInput(1)),
      network.add(new EntropyLossLayer(),
        network.add(new SoftmaxActivationLayer(),
          network.add(new BiasLayer(1024).setName("bias_out3_4a").addTo(monitoringRoot),
            network.add(new FullyConnectedLayer(Array(1024), Array(1024)).setName("syn_out3_4a").addTo(monitoringRoot),
              network.add(new DropoutNoiseLayer().setValue(0.7).setName("dropout_4a").addTo(monitoringRoot),
              network.add(new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_out3_4a").addTo(monitoringRoot),
                network.add(new ImgBandBiasLayer(1024).setName("bias_out2_4a").addTo(monitoringRoot),
                  network.add(new ConvolutionLayer(3, 3, 128, 1024).setWeightsLog(trainingShunt).setName("syn_out2_4a").addTo(monitoringRoot),
                    network.add(new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_out1_4a").addTo(monitoringRoot),
                      network.add(new ConvolutionLayer(1, 1, 512, 128).setWeightsLog(trainingShunt).setName("conv_out1_4a").addTo(monitoringRoot),
                        network.add(new PoolingLayer().setWindowXY(7, 7).setStrideXY(3, 3).setMode(PoolingMode.Avg).setName("pool_out1_4a").addTo(monitoringRoot),
                          inception_4a
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
      ), network.getInput(1)),
      network.add(new EntropyLossLayer(),
        network.add(new SoftmaxActivationLayer(),
          network.add(new BiasLayer(1024).setName("bias_out3_4d").addTo(monitoringRoot),
            network.add(new FullyConnectedLayer(Array(1024), Array(1024)).setName("syn_out3_4d").addTo(monitoringRoot),
              network.add(new DropoutNoiseLayer().setValue(0.7).setName("dropout_4d").addTo(monitoringRoot),
              network.add(new ActivationLayer(ActivationLayer.Mode.RELU).freeze().setName("relu_out3_4d").addTo(monitoringRoot),
                network.add(new ImgBandBiasLayer(1024).setName("bias_out2_4d").addTo(monitoringRoot),
                  network.add(new ConvolutionLayer(3, 3, 128, 1024).setWeightsLog(trainingShunt).setName("syn_out2_4d").addTo(monitoringRoot),
                    network.add(new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_out1_4d").addTo(monitoringRoot),
                      network.add(new ConvolutionLayer(1, 1, 528, 128).setWeightsLog(trainingShunt).setName("conv_out1_4d").addTo(monitoringRoot),
                        network.add(new PoolingLayer().setWindowXY(7, 7).setStrideXY(3, 3).setMode(PoolingMode.Avg).setName("pool_out1_4d").addTo(monitoringRoot),
                          inception_4d
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
      ), network.getInput(1))
    )

    if (fitness) {
      def auxRmsLayer(layer: DAGNode, target: Double) = network.add(new AbsActivationLayer(),
        network.add(new LinearActivationLayer().setBias(-target).freeze(),
          network.add(new AvgReducerLayer(),
            network.add(new StdDevMetaLayer(), layer))
        ))

      network.add(new ProductLayer(),
        entropy,
        network.add(new SumInputsLayer(), (
          List(network.add(new ConstNNLayer(new Tensor(1).set(0, 0.1)))) ++
          List(
            inception_3a,
            inception_3b,
            inception_4a,
            inception_4b,
            inception_4c,
            inception_4d,
            inception_4e,
            inception_5a,
            inception_5b
        ).map(auxRmsLayer(_, 1))): _*
      ))
    }

    network
  }

  def fitness(monitor: TrainingMonitor, monitoringRoot: MonitoredObject, data: Array[Array[Tensor]], n: Int = 3): Double = {
    val values = (1 to n).map(i ⇒ {
      val network = getNetwork(monitor, monitoringRoot, fitness = true)
      require(!data.isEmpty)
      val fn = Java8Util.cvt((x: Tensor) => x.getData()(0))
      network.eval(new NNExecutionContext() {}, NNResult.batchResultArray(data: _*): _*)
        .getData.stream().mapToDouble(fn).sum / data.length
    }).toList
    val avg = values.sum / n
    monitor.log(s"Numeric Opt: $this => $avg ($values)")
    avg
  }

}


class GoogLeNetModeler(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  def run(awaitExit: Boolean = true): Unit = {
    recordMetrics = false
    defineHeader()
    declareTestHandler()
    out.out("<hr/>")
    val timeBlockMinutes = 60
    if (findFile(modelName).isEmpty || System.getProperties.containsKey("rebuild")) {
      step_Generate()
    }
    for (i <- 1 to 10) {
      monitor.clear()
      step_Train(trainingMin = timeBlockMinutes, numberOfCategories=5, sampleSize = 500, iterationsPerSample = 5)
    }
    step_GAN()
    for (i <- 1 to 10) {
      monitor.clear()
      step_Train(trainingMin = timeBlockMinutes, sampleSize = 250)
    }
    step_GAN()
    out.out("<hr/>")
    if (awaitExit) waitForExit()
  }

  def resize(source: BufferedImage, size: Int) = {
    val image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB)
    val graphics = image.getGraphics.asInstanceOf[Graphics2D]
    graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
    graphics.drawImage(source, 0, 0, size, size, null)
    image
  }

  def resize(source: BufferedImage, width: Int, height: Int) = {
    val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
    val graphics = image.getGraphics.asInstanceOf[Graphics2D]
    graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
    graphics.drawImage(source, 0, 0, width, height, null)
    image
  }

  def step_Generate() = phase({
    //new GoogLeNet().getNetwork(monitor, monitoringRoot, false)
    //    lazy val optTraining: Array[Array[Tensor]] = Random.shuffle(data.values.flatten).take(5).mapCoords(_.get()).toArray
    //    require(0 < optTraining.length)
    //    SimplexOptimizer[GoogLeNet](
    //      GoogLeNet(), // GoogLeNet(),
    //      x ⇒ x.fitness(monitor, monitoringRoot, optTraining, n=3), relativeTolerance=0.01
    //    ).getNetwork(monitor, monitoringRoot)
    GoogLeNet(
      -1.17, -0.8, -1.7599999999999998, 1.0, 0.875, 0.60303759765625, 0.5750375976562501, 0.5750375976562501,
      0.5750375976562501, 0.5620375976562499, 0.3610414123535156, 0.3610414123535156, -3.6989585876464846,
      -1.2789585876464842, -1.3111851501464842, -1.8149351501464843, -1.3424351501464842, -1.6625, -1.0299999999999998, -2.0
    ).getNetwork(monitor, monitoringRoot)
  }, (model: NNLayer) ⇒ {
    // Do Nothing
  }: Unit, modelName)



  def step_Train(trainingMin: Int = 15, numberOfCategories: Int = 2, sampleSize: Int = 250, iterationsPerSample: Int = 5) = {
    var selectedCategories = selectCategories(numberOfCategories)
    val categoryArray = selectedCategories.keys.toArray
    val categoryIndices = categoryArray.zipWithIndex.toMap
    selectedCategories = selectedCategories.map(e=>{
      e._1 -> e._2.map(f=>new WeakCachedSupplier[Array[Tensor]](()=>{
        f.get().take(1) ++ Array(toOutNDArray(categoryIndices.size, categoryIndices(e._1)))
      }))
    })
    phase(modelName, (model: NNLayer) ⇒ {
      out.h1("Integration Training")
      //      model.asInstanceOf[DAGNetwork].visitLayers((layer:NNLayer)=>if(layer.isInstanceOf[SchemaComponent]) {
      //        layer.asInstanceOf[SchemaComponent].setSchema(categoryArray:_*)
      //      } : Unit)
      val trainer2 = out.eval {
        assert(null != data)
        var inner: Trainable = new SampledArrayTrainable(selectedCategories.values.flatten.toList.asJava, model, sampleSize)
        val trainer = new IterativeTrainer(inner)
        trainer.setMonitor(monitor)
        trainer.setTimeout(trainingMin, TimeUnit.MINUTES)
        trainer.setIterationsPerSample(iterationsPerSample)
        trainer.setOrientation(new LBFGS() {
          override def reset(): Unit = {
            model.asInstanceOf[DAGNetwork].visitLayers(Java8Util.cvt(layer => layer match {
              case layer: DropoutNoiseLayer => layer.shuffle()
              case _ =>
            }))
            super.reset()
          }
        }.setMinHistory(4).setMaxHistory(20))
        trainer.setLineSearchFactory(Java8Util.cvt((s: String) ⇒ (s match {
          case s if s.contains("LBFGS") ⇒ new StaticLearningRate(1.0)
          case _ ⇒ new ArmijoWolfeSearch().setAlpha(1e-5)
        })))
        trainer.setTerminateThreshold(0.0)
        trainer
      }
      trainer2.run()
    }: Unit, modelName)
    (for (i <- 1 to 3) yield Random.shuffle(selectedCategories.keys).take(2).toList).distinct.foreach {
      case Seq(from: String, to: String) => gan(out, model)(imageCount = 1, sourceCategory = from, targetCategory = to)
    }
  }

  def step_GAN(imageCount: Int = 10, sourceCategory: String = "fire-hydrant", targetCategory: String = "bear") = phase(modelName, (model: NNLayer) ⇒ {
    gan(out, model)(imageCount = imageCount, sourceCategory = sourceCategory, targetCategory = targetCategory)
  }: Unit, null)

  def gan(out: HtmlNotebookOutput with ScalaNotebookOutput, model: NNLayer)
         (imageCount: Int = 1, sourceCategory: String = "fire-hydrant", targetCategory: String = "bear") = {
    assert(null != model)
    val categoryArray = Array(sourceCategory, targetCategory)
    val categoryIndices = categoryArray.zipWithIndex.toMap
    val sourceClassId = categoryIndices(sourceCategory)
    val targetClassId = categoryIndices(targetCategory)
    out.h1(s"GAN Images Generation: $sourceCategory to $targetCategory")
    val sourceClass = toOutNDArray(categoryArray.length, sourceClassId)
    val targetClass = toOutNDArray(categoryArray.length, targetClassId)
    val adversarialOutput = new ArrayBuffer[Array[Tensor]]()
    //    model.asInstanceOf[DAGNetwork].visitLayers((layer:NNLayer)=>if(layer.isInstanceOf[SchemaComponent]) {
    //      layer.asInstanceOf[SchemaComponent].setSchema(categoryArray:_*)
    //    } : Unit)
    val rows = data(sourceCategory)
      .filter(_!=null)
      .take(imageCount)
      .grouped(1).map(group => {
      val adversarialData: Array[Array[Tensor]] = group.map(_.get().take(1) ++ Array(targetClass)).toArray
      val biasLayer = new BiasLayer(data.values.flatten.head.get().head.getDimensions(): _*)
      val trainingNetwork = new PipelineNetwork()
      trainingNetwork.add(biasLayer)
      val pipelineNetwork = KryoUtil.kryo().copy(model).freeze().asInstanceOf[PipelineNetwork]
      pipelineNetwork.setHead(pipelineNetwork.getByLabel("classify")).removeLastInput()
      trainingNetwork.add(pipelineNetwork)
      System.out.print(s"Starting to process ${adversarialData.length} images")
      val trainer1 = out.eval {
        var inner: Trainable = new ArrayTrainable(adversarialData,
          new SimpleLossNetwork(trainingNetwork, new EntropyLossLayer()))
        val trainer = new IterativeTrainer(inner)
        trainer.setMonitor(monitor)
        trainer.setTimeout(1, TimeUnit.MINUTES)
        trainer.setOrientation(new GradientDescent)
        trainer.setLineSearchFactory(Java8Util.cvt((s) ⇒ new ArmijoWolfeSearch().setMaxAlpha(1e8)))
        //trainer.setLineSearchFactory(Java8Util.cvt((s) ⇒ new QuadraticSearch))
        trainer.setTerminateThreshold(0.01)
        trainer
      }
      trainer1.run()
      System.out.print(s"Finished processing ${adversarialData.length} images")
      val evalNetwork = new PipelineNetwork()
      evalNetwork.add(biasLayer)
      val adversarialImage = evalNetwork.eval(new NNExecutionContext {}, adversarialData.head.head).getData.get(0)
      adversarialOutput += Array(adversarialImage, sourceClass)
      Map[String, AnyRef](
        "Original Image" → out.image(adversarialData.head.head.toRgbImage, ""),
        "Adversarial" → out.image(adversarialImage.toRgbImage, "")
      ).asJava
    }).toArray
    out.eval {
      TableOutput.create(rows: _*)
    }

  }

  def declareTestHandler() = {
    out.p("<a href='testCat.html'>Test Categorization</a><br/>")
    server.addSyncHandler("testCat.html", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(out ⇒ {
        testCategorization(out, getModelCheckpoint)
      })
    }), false)
    out.p("<a href='gan.html'>Generate Adversarial Images</a><br/>")
    server.addSyncHandler("gan.html", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(out ⇒ {
        gan(out, getModelCheckpoint)()
      })
    }), false)
  }

  def testCategorization(out: HtmlNotebookOutput with ScalaNotebookOutput, model: NNLayer) = {
    try {
      out.eval {
        TableOutput.create(takeData(5, 10).map(_.get()).map(testObj ⇒ Map[String, AnyRef](
          "Image" → out.image(testObj(0).toRgbImage(), ""),
          "Categorization" → categories.toList.sortBy(_._2).map(_._1)
            .zip(model.eval(new NNExecutionContext() {}, testObj(0)).getData.get(0).getData.map(_ * 100.0))
        ).asJava): _*)
      }
    } catch {
      case e: Throwable ⇒ e.printStackTrace()
    }
  }

  lazy val categories: Map[String, Int] = categoryList.zipWithIndex.toMap
  lazy val (categoryList, data) = {
    out.p("Loading data from " + source)
    val (categoryList, data) = {
      val categoryDirs = Random.shuffle(new File(source).listFiles().toStream)
        .filter(dir => categoryWhitelist.isEmpty || categoryWhitelist.find(str => dir.getName.contains(str)).isDefined)
        .take(numberOfCategories)
      val categoryList = categoryDirs.map((categoryDirectory: File) ⇒ {
        categoryDirectory.getName.split('.').last
      }).sorted.toArray
      val categoryMap: Map[String, Int] = categoryList.zipWithIndex.toMap
      (categoryList, categoryDirs
        .map((categoryDirectory: File) ⇒ {
          val categoryName = categoryDirectory.getName.split('.').last
          categoryName -> categoryDirectory.listFiles()
            .filterNot(_ == null)
            .filter(_.exists())
            .filter(_.length() > 0)
            .par.map(readImage(_))
            .flatMap(variants(_, artificialVariants))
            .map(resize(_, tileSize))
            .map(toTenors(_, toOutNDArray(categoryMap.size, categoryMap(categoryName))))
            .toList
        }).toMap)
    }
    val categories: Map[String, Int] = categoryList.zipWithIndex.toMap
    out.p("<ol>" + categories.toList.sortBy(_._2).map(x ⇒ "<li>" + x + "</li>").mkString("\n") + "</ol>")
    out.eval {
      TableOutput.create(Random.shuffle(data.toList).take(10).flatMap(x => Random.shuffle(x._2).take(10)).par.filter(_.get() != null).map(x ⇒ {
        val e = x.get()
        Map[String, AnyRef](
          "Image" → out.image(e(0).toRgbImage(), e(1).toString),
          "Classification" → e(1)
        ).asJava
      }).toArray: _*)
    }
    out.p("Loading data complete")
    (categoryList, data)
  }

  private def readImage(file: File): Supplier[BufferedImage] = {
    new WeakCachedSupplier[BufferedImage](Java8Util.cvt(() => {
      try {
        val image = ImageIO.read(file)
        if (null == image) {
          System.err.println(s"Error reading ${file.getAbsolutePath}: No image found")
        }
        image
      } catch {
        case e: Throwable =>
          System.err.println(s"Error reading ${file.getAbsolutePath}: $e")
          file.delete()
          null
      }
    }))
  }

  private def toTenors(originalRef:Supplier[BufferedImage], expectedOutput: Tensor): Supplier[Array[Tensor]] = {
    new SoftCachedSupplier[Array[Tensor]](Java8Util.cvt(() => {
      try {
        val resized = originalRef.get()
        if (null == resized) null
        else {
          Array(Tensor.fromRGB(resized), expectedOutput)
        }
      } catch {
        case e: Throwable =>
          e.printStackTrace(System.err)
          null
      }
    }
    ))
  }

  private def resize(originalRef:Supplier[BufferedImage], tileSize:Int): Supplier[BufferedImage] = {
    new SoftCachedSupplier[BufferedImage](Java8Util.cvt(() => {
      try {
        val original = originalRef.get()
        if (null == original) null
        else {
          val fromWidth = original.getWidth()
          val fromHeight = original.getHeight()
          val scale = tileSize.toDouble / Math.min(fromWidth, fromHeight)
          val toWidth = ((fromWidth * scale).toInt)
          val toHeight = ((fromHeight * scale).toInt)
          val resized = new BufferedImage(tileSize, tileSize, BufferedImage.TYPE_INT_ARGB)
          val graphics = resized.getGraphics.asInstanceOf[Graphics2D]
          graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
          if (toWidth < toHeight) {
            graphics.drawImage(original, 0, (toWidth - toHeight) / 2, toWidth, toHeight, null)
          } else {
            graphics.drawImage(original, (toHeight - toWidth) / 2, 0, toWidth, toHeight, null)
          }
          resized
        }
      } catch {
        case e: Throwable =>
          e.printStackTrace(System.err)
          null
      }
    }
    ))
  }

  def variants(imageFn: Supplier[BufferedImage], items: Int): Stream[Supplier[BufferedImage]] = {
    Stream.continually({
      val sy = 1.05 + Random.nextDouble() * 0.05
      val sx = 1.05 + Random.nextDouble() * 0.05
      val theta = (Random.nextDouble() - 0.5) * 0.2
      new SoftCachedSupplier[BufferedImage](()=>{
        val image = imageFn.get()
        if(null == image) return null
        val resized = new BufferedImage(image.getWidth, image.getHeight, BufferedImage.TYPE_INT_ARGB)
        val graphics = resized.getGraphics.asInstanceOf[Graphics2D]
        val transform = AffineTransform.getScaleInstance(sx,sy)
        transform.concatenate(AffineTransform.getRotateInstance(theta))
        graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
        graphics.drawImage(image, transform, null)
        resized
      })
    }).take(items)
  }

  def toOutNDArray(max: Int, out: Int*): Tensor = {
    val ndArray = new Tensor(max)
    for (i <- 0 until max) ndArray.set(i, fuzz)
    out.foreach(out=>ndArray.set(out, 1 - (fuzz * (max - 1))))
    ndArray
  }

  def takeData(numCategories: Int = 2, numImages: Int = 5000): List[_ <: Supplier[Array[Tensor]]] = {
    val selectedCategories = selectCategories(numCategories)
    takeNonNull(numImages, selectedCategories)
  }

  def takeNonNull[X <: Supplier[Array[Tensor]]](numImages: Int, selectedCategories: Map[String, List[X]])(implicit classTag: ClassTag[X]): List[X] = {
    monitor.log(s"Selecting $numImages images from categories ${selectedCategories.keySet}")
    Random.shuffle(selectedCategories.values.flatten.toList).par.take(2*numImages).filter(_.get() != null).take(numImages).toArray.toList
  }

  def selectCategories(numCategories: Int) = {
    Random.shuffle(data.toList).take(numCategories).toMap
  }

}

