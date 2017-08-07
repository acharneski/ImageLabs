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

import java.awt.image.BufferedImage
import java.awt.{Graphics2D, RenderingHints}
import java.io._
import java.util.concurrent.TimeUnit
import java.util.function.Supplier
import javax.imageio.ImageIO

import _root_.util.Java8Util.cvt
import _root_.util._
import com.simiacryptus.mindseye.layers.NNLayer.NNExecutionContext
import com.simiacryptus.mindseye.layers.activation._
import com.simiacryptus.mindseye.layers.cudnn.CuDNN
import com.simiacryptus.mindseye.layers.cudnn.f32.PoolingLayer.PoolingMode
import com.simiacryptus.mindseye.layers.cudnn.f32._
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer
import com.simiacryptus.mindseye.layers.meta.StdDevMetaLayer
import com.simiacryptus.mindseye.layers.reducers.{AvgReducerLayer, ImgConcatLayer, ProductInputsLayer, SumInputsLayer}
import com.simiacryptus.mindseye.layers.synapse.{BiasLayer, DenseSynapseLayer}
import com.simiacryptus.mindseye.layers.util.{AssertDimensionsLayer, ConstNNLayer}
import com.simiacryptus.mindseye.layers.{NNLayer, NNResult}
import com.simiacryptus.mindseye.network.graph.{DAGNetwork, DAGNode}
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line._
import com.simiacryptus.mindseye.opt.orient._
import com.simiacryptus.mindseye.opt.trainable._
import com.simiacryptus.util.io.{HtmlNotebookOutput, KryoUtil}
import com.simiacryptus.util.ml.{CachedSupplier, SoftCachedSupplier, Tensor, WeakCachedSupplier}
import com.simiacryptus.util.text.TableOutput
import com.simiacryptus.util.{MonitoredObject, StreamNanoHTTPD}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.Random

object GoogleLeNetModeler extends Report {
  val modelName = System.getProperty("modelName", "googlenet_1")
  val tileSize = 224
  val categoryWhitelist = Set[String]()
  //("greyhound", "soccer-ball", "telephone-box", "windmill")
  val numberOfCategories = 256 // categoryWhitelist.size
  val imagesPerCategory = 500
  val fuzz = 1e-4

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒ new GoogleLeNetModeler(source, server, out).run()
      case _ ⇒ new GoogleLeNetModeler("D:\\testImages\\256_ObjectCategories", server, out).run()
    })

  }

}

import interactive.classify.GoogleLeNetModeler._

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

  def getNetwork(monitor: TrainingMonitor,
                 monitoringRoot: MonitoredObject,
                 fitness: Boolean = false): NNLayer = {
    val network = new PipelineNetwork(2)
    network.add(new AssertDimensionsLayer(224, 224, 3))
    network.addAll(
      new ConvolutionLayer(7, 7, 3, 64).setWeightsLog(layer1).setStrideXY(2, 2).setName("conv_1"),
      new ImgBandBiasLayer(64).setName("bias_1"),
      new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_1"))
    network.add(new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1).setName("pool_1"))
    network.addAll(
      new ConvolutionLayer(1, 1, 64, 64).setWeightsLog(layer2).setName("conv_2"),
      new ImgBandBiasLayer(64).setName("bias_2"),
      new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_2"))
    network.addAll(
      new ConvolutionLayer(3, 3, 64, 192).setWeightsLog(layer3).setName("conv_3"),
      new ImgBandBiasLayer(192).setName("bias_3"),
      new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_3"))
    network.add(new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1).setName("pooling_3"))
    val prefix_head = network.getHead
    val inception_3a = network.add(new ImgConcatLayer(),
      network.addAll(prefix_head,
        new ConvolutionLayer(1, 1, 192, 64).setWeightsLog(layer4 + conv1a),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(prefix_head,
        new ConvolutionLayer(1, 1, 192, 96).setWeightsLog(layer4 + conv3a),
        new ImgBandBiasLayer(96),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 96, 128).setWeightsLog(layer4 + conv3b),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(prefix_head,
        new ConvolutionLayer(1, 1, 192, 16).setWeightsLog(layer4 + conv5a),
        new ImgBandBiasLayer(16),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 16, 32).setWeightsLog(layer4 + conv5b),
        new ImgBandBiasLayer(32),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(prefix_head,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 192, 32).setWeightsLog(layer4 + conv1b),
        new ImgBandBiasLayer(32),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    val inception_3b = network.add(new ImgConcatLayer(),
      network.addAll(inception_3a,
        new ConvolutionLayer(1, 1, 256, 128).setWeightsLog(layer5 + conv1a),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_3a,
        new ConvolutionLayer(1, 1, 256, 128).setWeightsLog(layer5 + conv3a),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 128, 192).setWeightsLog(layer5 + conv3b),
        new ImgBandBiasLayer(192),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_3a,
        new ConvolutionLayer(1, 1, 256, 32).setWeightsLog(layer5 + conv5a),
        new ImgBandBiasLayer(32),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 32, 96).setWeightsLog(layer5 + conv5b),
        new ImgBandBiasLayer(96),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_3a,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 256, 64).setWeightsLog(layer5 + conv1b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    network.add(new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1))
    val pooling_3 = network.getHead
    val inception_4a = network.add(new ImgConcatLayer(),
      network.addAll(pooling_3,
        new ConvolutionLayer(1, 1, 480, 192).setWeightsLog(layer6 + conv1a),
        new ImgBandBiasLayer(192),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(pooling_3,
        new ConvolutionLayer(1, 1, 480, 96).setWeightsLog(layer6 + conv3a),
        new ImgBandBiasLayer(96),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 96, 208).setWeightsLog(layer6 + conv3b),
        new ImgBandBiasLayer(208),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(pooling_3,
        new ConvolutionLayer(1, 1, 480, 16).setWeightsLog(layer6 + conv5a),
        new ImgBandBiasLayer(16),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 16, 48).setWeightsLog(layer6 + conv5b),
        new ImgBandBiasLayer(48),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(pooling_3,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 480, 64).setWeightsLog(layer6 + conv1b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    val inception_4b = network.add(new ImgConcatLayer(),
      network.addAll(inception_4a,
        new ConvolutionLayer(1, 1, 512, 160).setWeightsLog(layer7 + conv1a),
        new ImgBandBiasLayer(160),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4a,
        new ConvolutionLayer(1, 1, 512, 112).setWeightsLog(layer7 + conv3a),
        new ImgBandBiasLayer(112),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 112, 224).setWeightsLog(layer7 + conv3b),
        new ImgBandBiasLayer(224),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4a,
        new ConvolutionLayer(1, 1, 512, 24).setWeightsLog(layer7 + conv5a),
        new ImgBandBiasLayer(24),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 24, 64).setWeightsLog(layer7 + conv5b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4a,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 512, 64).setWeightsLog(layer7 + conv1b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    val inception_4c = network.add(new ImgConcatLayer(),
      network.addAll(inception_4b,
        new AssertDimensionsLayer(14, 14, 512),
        new ConvolutionLayer(1, 1, 512, 128).setWeightsLog(layer8 + conv1a),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4b,
        new ConvolutionLayer(1, 1, 512, 128).setWeightsLog(layer8 + conv3a),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 128, 256).setWeightsLog(layer8 + conv3b),
        new ImgBandBiasLayer(256),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4b,
        new ConvolutionLayer(1, 1, 512, 24).setWeightsLog(layer8 + conv5a),
        new ImgBandBiasLayer(24),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 24, 64).setWeightsLog(layer8 + conv5b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4b,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 512, 64).setWeightsLog(layer8 + conv1b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    val inception_4d = network.add(new ImgConcatLayer(),
      network.addAll(inception_4c,
        new ConvolutionLayer(1, 1, 512, 112).setWeightsLog(layer9 + conv1a),
        new ImgBandBiasLayer(112),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4c,
        new ConvolutionLayer(1, 1, 512, 144).setWeightsLog(layer9 + conv3a),
        new ImgBandBiasLayer(144),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 144, 288).setWeightsLog(layer9 + conv3b),
        new ImgBandBiasLayer(288),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4c,
        new ConvolutionLayer(1, 1, 512, 32).setWeightsLog(layer9 + conv5a),
        new ImgBandBiasLayer(32),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 32, 64).setWeightsLog(layer9 + conv5b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4c,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 512, 64).setWeightsLog(layer9 + conv1b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    val inception_4e = network.add(new ImgConcatLayer(),
      network.addAll(inception_4d,
        new ConvolutionLayer(1, 1, 528, 256).setWeightsLog(layer10 + conv1a),
        new ImgBandBiasLayer(256),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4d,
        new ConvolutionLayer(1, 1, 528, 160).setWeightsLog(layer10 + conv3a),
        new ImgBandBiasLayer(160),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 160, 320).setWeightsLog(layer10 + conv3b),
        new ImgBandBiasLayer(320),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4d,
        new ConvolutionLayer(1, 1, 528, 32).setWeightsLog(layer10 + conv5a),
        new ImgBandBiasLayer(32),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 32, 128).setWeightsLog(layer10 + conv5b),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4d,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 528, 128).setWeightsLog(layer10 + conv1b),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    network.add(new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1))
    val pooling_4 = network.getHead
    val inception_5a = network.add(new ImgConcatLayer(),
      network.addAll(pooling_4,
        new ConvolutionLayer(1, 1, 832, 256).setWeightsLog(layer11 + conv1a),
        new ImgBandBiasLayer(256),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(pooling_4,
        new ConvolutionLayer(1, 1, 832, 160).setWeightsLog(layer11 + conv3a),
        new ImgBandBiasLayer(160),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 160, 320).setWeightsLog(layer11 + conv3b),
        new ImgBandBiasLayer(320),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(pooling_4,
        new ConvolutionLayer(1, 1, 832, 32).setWeightsLog(layer11 + conv5a),
        new ImgBandBiasLayer(32),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 32, 128).setWeightsLog(layer11 + conv5b),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(pooling_4,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 832, 128).setWeightsLog(layer11 + conv1b),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    val inception_5b = network.add(new ImgConcatLayer(),
      network.addAll(inception_5a,
        new ConvolutionLayer(1, 1, 832, 384).setWeightsLog(layer12 + conv1a),
        new ImgBandBiasLayer(384),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_5a,
        new ConvolutionLayer(1, 1, 832, 192).setWeightsLog(layer12 + conv3a),
        new ImgBandBiasLayer(192),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 192, 384).setWeightsLog(layer12 + conv3b),
        new ImgBandBiasLayer(384),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_5a,
        new ConvolutionLayer(1, 1, 832, 48).setWeightsLog(layer12 + conv5a),
        new ImgBandBiasLayer(48),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 48, 128).setWeightsLog(layer12 + conv5b),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_5a,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 832, 128).setWeightsLog(layer12 + conv1b),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    network.add(new PoolingLayer().setWindowXY(7, 7).setStrideXY(1, 1).setPaddingXY(0, 0).setMode(PoolingMode.Avg))
    network.add(new DropoutNoiseLayer().setValue(0.4))
    network.add(new DenseSynapseLayer(Array(1, 1, 1024), Array(numberOfCategories)).setWeightsLog(layer13))
    network.add(new BiasLayer(numberOfCategories))
    val finalClassification = network.add("classify", new SoftmaxActivationLayer(), network.getHead)


    val entropy = network.add(new SumInputsLayer(),
      network.add(new EntropyLossLayer(), finalClassification, network.getInput(1)),
      network.add(new EntropyLossLayer(), network.add(
        new SoftmaxActivationLayer(),
        network.add(
          new BiasLayer(numberOfCategories),
          network.add(
            new DenseSynapseLayer(Array(1024), Array(numberOfCategories)).setWeightsLog(trainingShunt),
            network.add(
              new DropoutNoiseLayer().setValue(0.7),
              network.add(
                new ReLuActivationLayer().freeze(),
                network.add(
                  new BiasLayer(1024),
                  network.add(
                    new DenseSynapseLayer(Array(3, 3, 128), Array(1024)).setWeightsLog(trainingShunt),
                    network.add(
                      new ActivationLayer(ActivationLayer.Mode.RELU),
                      network.add(
                        new ConvolutionLayer(1, 1, 512, 128).setWeightsLog(trainingShunt),
                        network.add(
                          new PoolingLayer().setWindowXY(7, 7).setStrideXY(3, 3).setMode(PoolingMode.Avg),
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
      network.add(new EntropyLossLayer(), network.add(
        new SoftmaxActivationLayer(),
        network.add(
          new BiasLayer(numberOfCategories),
          network.add(
            new DenseSynapseLayer(Array(1024), Array(numberOfCategories)).setWeightsLog(trainingShunt),
            network.add(
              new DropoutNoiseLayer().setValue(0.7),
              network.add(
                new ReLuActivationLayer().freeze(),
                network.add(
                  new BiasLayer(1024),
                  network.add(
                    new DenseSynapseLayer(Array(3, 3, 128), Array(1024)).setWeightsLog(trainingShunt),
                    network.add(
                      new ActivationLayer(ActivationLayer.Mode.RELU),
                      network.add(
                        new ConvolutionLayer(1, 1, 528, 128).setWeightsLog(trainingShunt),
                        network.add(
                          new PoolingLayer().setWindowXY(7, 7).setStrideXY(3, 3).setMode(PoolingMode.Avg),
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

      network.add(new ProductInputsLayer(), entropy, network.add(new SumInputsLayer(),
        (List(network.add(new ConstNNLayer(new Tensor(1).set(0, 0.1)))) ++ List(
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
      network.eval(new NNLayer.NNExecutionContext() {}, NNResult.batchResultArray(data))
        .data.stream().mapToDouble(fn).sum / data.length
    }).toList
    val avg = values.sum / n
    monitor.log(s"Numeric Opt: $this => $avg ($values)")
    avg
  }

}


class GoogleLeNetModeler(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  def run(awaitExit: Boolean = true): Unit = {
    defineHeader()
    declareTestHandler()
    out.out("<hr/>")
    val timeBlockMinutes = 60
    if (findFile(modelName).isEmpty || System.getProperties.containsKey("rebuild")) {
      step_Generate()
    }
    for (i <- 1 to 10) {
      step_Train(trainingMin = timeBlockMinutes, sampleSize = 100)
      step_GAN()
    }
    for (i <- 1 to 10) {
      step_Train(trainingMin = timeBlockMinutes, sampleSize = 250)
      step_GAN()
    }
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
    //    lazy val optTraining: Array[Array[Tensor]] = Random.shuffle(data.values.flatten).take(5).map(_.get()).toArray
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


  def step_Train(trainingMin: Int = 15, numberOfCategories: Int = 2, sampleSize: Int = 250, iterationsPerSample: Int = 10) = {
    val selectedCategories = selectCategories(numberOfCategories)
    phase(modelName, (model: NNLayer) ⇒ {
      out.h1("Integration Training")
      val trainer2 = out.eval {
        assert(null != data)
        val trainingData = takeData(sampleSize, selectedCategories).map(x=>new CachedSupplier[Array[Tensor]](Java8Util.cvt(()=>x.get())))
        var inner: Trainable = new StochasticArrayTrainable(trainingData.asJava, model, sampleSize)
        val trainer = new IterativeTrainer(inner)
        trainer.setMonitor(monitor)
        trainer.setTimeout(trainingMin, TimeUnit.MINUTES)
        trainer.setIterationsPerSample(iterationsPerSample)
        trainer.setOrientation(new LBFGS() {
          override def reset(): Unit = {
            model.asInstanceOf[DAGNetwork].visit(Java8Util.cvt(layer => layer match {
              case layer: DropoutNoiseLayer => layer.shuffle()
              case _ =>
            }))
            super.reset()
          }
        }.setMinHistory(4).setMaxHistory(20))
        trainer.setLineSearchFactory(Java8Util.cvt((s: String) ⇒ (s match {
          case s if s.contains("LBFGS") ⇒ new StaticLearningRate().setRate(1.0)
          case _ ⇒ new ArmijoWolfeSearch().setAlpha(1e-5)
        })))
        trainer.setTerminateThreshold(0.0)
        trainer
      }
      trainer2.run()
    }: Unit, modelName)
    (for (i <- 1 to 3) yield Random.shuffle(selectedCategories.keys).take(2).toList).distinct.foreach {
      case Seq(from: String, to: String) => gan(out, model)(imageCount = 5, sourceCategory = from, targetCategory = to)
    }
  }

  def step_GAN(imageCount: Int = 10, sourceCategory: String = "fire-hydrant", targetCategory: String = "bear") = phase(modelName, (model: NNLayer) ⇒ {
    gan(out, model)(imageCount = imageCount, sourceCategory = sourceCategory, targetCategory = targetCategory)
  }: Unit, null)

  def gan(out: HtmlNotebookOutput with ScalaNotebookOutput, model: NNLayer)
         (imageCount: Int = 1, sourceCategory: String = "fire-hydrant", targetCategory: String = "bear") = {
    assert(null != model)
    val sourceClassId = categories(sourceCategory)
    val targetClassId = categories(targetCategory)
    out.h1(s"GAN Images Generation: $sourceCategory to $targetCategory")
    val sourceClass = toOutNDArray(sourceClassId, numberOfCategories)
    val targetClass = toOutNDArray(targetClassId, numberOfCategories)
    val adversarialOutput = new ArrayBuffer[Array[Tensor]]()
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
      CuDNN.devicePool.`with`(Java8Util.cvt((device: CuDNN) => {
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
      }))
      val evalNetwork = new PipelineNetwork()
      evalNetwork.add(biasLayer)
      val adversarialImage = evalNetwork.eval(new NNExecutionContext {}, adversarialData.head.head).data.get(0)
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
            .zip(model.eval(new NNLayer.NNExecutionContext() {}, testObj(0)).data.get(0).getData.map(_ * 100.0))
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
      (categoryList, Random.shuffle(categoryDirs
        .map((categoryDirectory: File) ⇒ {
          val categoryName = categoryDirectory.getName.split('.').last
          categoryName -> Random.shuffle(categoryDirectory.listFiles().toStream)
            .filterNot(_ == null).filterNot(_ == null)
            .filter(_.exists())
            .filter(_.length() > 0)
            .par.map(file ⇒ {
              new WeakCachedSupplier[BufferedImage](Java8Util.cvt(() => {
                try {
                  ImageIO.read(file)
                } catch {
                  case e: Throwable =>
                    new RuntimeException(s"Error map ${file.getAbsolutePath}", e)
                    null
                }
              }))
            }).map(originalRef ⇒ {
              new SoftCachedSupplier[BufferedImage](Java8Util.cvt(() => {
                try {
                  val original = originalRef.get()
                  if(null == original) null
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
                    e.printStackTrace()
                    null
                }
              }
            ))
          }).map(originalRef ⇒ {
            new WeakCachedSupplier[Array[Tensor]](Java8Util.cvt(() => {
              try {
                val resized = originalRef.get()
                if(null == resized) null
                else {
                  Array(Tensor.fromRGB(resized), toOutNDArray(categoryMap(categoryName), categoryMap.size))
                }
              } catch {
                case e: Throwable =>
                  e.printStackTrace()
                  null
              }
            }
            ))
          }).toArray.toList
        })).toMap)
    }
    val categories: Map[String, Int] = categoryList.zipWithIndex.toMap
    out.p("<ol>" + categories.toList.sortBy(_._2).map(x ⇒ "<li>" + x + "</li>").mkString("\n") + "</ol>")
    out.eval {
      TableOutput.create(Random.shuffle(data.values.flatten).par.filter(_.get() != null).take(100).map(x ⇒ {
        val e = x.get()
        Map[String, AnyRef](
          "Image" → out.image(e(0).toRgbImage(), e(1).toString),
          "Classification" → e(1)
        ).asJava
      }).toList: _*)
    }
    out.p("Loading data complete")
    (categoryList, data)
  }

  def toOutNDArray(out: Int, max: Int): Tensor = {
    val ndArray = new Tensor(max)
    for (i <- 0 until max) ndArray.set(i, fuzz)
    ndArray.set(out, 1 - (fuzz * (max - 1)))
    ndArray
  }

  def takeData(numCategories: Int = 2, numImages: Int = 5000): List[_ <: Supplier[Array[Tensor]]] = {
    val selectedCategories = selectCategories(numCategories)
    takeData(numImages, selectedCategories)
  }

  def takeData[X <: Supplier[Array[Tensor]]](numImages: Int, selectedCategories: Map[String, List[X]])(implicit classTag: ClassTag[X]): List[X] = {
    monitor.log(s"Selecting $numImages images from categories ${selectedCategories.keySet}")
    Random.shuffle(selectedCategories.values.flatten.toList).par.filter(_.get() != null).take(numImages).toArray.toList
  }

  def selectCategories(numCategories: Int) = {
    Random.shuffle(data.toList).take(numCategories).toMap
  }

}

