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
import javax.imageio.ImageIO

import _root_.util.Java8Util.cvt
import _root_.util._
import com.simiacryptus.mindseye.layers.NNLayer
import com.simiacryptus.mindseye.layers.NNLayer.NNExecutionContext
import com.simiacryptus.mindseye.layers.activation._
import com.simiacryptus.mindseye.layers.cudnn.f32.PoolingLayer.PoolingMode
import com.simiacryptus.mindseye.layers.cudnn.f32._
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer
import com.simiacryptus.mindseye.layers.meta.StdDevMetaLayer
import com.simiacryptus.mindseye.layers.reducers.{AvgReducerLayer, ImgConcatLayer, ProductInputsLayer, SumInputsLayer}
import com.simiacryptus.mindseye.layers.synapse.{BiasLayer, DenseSynapseLayer}
import com.simiacryptus.mindseye.layers.util.{AssertDimensionsLayer, ConstNNLayer}
import com.simiacryptus.mindseye.network.graph.DAGNode
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line._
import com.simiacryptus.mindseye.opt.orient._
import com.simiacryptus.mindseye.opt.trainable._
import com.simiacryptus.util.io.{HtmlNotebookOutput, KryoUtil}
import com.simiacryptus.util.ml.{Tensor, WeakCachedSupplier}
import com.simiacryptus.util.text.TableOutput
import com.simiacryptus.util.{MonitoredObject, StreamNanoHTTPD}
import interactive.classify.IncrementalClassifierModeler.numberOfCategories
import interactive.superres.SimplexOptimizer

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object GoogleLeNetModeler extends Report {
  val modelName = System.getProperty("modelName","googlenet_1")
  val tileSize = 224
  val categoryWhitelist = Set[String]()//("greyhound", "soccer-ball", "telephone-box", "windmill")
  val numberOfCategories = 10 // categoryWhitelist.size
  val imagesPerCategory = 10
  val fuzz = 1e-2

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒ new GoogleLeNetModeler(source, server, out).run()
      case _ ⇒ new GoogleLeNetModeler("D:\\testImages\\256_ObjectCategories", server, out).run()
    })

  }

}
import interactive.classify.GoogleLeNetModeler._

case class GoogLeNet(
                           layer1 : Double = -2,
                           layer2 : Double = -2,
                           layer3 : Double = -2,
                           layer4 : Double = -2,
                           layer5 : Double = -2,
                           layer6 : Double = -2,
                           layer7 : Double = -2,
                           layer8 : Double = -2,
                           layer9 : Double = -2,
                           layer10 : Double = -2,
                           layer11 : Double = -2,
                           layer12 : Double = -2,
                           layer13 : Double = -2,
                           conv1a : Double = -2,
                           conv3a : Double = -2,
                           conv3b : Double = -2,
                           conv5a : Double = -2,
                           conv5b : Double = -2,
                           conv1b : Double = -2,
                           trainingShunt : Double = -2
                         ) {

  def getNetwork(monitor: TrainingMonitor,
                 monitoringRoot : MonitoredObject,
                 fitness : Boolean = false) : NNLayer = {
    val network = new PipelineNetwork(2)
    network.add(new AssertDimensionsLayer(224,224,3))
    network.addAll(
      new ConvolutionLayer(7, 7, 3, 64, true).setWeightsLog(layer1).setStrideXY(2, 2).setName("conv_1"),
      new ImgBandBiasLayer(64).setName("bias_1"),
      new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_1"))
    network.add(new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1).setName("pool_1"))
    network.addAll(
      new ConvolutionLayer(1, 1, 64, 64).setWeightsLog(layer2).setName("conv_2"),
      new ImgBandBiasLayer(64).setName("bias_2"),
      new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_2"))
    network.addAll(
      new ConvolutionLayer(3, 3, 64, 192, true).setWeightsLog(layer3).setName("conv_3"),
      new ImgBandBiasLayer(192).setName("bias_3"),
      new ActivationLayer(ActivationLayer.Mode.RELU).setName("relu_3"))
    network.add(new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1).setName("pooling_3"))
    val prefix_head = network.getHead
    val inception_3a = network.add(new ImgConcatLayer(),
      network.addAll(prefix_head,
        new ConvolutionLayer(1, 1, 192, 64).setWeightsLog(layer4+conv1a),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(prefix_head,
        new ConvolutionLayer(1, 1, 192, 96).setWeightsLog(layer4+conv3a),
        new ImgBandBiasLayer(96),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 96, 128).setWeightsLog(layer4+conv3b),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(prefix_head,
        new ConvolutionLayer(1, 1, 192, 16).setWeightsLog(layer4+conv5a),
        new ImgBandBiasLayer(16),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 16, 32).setWeightsLog(layer4+conv5b),
        new ImgBandBiasLayer(32),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(prefix_head,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 192, 32).setWeightsLog(layer4+conv1b),
        new ImgBandBiasLayer(32),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    val inception_3b = network.add(new ImgConcatLayer(),
      network.addAll(inception_3a,
        new ConvolutionLayer(1, 1, 256, 128).setWeightsLog(layer5+conv1a),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_3a,
        new ConvolutionLayer(1, 1, 256, 128).setWeightsLog(layer5+conv3a),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 128, 192).setWeightsLog(layer5+conv3b),
        new ImgBandBiasLayer(192),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_3a,
        new ConvolutionLayer(1, 1, 256, 32).setWeightsLog(layer5+conv5a),
        new ImgBandBiasLayer(32),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 32, 96).setWeightsLog(layer5+conv5b),
        new ImgBandBiasLayer(96),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_3a,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 256, 64).setWeightsLog(layer5+conv1b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    network.add(new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1))
    val pooling_3 = network.getHead
    val inception_4a = network.add(new ImgConcatLayer(),
      network.addAll(pooling_3,
        new ConvolutionLayer(1, 1, 480, 192).setWeightsLog(layer6+conv1a),
        new ImgBandBiasLayer(192),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(pooling_3,
        new ConvolutionLayer(1, 1, 480, 96).setWeightsLog(layer6+conv3a),
        new ImgBandBiasLayer(96),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 96, 208).setWeightsLog(layer6+conv3b),
        new ImgBandBiasLayer(208),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(pooling_3,
        new ConvolutionLayer(1, 1, 480, 16).setWeightsLog(layer6+conv5a),
        new ImgBandBiasLayer(16),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 16, 48).setWeightsLog(layer6+conv5b),
        new ImgBandBiasLayer(48),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(pooling_3,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 480, 64).setWeightsLog(layer6+conv1b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    val inception_4b = network.add(new ImgConcatLayer(),
      network.addAll(inception_4a,
        new ConvolutionLayer(1, 1, 512, 160).setWeightsLog(layer7+conv1a),
        new ImgBandBiasLayer(160),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4a,
        new ConvolutionLayer(1, 1, 512, 112).setWeightsLog(layer7+conv3a),
        new ImgBandBiasLayer(112),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 112, 224).setWeightsLog(layer7+conv3b),
        new ImgBandBiasLayer(224),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4a,
        new ConvolutionLayer(1, 1, 512, 24).setWeightsLog(layer7+conv5a),
        new ImgBandBiasLayer(24),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 24, 64).setWeightsLog(layer7+conv5b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4a,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 512, 64).setWeightsLog(layer7+conv1b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    val inception_4c = network.add(new ImgConcatLayer(),
      network.addAll(inception_4b,
        new AssertDimensionsLayer(14,14,512),
        new ConvolutionLayer(1, 1, 512, 128).setWeightsLog(layer8+conv1a),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4b,
        new ConvolutionLayer(1, 1, 512, 128).setWeightsLog(layer8+conv3a),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 128, 256).setWeightsLog(layer8+conv3b),
        new ImgBandBiasLayer(256),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4b,
        new ConvolutionLayer(1, 1, 512, 24).setWeightsLog(layer8+conv5a),
        new ImgBandBiasLayer(24),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 24, 64).setWeightsLog(layer8+conv5b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4b,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 512, 64).setWeightsLog(layer8+conv1b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    val inception_4d = network.add(new ImgConcatLayer(),
      network.addAll(inception_4c,
        new ConvolutionLayer(1, 1, 512, 112).setWeightsLog(layer9+conv1a),
        new ImgBandBiasLayer(112),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4c,
        new ConvolutionLayer(1, 1, 512, 144).setWeightsLog(layer9+conv3a),
        new ImgBandBiasLayer(144),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 144, 288).setWeightsLog(layer9+conv3b),
        new ImgBandBiasLayer(288),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4c,
        new ConvolutionLayer(1, 1, 512, 32).setWeightsLog(layer9+conv5a),
        new ImgBandBiasLayer(32),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 32, 64).setWeightsLog(layer9+conv5b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4c,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 512, 64).setWeightsLog(layer9+conv1b),
        new ImgBandBiasLayer(64),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    val inception_4e = network.add(new ImgConcatLayer(),
      network.addAll(inception_4d,
        new ConvolutionLayer(1, 1, 528, 256).setWeightsLog(layer10+conv1a),
        new ImgBandBiasLayer(256),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4d,
        new ConvolutionLayer(1, 1, 528, 160).setWeightsLog(layer10+conv3a),
        new ImgBandBiasLayer(160),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 160, 320).setWeightsLog(layer10+conv3b),
        new ImgBandBiasLayer(320),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4d,
        new ConvolutionLayer(1, 1, 528, 32).setWeightsLog(layer10+conv5a),
        new ImgBandBiasLayer(32),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 32, 128).setWeightsLog(layer10+conv5b),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_4d,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 528, 128).setWeightsLog(layer10+conv1b),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    network.add(new PoolingLayer().setWindowXY(3, 3).setStrideXY(2, 2).setPaddingXY(1, 1))
    val pooling_4 = network.getHead
    val inception_5a = network.add(new ImgConcatLayer(),
      network.addAll(pooling_4,
        new ConvolutionLayer(1, 1, 832, 256).setWeightsLog(layer11+conv1a),
        new ImgBandBiasLayer(256),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(pooling_4,
        new ConvolutionLayer(1, 1, 832, 160).setWeightsLog(layer11+conv3a),
        new ImgBandBiasLayer(160),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 160, 320).setWeightsLog(layer11+conv3b),
        new ImgBandBiasLayer(320),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(pooling_4,
        new ConvolutionLayer(1, 1, 832, 32).setWeightsLog(layer11+conv5a),
        new ImgBandBiasLayer(32),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 32, 128).setWeightsLog(layer11+conv5b),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(pooling_4,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 832, 128).setWeightsLog(layer11+conv1b),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    val inception_5b = network.add(new ImgConcatLayer(),
      network.addAll(inception_5a,
        new ConvolutionLayer(1, 1, 832, 384).setWeightsLog(layer12+conv1a),
        new ImgBandBiasLayer(384),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_5a,
        new ConvolutionLayer(1, 1, 832, 192).setWeightsLog(layer12+conv3a),
        new ImgBandBiasLayer(192),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(3, 3, 192, 384).setWeightsLog(layer12+conv3b),
        new ImgBandBiasLayer(384),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_5a,
        new ConvolutionLayer(1, 1, 832, 48).setWeightsLog(layer12+conv5a),
        new ImgBandBiasLayer(48),
        new ActivationLayer(ActivationLayer.Mode.RELU),
        new ConvolutionLayer(5, 5, 48, 128).setWeightsLog(layer12+conv5b),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)),
      network.addAll(inception_5a,
        new PoolingLayer().setWindowXY(3, 3).setStrideXY(1, 1).setPaddingXY(1, 1),
        new ConvolutionLayer(1, 1, 832, 128).setWeightsLog(layer12+conv1b),
        new ImgBandBiasLayer(128),
        new ActivationLayer(ActivationLayer.Mode.RELU)))
    network.add(new PoolingLayer().setWindowXY(7, 7).setStrideXY(1,1).setPaddingXY(0, 0).setMode(PoolingMode.Avg))
    network.add(new DropoutNoiseLayer().setValue(0.4))
    network.add(new DenseSynapseLayer(Array(1,1,1024),Array(numberOfCategories)).setWeightsLog(layer13))
    val finalClassification = network.add("classify", new SoftmaxActivationLayer(), network.getHead)


    val entropy = network.add(new SumInputsLayer(),
      network.add(new EntropyLossLayer(), network.getInput(1), finalClassification),
      network.add(new EntropyLossLayer(), network.getInput(1), network.add(
        new SoftmaxActivationLayer(),
        network.add(
          new DenseSynapseLayer(Array(1024),Array(numberOfCategories)).setWeightsLog(trainingShunt),
          network.add(
            new DropoutNoiseLayer().setValue(0.7),
            network.add(
              new ReLuActivationLayer(),
              network.add(
                new DenseSynapseLayer(Array(3,3,128),Array(1024)).setWeightsLog(trainingShunt),
                network.add(
                  new ActivationLayer(ActivationLayer.Mode.RELU),
                  network.add(
                    new ConvolutionLayer(1, 1, 512, 128).setWeightsLog(trainingShunt),
                    network.add(
                      new PoolingLayer().setWindowXY(7, 7).setStrideXY(3,3).setMode(PoolingMode.Avg),
                      inception_4a
                    )
                  )
                )
              )
            )
          )
        )
      )),
      network.add(new EntropyLossLayer(), network.getInput(1), network.add(
        new SoftmaxActivationLayer(),
        network.add(
          new DenseSynapseLayer(Array(1024),Array(numberOfCategories)).setWeightsLog(trainingShunt),
          network.add(
            new DropoutNoiseLayer().setValue(0.7),
            network.add(
              new ReLuActivationLayer(),
              network.add(
                new DenseSynapseLayer(Array(3,3,128),Array(1024)).setWeightsLog(trainingShunt),
                network.add(
                  new ActivationLayer(ActivationLayer.Mode.RELU),
                  network.add(
                    new ConvolutionLayer(1, 1, 528, 128).setWeightsLog(trainingShunt),
                    network.add(
                      new PoolingLayer().setWindowXY(7, 7).setStrideXY(3,3).setMode(PoolingMode.Avg),
                      inception_4d
                    )
                  )
                )
              )
            )
          )
        )
      ))
    )

    if(fitness) {
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

  def fitness(monitor: TrainingMonitor, monitoringRoot : MonitoredObject, data: Array[Array[Tensor]], n: Int = 3) : Double = {
    val values = (1 to n).map(i ⇒ {
      val network = getNetwork(monitor, monitoringRoot, fitness = true)
      val measure = new ArrayTrainable(data, network, 1).measure()
      measure.value
    }).toList
    val avg = values.sum / n
    monitor.log(s"Numeric Opt: $this => $avg ($values)")
    avg
  }

}


class GoogleLeNetModeler(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  def run(awaitExit:Boolean=true): Unit = {
    defineHeader()
    declareTestHandler()
    out.out("<hr/>")
    if(findFile(modelName).isEmpty || System.getProperties.containsKey("rebuild")) {
      step_Generate()
      step_Train(trainingMin = 45)
      step_GAN()

      step_Train(trainingMin = 45)
      step_GAN()

      step_Train(trainingMin = 45)
      step_GAN()

      step_Train(trainingMin = 45)
      step_GAN()
    }
    step_Train(trainingMin = 45)
    step_GAN()
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

  def resize(source: BufferedImage, width: Int, height: Int) = {
    val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
    val graphics = image.getGraphics.asInstanceOf[Graphics2D]
    graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
    graphics.drawImage(source, 0, 0, width, height, null)
    image
  }

  def step_Generate() = phase({
    //new GoogLeNet().getNetwork(monitor, monitoringRoot, false)
    lazy val optTraining: Array[Array[Tensor]] = Random.shuffle(data.values.flatten).take(5).map(_.get()).toArray
    require(0 < optTraining.length)
    SimplexOptimizer[GoogLeNet](
      GoogLeNet(),
      x ⇒ x.fitness(monitor, monitoringRoot, optTraining, n=3), relativeTolerance=0.01
    ).getNetwork(monitor, monitoringRoot)
  }, (model: NNLayer) ⇒ {
    // Do Nothing
  }: Unit, modelName)


  def step_Train(trainingMin: Int = 15, sampleSize: Int = 250, iterationsPerSample: Int = 50) = phase(modelName, (model: NNLayer) ⇒ {
    out.h1("Integration Training")
    val trainer2 = out.eval {
      assert(null != data)
      var inner: Trainable = new StochasticArrayTrainable(takeData(2,imagesPerCategory).asJava,
        new SimpleLossNetwork(model, new EntropyLossLayer()), sampleSize, 20)
      val trainer = new IterativeTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(trainingMin, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(iterationsPerSample)
      trainer.setOrientation(new LBFGS)
      trainer.setLineSearchFactory(Java8Util.cvt((s:String)⇒(s match {
        case s if s.contains("LBFGS") ⇒ new StaticLearningRate().setRate(1.0)
        case _ ⇒ new ArmijoWolfeSearch().setAlpha(1e-5)
      })))
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    trainer2.run()
  }: Unit, modelName)

  def step_GAN() = phase(modelName, (model: NNLayer) ⇒ {
    val sourceClassId = 0
    val imageCount = 10
    out.h1("GAN Images Generation")
    val sourceClass = toOutNDArray(0, numberOfCategories)
    val targetClass = toOutNDArray(1, numberOfCategories)
    val adversarialData = data.values.flatten.map(_.get()).filter(x=>x(1).get(sourceClassId) > 0.9).map(x=>Array(x(0), targetClass))
    val adversarialOutput = new ArrayBuffer[Array[Tensor]]()
    val rows = adversarialData.take(imageCount).grouped(1).map(adversarialData => {
      val biasLayer = new BiasLayer(data.values.flatten.head.get().head.getDimensions(): _*)
      val trainingNetwork = new PipelineNetwork()
      trainingNetwork.add(biasLayer)
      trainingNetwork.add(KryoUtil.kryo().copy(model).freeze())

      val trainer1 = out.eval {
        assert(null != data)
        var inner: Trainable = new ArrayTrainable(adversarialData.toArray,
          new SimpleLossNetwork(trainingNetwork, new EntropyLossLayer()))
        val trainer = new IterativeTrainer(inner)
        trainer.setMonitor(monitor)
        trainer.setTimeout(1, TimeUnit.MINUTES)
        trainer.setOrientation(new GradientDescent)
        //trainer.setLineSearchFactory(Java8Util.cvt((s) ⇒ new ArmijoWolfeSearch().setMaxAlpha(1e8)))
        trainer.setLineSearchFactory(Java8Util.cvt((s) ⇒ new QuadraticSearch))
        trainer.setTerminateThreshold(0.01)
        trainer
      }
      trainer1.run()

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
    out.h1("GAN Images Training")
    val trainer2 = out.eval {
      assert(null != data)
      var inner: Trainable = new ArrayTrainable(adversarialOutput.toArray,
        new SimpleLossNetwork(model, new EntropyLossLayer()))
      val trainer = new IterativeTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(10, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(1000)
      trainer.setOrientation(new LBFGS)
      trainer.setLineSearchFactory(Java8Util.cvt((s) ⇒ new ArmijoWolfeSearch))
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    trainer2.run()

  }: Unit, modelName)


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
        TableOutput.create(takeData(5,10).map(_.get()).map(testObj ⇒ Map[String, AnyRef](
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
  lazy val (categoryList, data: Map[String, Stream[WeakCachedSupplier[Array[Tensor]]]]) = {
    out.p("Loading data from " + source)
    val (categoryList, data) = load()
    val categories: Map[String, Int] = categoryList.zipWithIndex.toMap
    out.p("<ol>" + categories.toList.sortBy(_._2).map(x ⇒ "<li>" + x + "</li>").mkString("\n") + "</ol>")
    out.eval {
      TableOutput.create(Random.shuffle(data.values.flatten.toList).take(100).map(_.get()).map(e ⇒ {
        Map[String, AnyRef](
          "Image" → out.image(e(0).toRgbImage(), e(1).toString),
          "Classification" → e(1)
        ).asJava
      }): _*)
    }
    out.p("Loading data complete")
    (categoryList, data)
  }

  def toOutNDArray(out: Int, max: Int): Tensor = {
    val ndArray = new Tensor(max)
    for (i <- 0 until max) ndArray.set(i, fuzz)
    ndArray.set(out, 1-(fuzz*(max-1)))
    ndArray
  }



  def takeData(numCategories : Int = 2, numImages : Int = 5000) = {
    Random.shuffle(Random.shuffle(data.toList).take(numCategories).map(_._2).flatten.toList).take(numImages)
  }

  def load(maxDim: Int = tileSize,
           imagesPerCategory: Int = imagesPerCategory
          ) = {
    val categoryDirs = Random.shuffle(new File(source).listFiles().toStream)
      .filter(dir => categoryWhitelist.isEmpty||categoryWhitelist.find(str => dir.getName.contains(str)).isDefined)
      .take(numberOfCategories)
    val categoryList = categoryDirs.map((categoryDirectory: File) ⇒ {
      categoryDirectory.getName.split('.').last
    })
    val categoryMap: Map[String, Int] = categoryList.zipWithIndex.toMap
    (categoryList, Random.shuffle(categoryDirs
      .map((categoryDirectory: File) ⇒ {
        val categoryName = categoryDirectory.getName.split('.').last
        categoryName -> Random.shuffle(categoryDirectory.listFiles().toStream).take(imagesPerCategory)
          .filterNot(_ == null).filterNot(_ == null)
          .map(file ⇒ {
            new WeakCachedSupplier[Array[Tensor]](Java8Util.cvt(()=>{
              val original = ImageIO.read(file)
              val fromWidth = original.getWidth()
              val fromHeight = original.getHeight()
              val scale = maxDim.toDouble / Math.min(fromWidth, fromHeight)
              val toWidth = ((fromWidth * scale).toInt)
              val toHeight = ((fromHeight * scale).toInt)
              val resized = new BufferedImage(maxDim, maxDim, BufferedImage.TYPE_INT_ARGB)
              val graphics = resized.getGraphics.asInstanceOf[Graphics2D]
              graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
              if (toWidth < toHeight) {
                graphics.drawImage(original, 0, (toWidth - toHeight) / 2, toWidth, toHeight, null)
              } else {
                graphics.drawImage(original, (toHeight - toWidth) / 2, 0, toWidth, toHeight, null)
              }
              Array(Tensor.fromRGB(resized), toOutNDArray(categoryMap(categoryName), categoryMap.size))
            }))
          })
      })).toMap)
  }
}

