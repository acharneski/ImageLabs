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
import java.util.concurrent.TimeUnit

import _root_.util._
import com.simiacryptus.mindseye.data.ImageTiles.ImageTensorLoader
import com.simiacryptus.mindseye.eval.ArrayTrainable
import com.simiacryptus.mindseye.lang.{NNLayer, Tensor}
import com.simiacryptus.mindseye.layers.java._
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line._
import com.simiacryptus.mindseye.opt.orient._
import com.simiacryptus.text.TableOutput
import com.simiacryptus.util.StreamNanoHTTPD
import com.simiacryptus.util.io.HtmlNotebookOutput

import scala.collection.JavaConverters._
import scala.util.Random

class UpsamplingOptimizer(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  val tileSize = 64
  val fitnessBorderPadding = 8
  val scaleFactor: Double = (64 * 64.0) / (tileSize * tileSize)
  val sampleTiles = 1000

  lazy val rawData: Array[Tensor] = {
    val loader = new ImageTensorLoader(new File(source), tileSize, tileSize, tileSize, tileSize, 10, 10)
    val data = loader.stream().iterator().asScala.take(sampleTiles).toArray
    loader.stop()
    data
  }
  lazy val discriminatorNetwork = loadModel("descriminator_1")
  lazy val forwardNetwork = loadModel("downsample_1")

  def run(awaitExit:Boolean=true): Unit = {
    defineHeader()
    out.out("<hr/>")
    Random.shuffle(rawData.toList).grouped(10).map(data ⇒ {
      out.eval {
        TableOutput.create(data.map(original ⇒ {
          val downsampled = Tensor.fromRGB(UpsamplingOptimizer.resize(original.toRgbImage, tileSize / 4))
          val reconstructed = UpsamplingOptimizer.reconstructImage(forwardNetwork, discriminatorNetwork, downsampled, monitor)
          Map[String, AnyRef](
            "Original Image" → out.image(original.toRgbImage, ""),
            "Downsampled" → out.image(downsampled.toRgbImage, ""),
            "Reconsructed" → out.image(reconstructed.toRgbImage, "")
          ).asJava
        }): _*)
      }
    })
    out.out("<hr/>")
    if(awaitExit) waitForExit()
  }

}

object UpsamplingOptimizer extends Report {

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒ new UpsamplingOptimizer(source, server, out).run()
      case _ ⇒ new UpsamplingOptimizer("E:\\testImages\\256_ObjectCategories", server, out).run()
    })
  }

  def reconstructImage(forwardModel: NNLayer, discriminatorModel: NNLayer, originalTensor: Tensor, monitor: TrainingMonitor, scaleFactor: Int = 4): Tensor = {

    val network = new PipelineNetwork(0)

    val targetTensor = Tensor.fromRGB(resize(originalTensor.toRgbImage, originalTensor.getDimensions.head * scaleFactor))
    val targetNode = network.constValue(targetTensor)
    targetNode.getLayer.asInstanceOf[ConstNNLayer].setFrozen(false)

    val wrongness = network.add(new MeanSqLossLayer(), network.add(forwardModel.freeze(), targetNode), network.constValue(originalTensor))

    val fakeness = network.add(new EntropyLossLayer(), network.add(discriminatorModel.freeze(), targetNode), network.constValue(new Tensor(3).set(0, 1)))

    network.add(new ProductInputsLayer(), fakeness, network.add(new SumInputsLayer(), wrongness, network.constValue(new Tensor(1).set(0, 1))))
    assert(!targetNode.getLayer.asInstanceOf[ConstNNLayer].isFrozen)

    val executorFunction = new ArrayTrainable(Array(Array()), network, 0)
    val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(executorFunction)
    trainer.setLineSearchFactory(Java8Util.cvt((s) ⇒ new ArmijoWolfeSearch()
      .setC1(0).setC2(1).setStrongWolfe(false).setMaxAlpha(1e8)))
    trainer.setOrientation(new LBFGS)
    trainer.setMonitor(monitor)
    trainer.setTimeout(1, TimeUnit.MINUTES)
    trainer.setTerminateThreshold(1e-4)
    trainer.setMaxIterations(100)
    trainer.run()

    targetTensor
  }

  def resize(source: BufferedImage, size: Int) = {
    val image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB)
    val graphics = image.getGraphics.asInstanceOf[Graphics2D]
    graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
    graphics.drawImage(source, 0, 0, size, size, null)
    image
  }


}
