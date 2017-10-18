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

import java.awt.event.AWTEventListener
import java.awt.image.BufferedImage
import java.awt.{AWTEvent, Graphics2D, RenderingHints}
import java.io._
import java.lang
import java.util.concurrent.TimeUnit
import java.util.function.{DoubleSupplier, IntToDoubleFunction}
import javax.imageio.ImageIO

import _root_.util.Java8Util.cvt
import _root_.util._
import com.simiacryptus.mindseye.lang.{NNLayer, Tensor}
import com.simiacryptus.mindseye.layers.activation.{AbsActivationLayer, SoftmaxActivationLayer}
import com.simiacryptus.mindseye.layers.cudnn.f32._
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer
import com.simiacryptus.mindseye.layers.media.MaxImageBandLayer
import com.simiacryptus.mindseye.layers.meta.StdDevMetaLayer
import com.simiacryptus.mindseye.layers.reducers.{AvgReducerLayer, ProductInputsLayer, SumInputsLayer}
import com.simiacryptus.mindseye.layers.util.ConstNNLayer
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.network.graph.DAGNode
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line._
import com.simiacryptus.mindseye.opt.orient._
import com.simiacryptus.mindseye.opt.trainable._
import com.simiacryptus.util.io.HtmlNotebookOutput
import com.simiacryptus.util.text.TableOutput
import com.simiacryptus.util.{MonitoredObject, StreamNanoHTTPD, Util}
import util.NNLayerUtil._

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object Activity extends Report {

  def main(args: Array[String]): Unit = {
    new Activity().run()
    Thread.sleep(60*1000*5)
  }

}
import interactive.classify.ClassifierModeler._



class Activity() {
  def run(): Unit = {
    import java.awt.Toolkit
    Toolkit.getDefaultToolkit.addAWTEventListener(new AWTEventListener() {
      override def eventDispatched(event: AWTEvent): Unit = {
        println(event.toString)
      }
    },
      //-1)
      AWTEvent.COMPONENT_EVENT_MASK |
        AWTEvent.CONTAINER_EVENT_MASK |
        AWTEvent.FOCUS_EVENT_MASK |
        AWTEvent.KEY_EVENT_MASK |
        AWTEvent.MOUSE_EVENT_MASK |
        AWTEvent.MOUSE_MOTION_EVENT_MASK |
        AWTEvent.WINDOW_EVENT_MASK |
        AWTEvent.ACTION_EVENT_MASK |
        AWTEvent.ADJUSTMENT_EVENT_MASK |
        AWTEvent.ITEM_EVENT_MASK |
        AWTEvent.TEXT_EVENT_MASK |
        AWTEvent.INPUT_METHOD_EVENT_MASK |
        AWTEvent.PAINT_EVENT_MASK |
        AWTEvent.INVOCATION_EVENT_MASK |
        AWTEvent.HIERARCHY_EVENT_MASK |
        AWTEvent.HIERARCHY_BOUNDS_EVENT_MASK |
        AWTEvent.MOUSE_WHEEL_EVENT_MASK |
        AWTEvent.WINDOW_STATE_EVENT_MASK |
        AWTEvent.WINDOW_FOCUS_EVENT_MASK)
  }
}

