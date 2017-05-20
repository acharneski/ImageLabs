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

package util

import java.awt.image.BufferedImage
import java.awt.{Graphics2D, RenderingHints}
import java.util.function.Supplier

import com.simiacryptus.util.io.NotebookOutput

/**
  * Created by Andrew Charneski on 5/14/2017.
  */
trait ScalaNotebookOutput extends NotebookOutput {

  def eval[T](fn: => T): T = {
    code(new Supplier[T] {
      override def get(): T = fn
    }, 8 * 1024, 4)
  }

  def code[T](fn: () => T): T = {
    code(new Supplier[T] {
      override def get(): T = fn()
    }, 8 * 1024, 4)
  }

  def draw[T](fn: (Graphics2D) ⇒ Unit, width: Int = 600, height: Int = 400): BufferedImage = {
    code(new Supplier[BufferedImage] {
      override def get(): BufferedImage = {
        val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
        val graphics = image.getGraphics.asInstanceOf[Graphics2D]
        graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
        fn(graphics)
        image
      }
    }, 8 * 1024, 4)
  }

}
