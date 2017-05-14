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

import java.awt.image.BufferedImage
import java.awt.{Graphics2D, RenderingHints}
import java.io.{File, FileNotFoundException}
import java.util.function.Supplier

import com.simiacryptus.util.io.{MarkdownPrintStream, NotebookOutput}

trait MarkdownReporter {
  def report[T](methodName: String, fn: ScalaNotebookOutput ⇒ T): T = try {
    MarkdownReporter.currentMethod = methodName
    val className: String = getClass.getCanonicalName
    val path: File = new File(List("reports", className, methodName + ".md").mkString(File.separator))
    path.getParentFile.mkdirs
    val log = new MarkdownPrintStream(path, methodName) with ScalaNotebookOutput
    log.addCopy(System.out)
    try {
      fn.apply(log)
    } finally {
      log.close()
    }
  } catch {
    case e: FileNotFoundException ⇒ {
      throw new RuntimeException(e)
    }
  } finally {
    MarkdownReporter.currentMethod = null
  }
}

trait ScalaNotebookOutput extends NotebookOutput {

  def eval[T](fn: => T): T = {
    code(new Supplier[T] {
      override def get(): T = fn
    }, 8 * 1024, 3)
  }

  def code[T](fn: () => T): T = {
    code(new Supplier[T] {
      override def get(): T = fn()
    }, 8 * 1024, 3)
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
    }, 8 * 1024, 3)
  }

}

object MarkdownReporter {
  var currentMethod: String = null
}
