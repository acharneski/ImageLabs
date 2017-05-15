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

import java.io.{File, FileNotFoundException}

import com.simiacryptus.util.io.MarkdownNotebookOutput

object ReportNotebook {
  var currentMethod: String = null
}

trait ReportNotebook {
  def report[T](methodName: String, fn: ScalaNotebookOutput ⇒ T): T = try {
    ReportNotebook.currentMethod = methodName
    val className: String = getClass.getCanonicalName
    val path: File = new File(List("reports", className, methodName + ".md").mkString(File.separator))
    path.getParentFile.mkdirs
    val log = new MarkdownNotebookOutput(path, methodName) with ScalaNotebookOutput
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
    ReportNotebook.currentMethod = null
  }
}


