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
import java.util.UUID

import com.simiacryptus.util.io.HtmlNotebookOutput
import com.simiacryptus.util.{StreamNanoHTTPD, Util}

/**
  * Created by Andrew Charneski on 5/14/2017.
  */
trait ServiceNotebook {
  def report[T](fn: (StreamNanoHTTPD, HtmlNotebookOutput with ScalaNotebookOutput) ⇒ T,
                port: Int = 0x1FF + (Math.random() * 0x700).toInt): T = try {
    val path = new File(Util.mkString(File.separator, "www", UUID.randomUUID.toString))
    path.mkdirs
    val logFile = new File(path, "index.html")
    //val port: Int = 0x1FF + (Math.random() * 0x700).toInt
    val server = new StreamNanoHTTPD(port, "text/html", logFile).init()
    val log = new HtmlNotebookOutput(path, server.dataReciever) with ScalaNotebookOutput
    log.addCopy(System.out)
    try {
      fn(server, log)
    } finally {
      log.close()
    }
  } catch {
    case e: FileNotFoundException ⇒ {
      throw new RuntimeException(e)
    }
  }
}
