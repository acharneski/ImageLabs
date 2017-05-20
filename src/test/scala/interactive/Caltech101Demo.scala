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

import java.io.{ByteArrayOutputStream, PrintStream}
import java.lang
import java.util.concurrent.Semaphore

import com.fasterxml.jackson.databind.ObjectMapper
import com.simiacryptus.mindseye.net.util.MonitoredObject
import com.simiacryptus.mindseye.opt.{IterativeTrainer, TrainingMonitor}
import com.simiacryptus.util.StreamNanoHTTPD
import com.simiacryptus.util.io.{HtmlNotebookOutput, TeeOutputStream}
import com.simiacryptus.util.text.TableOutput
import fi.iki.elonen.NanoHTTPD
import fi.iki.elonen.NanoHTTPD.IHTTPSession
import smile.plot.{PlotCanvas, ScatterPlot}
import util._

import scala.collection.JavaConverters._


object Caltech101Demo extends ServiceNotebook {

  def main(args: Array[String]): Unit = {
    report(new Caltech101Demo().run)
    System.exit(0)
  }
}
class Caltech101Demo {

  def run(server: StreamNanoHTTPD, log: HtmlNotebookOutput with ScalaNotebookOutput) {
    val inputSize = Array[Int](256, 256, 3)
    log.h1("Caltech 101")
    val history = new scala.collection.mutable.ArrayBuffer[IterativeTrainer.Step]()
    log.p("View the convergence history: <a href='/history.html'>/history.html</a>")
    server.addHandler("history.html", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        summarizeHistory(log, history.toList)
      })
    }), false)
    val monitoringRoot = new MonitoredObject()
    log.p("<a href='/netmon.json'>Network Monitoring</a>")
    server.addHandler("netmon.json", "application/json", Java8Util.cvt(out ⇒ {
      val mapper = new ObjectMapper().enableDefaultTyping(ObjectMapper.DefaultTyping.NON_FINAL)
      val buffer = new ByteArrayOutputStream()
      mapper.writeValue(buffer, monitoringRoot.getMetrics)
      out.write(buffer.toByteArray)
    }), false)
    log.p("View the log: <a href='/log'>/log</a>")
    val logOut = new TeeOutputStream(log.file("log.txt"), true)
    val logPrintStream = new PrintStream(logOut)
    server.addHandler2("log", Java8Util.cvt((session : IHTTPSession)⇒{
      NanoHTTPD.newChunkedResponse(NanoHTTPD.Response.Status.OK, "text/plain", logOut.newInputStream())
    }))
    val monitor = new TrainingMonitor {
      override def log(msg: String): Unit = {
        logPrintStream.println(msg);
      }

      override def onStepComplete(currentPoint: IterativeTrainer.Step): Unit = {
        history += currentPoint
      }
    }
    log.out("<hr/>")



    log.out("<hr/>")
    logOut.close()
    val onExit = new Semaphore(0)
    log.p("To exit the sever: <a href='/exit'>/exit</a>")
    server.addHandler("exit", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        log.h1("OK")
        onExit.release(1)
      })
    }), false)
    onExit.acquire()
  }

  private def summarizeHistory(log: ScalaNotebookOutput, history: List[com.simiacryptus.mindseye.opt.IterativeTrainer.Step]) = {
    if(!history.isEmpty) {
      log.eval {
        val step = Math.max(Math.pow(10, Math.ceil(Math.log(history.size) / Math.log(10)) - 2), 1).toInt
        TableOutput.create(history.filter(0 == _.iteration % step).map(state ⇒
          Map[String, AnyRef](
            "iteration" → state.iteration.toInt.asInstanceOf[Integer],
            "time" → state.time.toDouble.asInstanceOf[lang.Double],
            "fitness" → state.point.value.toDouble.asInstanceOf[lang.Double]
          ).asJava
        ): _*)
      }
      log.eval {
        val plot: PlotCanvas = ScatterPlot.plot(history.map(item ⇒ Array[Double](
          item.iteration, Math.log(item.point.value)
        )).toArray: _*)
        plot.setTitle("Convergence Plot")
        plot.setAxisLabels("Iteration", "log(Fitness)")
        plot.setSize(600, 400)
        plot
      }
    }
  }

}