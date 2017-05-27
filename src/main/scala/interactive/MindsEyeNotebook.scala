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
import java.util.UUID
import java.util.concurrent.Semaphore

import scala.concurrent.ExecutionContext.Implicits.global
import com.fasterxml.jackson.databind.ObjectMapper
import com.simiacryptus.mindseye.opt.{IterativeTrainer, TrainingMonitor}
import com.simiacryptus.util.io.{HtmlNotebookOutput, TeeOutputStream}
import com.simiacryptus.util.text.TableOutput
import com.simiacryptus.util.{ArrayUtil, MonitoredObject, StreamNanoHTTPD}
import fi.iki.elonen.NanoHTTPD.IHTTPSession
import smile.plot.{PlotCanvas, ScatterPlot}
import util.{Java8Util, ScalaNotebookOutput}
import java.{lang, util}

import com.simiacryptus.mindseye.layers.NNLayer
import fi.iki.elonen.NanoHTTPD

import scala.collection.JavaConverters._
import scala.concurrent.Future
import ArrayUtil._

abstract class MindsEyeNotebook(server: StreamNanoHTTPD, log: HtmlNotebookOutput with ScalaNotebookOutput) {

  val history = new scala.collection.mutable.ArrayBuffer[IterativeTrainer.Step]()
  val logOut = new TeeOutputStream(log.file("log.txt"), true)
  val logPrintStream = new PrintStream(logOut)
  val monitoringRoot = new MonitoredObject()
  val dataTable = new TableOutput()
  def model: NNLayer

  val monitor = new TrainingMonitor {
    override def log(msg: String): Unit = {
      println(msg)
      logPrintStream.println(msg)
    }

    override def onStepComplete(currentPoint: IterativeTrainer.Step): Unit = {
      history += currentPoint
      val iteration = currentPoint.iteration
      if(shouldReplotMetrics(iteration)) regenerateReports()
      def flatten(prefix:String,data:Map[String,AnyRef]) : Map[String,AnyRef] = {
        data.flatMap({
          case (key, value) ⇒ value match {
            case value : Number if prefix.isEmpty ⇒ Map(key → value)
            case value : Number ⇒ Map((prefix + key) → value)
            case value : util.Map[String,AnyRef] ⇒ flatten(prefix+key+".", value.asScala.toMap)
            case value : Map[String,AnyRef] ⇒ flatten(prefix+key, value)
          }
        }).map(e⇒(if(e._1.startsWith(".")) e._1.substring(1) else e._1)→e._2)
      }
      dataTable.putRow((flatten(".",monitoringRoot.getMetrics.asScala.toMap)++Map(
        "epoch" → currentPoint.iteration.asInstanceOf[lang.Long],
        "time" → currentPoint.time.asInstanceOf[lang.Long],
        "value" → currentPoint.point.value.asInstanceOf[lang.Double]
      )).asJava)
    }
  }

  private def shouldReplotMetrics(iteration: Long) = iteration match {
    case _ if List(10,50).contains(iteration) ⇒ true
    case _ if 100 > iteration ⇒ false
    case _ if 0 == iteration % 100 ⇒ true
    case _ ⇒ false
  }

  def defineMonitorReports(log: HtmlNotebookOutput with ScalaNotebookOutput = log): Unit = {

    log.p("<a href='/history.html'>View the Convergence History</a>")
    server.addSyncHandler("history.html", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        summarizeHistory(log)
      })
    }), false)
    log.p("<a href='/metricsHistory.html'>View the Metrics History</a>")

    log.p("<a href='/log'>View the Log</a>")
    server.addSessionHandler("log", Java8Util.cvt((session : IHTTPSession)⇒{
      NanoHTTPD.newChunkedResponse(NanoHTTPD.Response.Status.OK, "text/plain", logOut.newInputStream())
    }))

    log.p("<p><a href='/netmon.json'>Sample Metric Values</a></p>")
    server.addSyncHandler("netmon.json", "application/json", Java8Util.cvt(out ⇒ {
      val mapper = new ObjectMapper().enableDefaultTyping(ObjectMapper.DefaultTyping.NON_FINAL)
      val buffer = new ByteArrayOutputStream()
      mapper.writeValue(buffer, monitoringRoot.getMetrics)
      out.write(buffer.toByteArray)
    }), false)

    log.p("<a href='/table.html'>View Metrics History Table</a>")
    server.addSyncHandler("table.html", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        log.h1("Parameter History Data Table")
        log.p(dataTable.toHtmlTable(true))
      })
    }), false)

  }


  def summarizeHistory(log: ScalaNotebookOutput = log) = {
    if (!history.isEmpty) {
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

  def regenerateReports() = {
    Option(log.file("../metricsHistory.html")).foreach(file⇒{
      val report = new HtmlNotebookOutput(log.workingDir, file) with ScalaNotebookOutput
      generateMetricsHistoryReport(report).andThen({case _⇒report.close()})
    })
    Option(log.file("../mobility.html")).foreach(file⇒{
      val report = new HtmlNotebookOutput(log.workingDir, file) with ScalaNotebookOutput
      generateMobilityReport(report).andThen({case _⇒report.close()})
    })
  }

  def generateMobilityReport(log: ScalaNotebookOutput = log): Future[Unit] = Future {
    if (!history.isEmpty) {
      val layers: Array[NNLayer] = history.flatMap(_.point.weights.map.asScala.keySet).distinct.toArray
      log.out("<table>")
      layers.foreach(layer ⇒ {
        try {
          val transcript: List[Array[Double]] = history.map(_.point.weights.map.get(layer).delta).toList
          log.out("<tr><td>")
          log.p(s"Layer ${layer.getClass}<br/>id ${layer.id}<br/>")
          List(1, 5, 20).foreach(lag ⇒ {
            log.out("</td><td>")
            val xy = (lag until transcript.size).map(i ⇒ {
              i → Math.log(Math.sqrt(magnitude(subtract(transcript(i), transcript(i - lag)))))
            }).filter(d ⇒ java.lang.Double.isFinite(d._2))
            if (xy.size > 1) log.eval {
              val plot: PlotCanvas = ScatterPlot.plot(xy.map(xy ⇒ Array(xy._1.toDouble, xy._2)): _*)
              plot.setTitle(s"${layer.getClass}/${layer.id}")
              plot.setAxisLabels("Epoch", s"log(dist(n,n-$lag))")
              plot.setSize(600, 400)
              plot
            } else log.out("No Data")
          })
          log.out("</td></tr>")
        } catch {
          case e: Throwable ⇒
        }
      })
      log.out("</table>")
    }
  }

  def generateMetricsHistoryReport(log: ScalaNotebookOutput = log): Future[Unit] = {
    if(!history.isEmpty) {
      val dataAsScala: Array[Map[String, AnyRef]] = dataTable.rows.asScala.map(_.asScala.toMap).toArray
      val keys: Array[String] = dataTable.schema.asScala.keySet.toArray

      Future {
        log.out("<table><tr><th>Vs Iteration</th><th>Vs Objective</th></tr>")
        keys
          .filterNot(_.contains("Performance"))
          .filterNot(_.contains("PerItem"))
          .filterNot(_.contains("count"))
          .sorted
          .foreach(key⇒{
            log.out("<tr><td>")
            try {
              val data = dataAsScala.map(row ⇒ Array[Double](
                row("epoch").toString.toDouble, row(key).toString.toDouble
              )).filter(d ⇒ d.forall(java.lang.Double.isFinite))
              if(data.size > 1) log.eval {
                val plot: PlotCanvas = ScatterPlot.plot(data: _*)
                //plot.setTitle(s"$key vs Epoch")
                plot.setAxisLabels("Epoch", key)
                plot.setSize(600, 400)
                plot
              } else log.out("No Data")
            } catch {
              case e : Throwable ⇒
            }
            log.out("</td><td>")
            try {
              val data = dataAsScala.map(row ⇒ Array[Double](
                Math.log(row("value").toString.toDouble), row(key).toString.toDouble
              )).filter(d ⇒ d.forall(java.lang.Double.isFinite))
              if(data.size > 1) log.eval {
                val plot: PlotCanvas = ScatterPlot.plot(data: _*)
                //plot.setTitle(s"$key vs Epoch")
                plot.setAxisLabels("log(value)", key)
                plot.setSize(600, 400)
                plot
              } else log.out("No Data")
            } catch {
              case e : Throwable ⇒
            }
            log.out("</td></tr>")
          })
        log.out("</table>")
      }
    } else Future.successful()
  }

  def waitForExit(log: HtmlNotebookOutput with ScalaNotebookOutput = log): Unit = {
    logOut.close()
    val onExit = new Semaphore(0)
    log.p("To exit the sever: <a href='/exit'>/exit</a>")
    server.addAsyncHandler("exit", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        log.h1("OK")
        onExit.release(1)
      })
    }), false)
    onExit.acquire()
  }

}
