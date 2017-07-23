package interactive.superres

import util.Report

/**
  * Created by Andrew Charneski on 7/22/2017.
  */
object ModelScript extends Report {

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒
        new DownsamplingModel(source, server, out).run()
        new UpsamplingOptimizer(source, server, out).run()
        new UpsamplingModel(source, server, out).run()
      case _ ⇒
        new DownsamplingModel("E:\\testImages\\256_ObjectCategories", server, out).run()
        new UpsamplingOptimizer("E:\\testImages\\256_ObjectCategories", server, out).run()
        new UpsamplingModel("E:\\testImages\\256_ObjectCategories", server, out).run()
    })

  }

}
