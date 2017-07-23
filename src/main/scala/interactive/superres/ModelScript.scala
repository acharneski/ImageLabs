package interactive.superres

import util.Report

/**
  * Created by Andrew Charneski on 7/22/2017.
  */
object ModelScript extends Report {

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒
        new DownsamplingModel(source, server, out).run(false)
        new BicubicDiscriminatorModel(source, server, out).run(false)
        new UpsamplingOptimizer(source, server, out).run(false)
        new UpsamplingModel(source, server, out).run(false)
      case _ ⇒
        new DownsamplingModel("E:\\testImages\\256_ObjectCategories", server, out).run(false)
        new BicubicDiscriminatorModel("E:\\testImages\\256_ObjectCategories", server, out).run(false)
        new UpsamplingOptimizer("E:\\testImages\\256_ObjectCategories", server, out).run(false)
        new UpsamplingModel("E:\\testImages\\256_ObjectCategories", server, out).run(false)
    })

  }

}
