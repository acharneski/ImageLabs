import java.awt.{Graphics, Graphics2D}
import java.awt.image.BufferedImage
import java.io.{File, FileNotFoundException}
import java.util.function.Supplier

import com.simiacryptus.util.test.MarkdownPrintStream

class ScalaMarkdownPrintStream(file : File, name : String) extends MarkdownPrintStream(file, name) {
  def code[T](fn: ()=>T):T = {
    code(new Supplier[T] {
      override def get(): T = fn()
    }, 8*1024, 3)
  }
  def draw[T](fn: (Graphics2D) ⇒ Unit, width: Int = 600, height: Int = 400):BufferedImage = {
    code(new Supplier[BufferedImage] {
      override def get(): BufferedImage = {
        val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
        fn(image.getGraphics.asInstanceOf[Graphics2D])
        image
      }
    }, 8*1024, 3)
  }


}
trait MarkdownReporter {
  def report[T](methodName:String, fn:ScalaMarkdownPrintStream⇒T) : T = try {
    val className: String = getClass.getCanonicalName
    val path: File = new File(List("reports", className, methodName + ".md").mkString(File.separator))
    path.getParentFile.mkdirs
    val log = new ScalaMarkdownPrintStream(path, methodName)
    try {
      fn.apply(log)
    } finally {
      log.close()
    }
  } catch {
    case e: FileNotFoundException ⇒ {
      throw new RuntimeException(e)
    }
  }
}
