import java.util.function.{IntToDoubleFunction, ToDoubleBiFunction, ToDoubleFunction}

/**
  * Created by Andrew Charneski on 5/12/2017.
  */
object AutoencoderUtil {

  implicit def cvt(fn:Int⇒Double) : IntToDoubleFunction = {
    new IntToDoubleFunction {
      override def applyAsDouble(v : Int): Double = fn(v)
    }
  }

  implicit def cvt[T](fn:T⇒Double) : ToDoubleFunction[T] = {
    new ToDoubleFunction[T] {
      override def applyAsDouble(v : T): Double = fn(v)
    }
  }

  implicit def cvt[T](fn:(T,T)⇒Double) : ToDoubleBiFunction[T,T] = {
    new ToDoubleBiFunction[T,T] {
      override def applyAsDouble(v : T, u : T): Double = fn(v, u)
    }
  }

}
