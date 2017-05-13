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

import java.util.function.{IntToDoubleFunction, ToDoubleBiFunction, ToDoubleFunction}

/**
  * Created by Andrew Charneski on 5/12/2017.
  */
object AutoencoderUtil {

  implicit def cvt(fn: Int ⇒ Double): IntToDoubleFunction = {
    new IntToDoubleFunction {
      override def applyAsDouble(v: Int): Double = fn(v)
    }
  }

  implicit def cvt[T](fn: T ⇒ Double): ToDoubleFunction[T] = {
    new ToDoubleFunction[T] {
      override def applyAsDouble(v: T): Double = fn(v)
    }
  }

  implicit def cvt[T](fn: (T, T) ⇒ Double): ToDoubleBiFunction[T, T] = {
    new ToDoubleBiFunction[T, T] {
      override def applyAsDouble(v: T, u: T): Double = fn(v, u)
    }
  }

}
