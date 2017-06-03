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

import java.util.function.{Consumer, DoubleSupplier, DoubleUnaryOperator, Function, IntToDoubleFunction, Supplier, ToDoubleBiFunction, ToDoubleFunction}

object Java8Util {

  implicit def cvt(fn: Int ⇒ Double): IntToDoubleFunction = {
    new IntToDoubleFunction {
      override def applyAsDouble(v: Int): Double = fn(v)
    }
  }

  implicit def cvt[T <: AnyRef](fn: () ⇒ T): Supplier[T] = {
    new Supplier[T] {
      override def get(): T = fn.apply()
    }
  }

  implicit def cvt[T <: AnyRef](fn: () ⇒ Double): DoubleSupplier = {
    new DoubleSupplier {
      override def getAsDouble: Double = fn.apply()
    }
  }

  implicit def cvt[T](fn: T ⇒ Double): ToDoubleFunction[T] = {
    new ToDoubleFunction[T] {
      override def applyAsDouble(v: T): Double = fn(v)
    }
  }

  implicit def cvt(fn: Double ⇒ Double): DoubleUnaryOperator = {
    new DoubleUnaryOperator {
      override def applyAsDouble(v: Double): Double = fn(v)
    }
  }

  implicit def cvt[T](fn: T ⇒ Unit): Consumer[T] = {
    new Consumer[T] {
      override def accept(t: T): Unit = fn(t)
    }
  }

  implicit def cvt[T,U](fn: T ⇒ U): Function[T, U] = {
    new Function[T, U] {
      override def apply(v1: T): U = fn(v1)
    }
  }

  implicit def cvt[T,U](fn: (T, U) ⇒ Double): ToDoubleBiFunction[T, U] = {
    new ToDoubleBiFunction[T, U] {
      override def applyAsDouble(v: T, u: U): Double = fn(v, u)
    }
  }

}
