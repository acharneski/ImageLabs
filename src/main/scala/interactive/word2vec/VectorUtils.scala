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

package interactive.word2vec

object VectorUtils {
  implicit def convert(values: Array[Float]) = new VectorUtils(values)
}
import VectorUtils._

case class VectorUtils(values: Array[Float]) {
  def +(right: Array[Float]): Array[Float] = values.zip(right).map(x => x._1 + x._2)

  def -(right: Array[Float]): Array[Float] = values.zip(right).map(x => x._1 - x._2)

  def *(right: Array[Float]): Array[Float] = values.zip(right).map(x => x._1 * x._2)

  def *(right: Float): Array[Float] = values.map(x => x * right)

  def /(right: Float): Array[Float] = values.map(x => x / right)

  def unitV: Array[Float] = {
    val a = this / l1
    a / a.magnitude
  }

  def sq: Array[Float] = values.map(x => x * x)

  def l0: Float = values.size.toFloat

  def l1: Float = values.sum

  def l2: Float = Math.sqrt(sq.sum).toFloat

  def max: Float = values.reduce(Math.max(_, _))

  def min: Float = values.reduce(Math.min(_, _))

  def mean: Float = l1 / l0

  def magnitude: Float = l2

  def withMath: VectorUtils = this

  def ^(right: Array[Float]): Double = Math.acos((this.unitV * right.unitV).l1)

  def ~(right: Array[Float]): Double = (this - right).magnitude

  def along(right: Array[Float]): Array[Float] = right.unitV * (this * right.unitV)

  def without(right: Array[Float]) = this - (this along right)
}