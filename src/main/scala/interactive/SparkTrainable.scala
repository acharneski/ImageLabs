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

import com.simiacryptus.mindseye.net.{DeltaSet, NNLayer, NNResult}
import com.simiacryptus.mindseye.opt.Trainable
import com.simiacryptus.util.ml.Tensor
import org.apache.spark.rdd.RDD
import scala.collection.JavaConverters._


case class ReduceableResult(
                           deltas : Map[String,Array[Double]],
                           sum : Double,
                           count : Int
                           ) {
  def meanValue: Double = sum / count

  def accumulate(source: DeltaSet) = {
    val idIndex: Map[String, NNLayer] = source.map.asScala.map(e⇒{
      e._1.id.toString → e._1
    }).toMap
    deltas.foreach((e)⇒{
      source.get(idIndex(e._1), null:Array[Double]).accumulate(e._2)
    })
    source
  }

  def +(right:ReduceableResult) : ReduceableResult = {
    new ReduceableResult(
      (deltas.keySet ++ right.deltas.keySet).map(key⇒{
        val l = deltas.get(key)
        val r = right.deltas.get(key)
        if (r.isDefined) {
          if (l.isDefined) {
            val larray: Array[Double] = l.get
            val rarray: Array[Double] = r.get
            assert(larray.length == rarray.length)
            key → larray.zip(rarray).map(x ⇒ x._1 + x._2)
          } else {
            key → r.get
          }
        } else {
          key → l.get
        }
      }).toMap,
      sum + right.sum,
      count + right.count
    )
  }

}

class SparkTrainable(dataRDD: RDD[Array[Tensor]], network: NNLayer) extends Trainable with Serializable {

  def getResult(delta: DeltaSet, values: Array[Double]) : ReduceableResult = {
    new ReduceableResult(
      delta.map.asScala.map(e⇒{
        val (k,v) = e
        k.id.toString → v.delta
      }).toMap,
      values.reduce(_ + _),
      values.length
    )
  }

  def getDelta(reduce: ReduceableResult): DeltaSet = {
    val deltaSet: DeltaSet = new DeltaSet()
    val prototype = dataRDD.take(1).head
    val input = NNResult.batchResultArray(Array(prototype))
    val result = network.eval(input: _*)
    result.accumulate(deltaSet, 0)
    reduce.accumulate(deltaSet)
    deltaSet
  }

  override def measure(): Trainable.PointSample = {
    val reduce = dataRDD.mapPartitions(partition ⇒ {
      val input = NNResult.batchResultArray(partition.toArray)
      val result = network.eval(input: _*)
      val deltaSet: DeltaSet = new DeltaSet
      result.accumulate(deltaSet)
      val doubles = result.data.map(_.getData()(0))
      List(getResult(deltaSet, doubles)).iterator
    }).reduce(_ + _)

    val deltaSet: DeltaSet = getDelta(reduce)

    val stateSet = new DeltaSet
    deltaSet.map.asScala.foreach(e ⇒ {
      val (layer, layerDelta) = e
      stateSet.get(layer, layerDelta.target).accumulate(layerDelta.target)
    })
    return new Trainable.PointSample(deltaSet, stateSet, reduce.meanValue);
  }

}
