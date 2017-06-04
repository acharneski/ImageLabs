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

import com.simiacryptus.mindseye.layers.NNLayer
import com.simiacryptus.mindseye.layers.activation.{LinearActivationLayer, NthPowerActivationLayer, SqActivationLayer}
import com.simiacryptus.mindseye.layers.meta.{AvgMetaLayer, BiasMetaLayer, ScaleMetaLayer, ScaleUniformMetaLayer}
import com.simiacryptus.mindseye.layers.reducers.AvgReducerLayer
import com.simiacryptus.mindseye.network.PipelineNetwork

object NetworkMetaNormalizers {

  def positionNormalizer2 = {
    var model: PipelineNetwork = new PipelineNetwork
    val input = model.getInput(0)
    model.add(new AvgReducerLayer(), input)
    val means = model.add(new AvgMetaLayer(), input)
    model.add(new BiasMetaLayer(), input, model.add(new LinearActivationLayer().setScale(-1).freeze(), means))
    model
  }

  def positionNormalizer = {
    var model: PipelineNetwork = new PipelineNetwork
    val input = model.getInput(0)
    val means = model.add(new AvgMetaLayer(), input)
    model.add(new BiasMetaLayer(), input, model.add(new LinearActivationLayer().setScale(-1).freeze(), means))
    model
  }

  def scaleNormalizer2 = {
    var model: PipelineNetwork = new PipelineNetwork
    val input = model.getInput(0)
    model.add(new SqActivationLayer(), input)
    model.add(new AvgReducerLayer())
    val variances = model.add(new AvgMetaLayer())
    model.add(new ScaleUniformMetaLayer(), input, model.add(new NthPowerActivationLayer().setPower(-0.5), variances))
    model
  }

  def scaleNormalizer = {
    var model: PipelineNetwork = new PipelineNetwork
    val input = model.getInput(0)
    model.add(new SqActivationLayer(), input)
    val variances = model.add(new AvgMetaLayer(), input)
    model.add(new ScaleMetaLayer(), input, model.add(new NthPowerActivationLayer().setPower(-0.5), variances))
    model
  }

}
