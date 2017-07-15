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

import java.util.concurrent.TimeUnit

import _root_.util._
import com.simiacryptus.mindseye.layers.NNLayer
import com.simiacryptus.mindseye.layers.activation.{ReLuActivationLayer, SoftmaxActivationLayer}
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer
import com.simiacryptus.mindseye.layers.media.{ImgConvolutionSynapseLayer, MaxSubsampleLayer}
import com.simiacryptus.mindseye.layers.synapse.{BiasLayer, DenseSynapseLayer}
import com.simiacryptus.mindseye.layers.util.{MonitoringSynapse, MonitoringWrapper}
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy
import com.simiacryptus.mindseye.opt.region.{GrowthSphere, LinearSumConstraint, TrustRegion}
import com.simiacryptus.mindseye.opt.trainable.{L12Normalizer, ScheduledSampleTrainable}
import com.simiacryptus.util.ml.Tensor

import scala.util.Random


object ConvolutionalMnistDemo extends ServiceNotebook {

  def main(args: Array[String]): Unit = {
    report((server,log)⇒new MnistDemo(server,log){
      override val trainingTime = 5

      model = log.eval {
        var model: PipelineNetwork = new PipelineNetwork
        model.add(new MonitoringWrapper(new BiasLayer(inputSize: _*)).addTo(monitoringRoot, "inbias"))
        model.add(new MonitoringWrapper(new ImgConvolutionSynapseLayer(3, 3, 8)
          .setWeights(Java8Util.cvt(() ⇒ 0.1 * (Random.nextDouble() - 0.5)))).addTo(monitoringRoot, "synapse1"))
        model.add(new MonitoringWrapper(new MaxSubsampleLayer(4, 4, 1)).addTo(monitoringRoot, "max1"))
        //model.add(new MonitoringWrapper(new ImgBandBiasLayer(28,28,8)).addTo(monitoringRoot, "imgbias1"))
        model.add(new MonitoringWrapper(new ReLuActivationLayer).addTo(monitoringRoot, "relu1"))
        model.add(new MonitoringSynapse().addTo(monitoringRoot, "hidden1"))

        model.add(new MonitoringWrapper(new DenseSynapseLayer(Array[Int](7, 7, 8), outputSize)
          .setWeights(Java8Util.cvt(() ⇒ 0.001 * (Random.nextDouble() - 0.5)))).addTo(monitoringRoot, "synapse3"))

        model.add(new MonitoringWrapper(new ReLuActivationLayer).addTo(monitoringRoot, "relu3"))
        model.add(new MonitoringSynapse().addTo(monitoringRoot, "output"))
        model.add(new MonitoringWrapper(new BiasLayer(outputSize: _*)).addTo(monitoringRoot, "outbias"))
        model.add(new SoftmaxActivationLayer)
        model
      }

      override def buildTrainer(data: Seq[Array[Tensor]]): Stream[IterativeTrainer] = Stream(log.eval {
        val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
        val trainable = ScheduledSampleTrainable.Pow(data.toArray, trainingNetwork, 100, 10, 0).setShuffled(true)
        val normalized = new L12Normalizer(trainable) {
          override protected def getL1(layer: NNLayer): Double = layer match {
            case _: BiasLayer ⇒ 0
            case _: DenseSynapseLayer ⇒ 0.01
            case _: ImgConvolutionSynapseLayer ⇒ 0.01
          }
          override protected def getL2(layer: NNLayer): Double = 0
        }
        val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(normalized)
        trainer.setMonitor(monitor)
        trainer.setOrientation(new TrustRegionStrategy() {
          override def getRegionPolicy(layer: NNLayer): TrustRegion = layer match {
            case _: BiasLayer ⇒ null //new SingleOrthant()
            case _: DenseSynapseLayer ⇒ new LinearSumConstraint()
            case _: ImgConvolutionSynapseLayer ⇒ new GrowthSphere().setGrowthFactor(0.0).setMinRadius(0.01)
            case _ ⇒ null
          }
        });
        trainer.setTimeout(trainingTime, TimeUnit.MINUTES)
        trainer.setTerminateThreshold(0.0)
        trainer
      })

    }.run)
    System.exit(0)
  }
}

