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

package interactive.classify

import java.awt.image.BufferedImage
import java.awt.{Graphics2D, RenderingHints}
import java.io._
import java.util.concurrent.TimeUnit
import java.util.stream.Collectors
import javax.imageio.ImageIO

import _root_.util.Java8Util.cvt
import _root_.util._
import com.simiacryptus.mindseye.eval.{ArrayTrainable, SampledArrayTrainable, Trainable}
import com.simiacryptus.mindseye.lang.{NNExecutionContext, NNLayer, NNResult, Tensor}
import com.simiacryptus.mindseye.layers.cudnn.f32
import com.simiacryptus.mindseye.layers.cudnn.f32.PoolingLayer.PoolingMode
import com.simiacryptus.mindseye.layers.cudnn.f32._
import com.simiacryptus.mindseye.layers.java._
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork}
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.mindseye.opt.line._
import com.simiacryptus.mindseye.opt.orient._
import com.simiacryptus.text.TableOutput
import com.simiacryptus.util.StreamNanoHTTPD
import com.simiacryptus.util.function.WeakCachedSupplier
import com.simiacryptus.util.io.{HtmlNotebookOutput, KryoUtil}

import scala.collection.JavaConverters._
import scala.collection.immutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object IncrementalClassifierModeler extends Report {
  val modelName = System.getProperty("modelName","incremental_classifier_3")
  val tileSize = 256
  val categoryWhitelist = Set[String]()//"greyhound", "soccer-ball", "telephone-box", "windmill")
  val numberOfCategories = 10
  val fuzz = 1e-2
  val imagesPerCategory = 1000

  def main(args: Array[String]): Unit = {

    report((server, out) ⇒ args match {
      case Array(source) ⇒ new IncrementalClassifierModeler(source, server, out).run()
      case _ ⇒ new IncrementalClassifierModeler("D:\\testImages\\256_ObjectCategories", server, out).run()
    })

  }

}
import interactive.classify.IncrementalClassifierModeler._


class IncrementalClassifierModeler(source: String, server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, out) {

  def run(awaitExit:Boolean=true): Unit = {
    defineHeader()
    declareTestHandler()
    out.out("<hr/>")
    val min = 45
    if(findFile(modelName).isEmpty || System.getProperties.containsKey("rebuild")) {
      step_Generate()

      step_AddLayer(trainingMin = min, inputBands = 3, featureBands = 10, radius = 5)
      step_Train(trainingMin = min)
      step_GAN()
    }

    step_AddLayer(trainingMin = min, inputBands = 10, featureBands = 10, mode = PoolingMode.Avg)
    step_Train(trainingMin = min)
    step_GAN()

    step_AddLayer(trainingMin = min, inputBands = 10, featureBands = 10, radius = 5)
    step_Train(trainingMin = min)
    step_GAN()

    step_AddLayer(trainingMin = min, inputBands = 8, featureBands = 10, radius = 7, mode = PoolingMode.Avg)
    step_Train(trainingMin = min)
    step_GAN()

    step_Train(trainingMin = min)
    step_GAN()
    out.out("<hr/>")
    if(awaitExit) waitForExit()
  }

  def resize(source: BufferedImage, size: Int) = {
    val image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB)
    val graphics = image.getGraphics.asInstanceOf[Graphics2D]
    graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
    graphics.drawImage(source, 0, 0, size, size, null)
    image
  }

  def resize(source: BufferedImage, width: Int, height: Int) = {
    val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
    val graphics = image.getGraphics.asInstanceOf[Graphics2D]
    graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
    graphics.drawImage(source, 0, 0, width, height, null)
    image
  }

  def step_Generate() = phase({
    new PipelineNetwork() // Use an empty pipeline to begin
  }, (model: NNLayer) ⇒ {
    // Do Nothing
  }: Unit, modelName)


  def step_AddLayer(trainingMin: Int = 15, sampleSize: Int = 100, inputBands: Int, featureBands: Int, radius: Int = 3, mode: PoolingMode = PoolingMode.Max): Any = phase(modelName, (model: NNLayer) ⇒ {
    addLayer(trainingMin, sampleSize, model){
      new PipelineNetwork(
        new ConvolutionLayer(radius, radius, inputBands, featureBands, false).setWeights(() => (Random.nextDouble() - 0.5) * Math.pow(10, -6)),
        new PoolingLayer().setMode(mode)
      )
    }
  }: Unit, modelName)


  private def addLayer(trainingMin: Int, sampleSize: Int, model: NNLayer)(additionalLayer: NNLayer) = {
    val weight = -6
    val stdDevTarget: Int = 1
    val rmsSmoothing: Int = 1
    val stdDevSmoothing: Double = 0.2
    val sourceNetwork = model.asInstanceOf[PipelineNetwork]
    val priorFeaturesNode = Option(sourceNetwork.getByLabel("features")).getOrElse(sourceNetwork.getHead)
    assert(null != data)
    val rawTrainingData: Array[Array[Tensor]] = takeData().map(_.get()).toArray
    val justInputs: Array[Array[Tensor]] = rawTrainingData.map(_.take(1))
    val featureTrainingData = priorFeaturesNode.get(new NNExecutionContext() {}, sourceNetwork.buildExeCtx(
      NNResult.batchResultArray(justInputs): _*)).getData
      .stream().collect(Collectors.toList()).asScala.toArray
    val trainingArray = (0 until featureTrainingData.length).map(i => Array(featureTrainingData(i), rawTrainingData(i)(1))).toArray
    val inputFeatureDimensions = featureTrainingData.head.getDimensions()
    val outputFeatureDimensions = additionalLayer.eval(new NNExecutionContext() {}, featureTrainingData.head).getData.get(0).getDimensions
    val inputBands: Int = inputFeatureDimensions(2)
    val featureBands: Int = outputFeatureDimensions(2)
    val reconstructionCrop = outputFeatureDimensions(0)
    val categorizationLayer = new ConvolutionLayer(1, 1, featureBands, numberOfCategories, false).setWeights(() => (Random.nextDouble() - 0.5) * Math.pow(10, weight))
    val reconstructionLayer = new ConvolutionLayer(1, 1, featureBands, 4 * inputBands, false).setWeights(() => (Random.nextDouble() - 0.5) * Math.pow(10, weight))
    val trainingNetwork = new PipelineNetwork(2)
    val features = trainingNetwork.add("features", additionalLayer, trainingNetwork.getInput(0))
    val fitness = trainingNetwork.add(new f32.ProductInputsLayer(),
      // Features should be relevant - predict the class given a final linear/softmax transform
      trainingNetwork.add(new EntropyLossLayer(),
        trainingNetwork.add(new SoftmaxActivationLayer(),
          trainingNetwork.add(new MaxImageBandLayer(),
            trainingNetwork.add(categorizationLayer, features))
        ),
        trainingNetwork.getInput(1)
      ),
      // Features should be able to reconstruct input - Preserve information
      trainingNetwork.add(new LinearActivationLayer().setScale(1.0 / 255).setBias(rmsSmoothing).freeze(),
        trainingNetwork.add(new NthPowerActivationLayer().setPower(0.5).freeze(),
          trainingNetwork.add(new MeanSqLossLayer(),
            trainingNetwork.add(new ImgReshapeLayer(2, 2, true),
              trainingNetwork.add(reconstructionLayer, features)),
            trainingNetwork.add(new ImgCropLayer(reconstructionCrop, reconstructionCrop), trainingNetwork.getInput(0))
          )
        )
      ),
      // Features signal should target a uniform magnitude to balance the network
      trainingNetwork.add(new LinearActivationLayer().setBias(stdDevSmoothing).freeze(),
        trainingNetwork.add(new AbsActivationLayer(),
          trainingNetwork.add(new LinearActivationLayer().setBias(-stdDevTarget).freeze(),
            trainingNetwork.add(new AvgReducerLayer(),
              trainingNetwork.add(new StdDevMetaLayer(), features))
          )
        )
      )
    )

    out.h1("Training New Layer")
    val trainer1 = out.eval {
      var inner: Trainable = new SampledArrayTrainable(trainingArray, trainingNetwork, sampleSize)
      val trainer = new IterativeTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(trainingMin, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(50)
      trainer.setOrientation(new LBFGS)
      trainer.setLineSearchFactory(Java8Util.cvt((s: String) ⇒ (s match {
        case s if s.contains("LBFGS") ⇒ new StaticLearningRate().setRate(1.0)
        case _ ⇒ new QuadraticSearch
      })))
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    trainer1.run()

    sourceNetwork.add(new SoftmaxActivationLayer(),
      sourceNetwork.add(
        new AvgImageBandLayer(),
        sourceNetwork.add(categorizationLayer,
          sourceNetwork.add("features", additionalLayer, priorFeaturesNode)))
    )
  }

  def step_Train(trainingMin: Int = 15, sampleSize: Int = 250, iterationsPerSample: Int = 50) = phase(modelName, (model: NNLayer) ⇒ {
    out.h1("Integration Training")
    val trainer2 = out.eval {
      assert(null != data)
      var inner: Trainable = new SampledArrayTrainable(data.values.flatten.toList.asJava,
        new SimpleLossNetwork(model, new EntropyLossLayer()), sampleSize, 20)
      val trainer = new IterativeTrainer(inner)
      trainer.setMonitor(monitor)
      trainer.setTimeout(trainingMin, TimeUnit.MINUTES)
      trainer.setIterationsPerSample(iterationsPerSample)
      trainer.setOrientation(new LBFGS)
      trainer.setLineSearchFactory(Java8Util.cvt((s:String)⇒(s match {
        case s if s.contains("LBFGS") ⇒ new StaticLearningRate().setRate(1.0)
        case _ ⇒ new ArmijoWolfeSearch().setAlpha(1e-5)
      })))
      trainer.setTerminateThreshold(0.0)
      trainer
    }
    trainer2.run()
  }: Unit, modelName)

  def step_GAN() = phase(modelName, (model: NNLayer) ⇒ {
    val sourceClassId = 0
    val imageCount = 10
    out.h1("GAN Images Generation")

    val sourceClass = toOutNDArray(0, numberOfCategories)
    val targetClass = toOutNDArray(1, numberOfCategories)
    val adversarialData = data.values.flatten.toList.map(_.get()).filter(x=>x(1).get(sourceClassId) > 0.9).map(x=>Array(x(0), targetClass)).toArray
    val adversarialOutput = new ArrayBuffer[Array[Tensor]]()
    val rows = adversarialData.take(imageCount).map(adversarialData => {
      val biasLayer = new BiasLayer(data.values.flatten.head.get().head.getDimensions(): _*)
      val trainingNetwork = new PipelineNetwork()
      trainingNetwork.add(biasLayer)
      trainingNetwork.add(KryoUtil.kryo().copy(model).freeze())

      val trainer1 = out.eval {
        assert(null != data)
        var inner: Trainable = new ArrayTrainable(Array(adversarialData),
          new SimpleLossNetwork(trainingNetwork, new EntropyLossLayer()))
        val trainer = new IterativeTrainer(inner)
        trainer.setMonitor(monitor)
        trainer.setTimeout(1, TimeUnit.MINUTES)
        trainer.setOrientation(new GradientDescent)
        //trainer.setLineSearchFactory(Java8Util.cvt((s) ⇒ new ArmijoWolfeSearch().setMaxAlpha(1e8)))
        trainer.setLineSearchFactory(Java8Util.cvt((s) ⇒ new QuadraticSearch))
        trainer.setTerminateThreshold(0.01)
        trainer
      }
      trainer1.run()

      val evalNetwork = new PipelineNetwork()
      evalNetwork.add(biasLayer)
      val adversarialImage = evalNetwork.eval(new NNExecutionContext {}, adversarialData.head).getData.get(0)
      adversarialOutput += Array(adversarialImage, sourceClass)
      Map[String, AnyRef](
        "Original Image" → out.image(adversarialData.head.toRgbImage, ""),
        "Adversarial" → out.image(adversarialImage.toRgbImage, "")
      ).asJava
    }).toArray
    out.eval {
      TableOutput.create(rows: _*)
    }
//    out.h1("GAN Images Training")
//    val trainer2 = out.eval {
//      assert(null != data)
//      var inner: Trainable = new ArrayTrainable(adversarialOutput.toArray,
//        new SimpleLossNetwork(model, new EntropyLossLayer()))
//      val trainer = new IterativeTrainer(inner)
//      trainer.setMonitor(monitor)
//      trainer.setTimeout(10, TimeUnit.MINUTES)
//      trainer.setIterationsPerSample(1000)
//      trainer.setOrientation(new LBFGS)
//      trainer.setLineSearchFactory(Java8Util.cvt((s) ⇒ new ArmijoWolfeSearch))
//      trainer.setTerminateThreshold(0.0)
//      trainer
//    }
//    trainer2.run()

  }: Unit, modelName)


  def declareTestHandler() = {
    out.p("<a href='testCat.html'>Test Categorization</a><br/>")
    server.addSyncHandler("testCat.html", "text/html", cvt(o ⇒ {
      Option(new HtmlNotebookOutput(out.workingDir, o) with ScalaNotebookOutput).foreach(out ⇒ {
        testCategorization(out, getModelCheckpoint)
      })
    }), false)
  }

  def testCategorization(out: HtmlNotebookOutput with ScalaNotebookOutput, model : NNLayer) = {
    try {
      out.eval {
        TableOutput.create(Random.shuffle(data.values.flatten.toList).take(100).map(_.get()).map(testObj ⇒ Map[String, AnyRef](
          "Image" → out.image(testObj(0).toRgbImage(), ""),
          "Categorization" → categories.toList.sortBy(_._2).map(_._1)
            .zip(model.eval(new NNExecutionContext() {}, testObj(0)).getData.get(0).getData.map(_ * 100.0))
        ).asJava): _*)
      }
    } catch {
      case e: Throwable ⇒ e.printStackTrace()
    }
  }

  def takeData(numCategories : Int = 2, numImages : Int = 5000) = {
    Random.shuffle(Random.shuffle(data.toList).take(numCategories).map(_._2).flatten.toList).take(numImages)
  }

  lazy val categories: Map[String, Int] = categoryList.zipWithIndex.toMap
  lazy val (categoryList: immutable.Seq[String], data: Map[String, Stream[WeakCachedSupplier[Array[Tensor]]]]) = {
    monitor.log("Valid Processing Sizes: " + TestClassifier.validSizes.take(100).toList)
    out.p("Preparing training dataset")
    out.p("Loading data from " + source)
    val (categoryList: Seq[String], data) = load()
    val categories: Map[String, Int] = categoryList.zipWithIndex.toMap
    out.p("<ol>" + categories.toList.sortBy(_._2).map(x ⇒ "<li>" + x + "</li>").mkString("\n") + "</ol>")
    out.eval {
      TableOutput.create(data.values.flatten.toList.take(100).map(_.get()).map(e ⇒ {
        Map[String, AnyRef](
          "Image" → out.image(e(0).toRgbImage(), e(1).toString),
          "Classification" → e(1)
        ).asJava
      }): _*)
    }
    out.p("Loading data complete")
    (categoryList, data)
  }

  def toOutNDArray(out: Int, max: Int): Tensor = {
    val ndArray = new Tensor(max)
    for (i <- 0 until max) ndArray.set(i, fuzz)
    ndArray.set(out, 1-(fuzz*(max-1)))
    ndArray
  }

  def load(maxDim: Int = tileSize,
           numberOfCategories: Int = numberOfCategories,
           imagesPerCategory: Int = imagesPerCategory
          ): (Stream[String], Map[String,Stream[WeakCachedSupplier[Array[Tensor]]]]) = {
    val categoryDirs = Random.shuffle(new File(source).listFiles().toStream)
      .filter(dir => categoryWhitelist.isEmpty||categoryWhitelist.find(str => dir.getName.contains(str)).isDefined)
      .take(numberOfCategories)
    val categoryList = categoryDirs.map((categoryDirectory: File) ⇒ {
      categoryDirectory.getName.split('.').last
    })
    val categoryMap: Map[String, Int] = categoryList.zipWithIndex.toMap
    (categoryList, Random.shuffle(categoryDirs
      .map((categoryDirectory: File) ⇒ {
        val categoryName = categoryDirectory.getName.split('.').last
        categoryName -> Random.shuffle(categoryDirectory.listFiles().toStream).take(imagesPerCategory)
          .filterNot(_ == null).filterNot(_ == null)
          .map(file ⇒ {
            new WeakCachedSupplier[Array[Tensor]](Java8Util.cvt(()=>{
              val original = ImageIO.read(file)
              val fromWidth = original.getWidth()
              val fromHeight = original.getHeight()
              val scale = maxDim.toDouble / Math.min(fromWidth, fromHeight)
              val toWidth = ((fromWidth * scale).toInt)
              val toHeight = ((fromHeight * scale).toInt)
              val resized = new BufferedImage(maxDim, maxDim, BufferedImage.TYPE_INT_ARGB)
              val graphics = resized.getGraphics.asInstanceOf[Graphics2D]
              graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
              if (toWidth < toHeight) {
                graphics.drawImage(original, 0, (toWidth - toHeight) / 2, toWidth, toHeight, null)
              } else {
                graphics.drawImage(original, (toHeight - toWidth) / 2, 0, toWidth, toHeight, null)
              }
              Array(Tensor.fromRGB(resized), toOutNDArray(categoryMap(categoryName), categoryMap.size))
            }))
          })
      })).toMap)
  }
}

