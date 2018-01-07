/*
 * Copyright (c) 2018 by Andrew Charneski.
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

package report

import java.awt.image.BufferedImage
import java.awt.{BasicStroke, Color, Graphics2D, RenderingHints}
import javax.imageio.ImageIO

import util._
import java.util

import boofcv.abst.feature.associate.ScoreAssociation
import boofcv.abst.feature.describe.ConfigSurfDescribe.Stability
import boofcv.abst.feature.describe.{ConfigBrief, DescribeRegionPoint}
import boofcv.abst.feature.detdesc.DetectDescribePoint
import boofcv.abst.feature.detect.interest.{ConfigFastHessian, ConfigGeneralDetector, InterestPointDetector}
import boofcv.abst.feature.detect.line.{DetectLine, DetectLineSegment}
import boofcv.abst.feature.orientation.ConfigSlidingIntegral
import boofcv.abst.feature.tracker.{PointTrack, PointTracker}
import boofcv.alg.descriptor.UtilFeature
import boofcv.alg.distort.impl.DistortSupport
import boofcv.alg.distort.{ImageDistort, PixelTransformHomography_F32}
import boofcv.alg.feature.detect.interest.GeneralFeatureDetector
import boofcv.alg.filter.derivative.GImageDerivativeOps
import boofcv.core.image.border.BorderType
import boofcv.factory.feature.associate.FactoryAssociation
import boofcv.factory.feature.describe.FactoryDescribeRegionPoint
import boofcv.factory.feature.detdesc.FactoryDetectDescribe
import boofcv.factory.feature.detect.interest.{FactoryDetectPoint, FactoryInterestPoint}
import boofcv.factory.feature.detect.line.{ConfigHoughFoot, ConfigHoughFootSubimage, ConfigHoughPolar, FactoryDetectLineAlgs}
import boofcv.factory.geo.{ConfigRansac, FactoryMultiViewRobust}
import boofcv.factory.interpolate.FactoryInterpolation
import boofcv.io.image.ConvertBufferedImage
import boofcv.struct.feature.{AssociatedIndex, BrightFeature, TupleDesc_B}
import boofcv.struct.geo.AssociatedPair
import boofcv.struct.image._
import georegression.struct.homography.Homography2D_F64
import georegression.struct.line.{LineParametric2D_F32, LineSegment2D_F32}
import georegression.struct.point.Point2D_F64
import georegression.transform.homography.HomographyPointOps_F64
import org.ddogleg.fitting.modelset.ModelMatcher
import org.ddogleg.struct.FastQueue
import org.scalatest.{MustMatchers, WordSpec}

import scala.collection.JavaConverters._

class BoofcvTest extends WordSpec with MustMatchers with ReportNotebook {


  "BoofCV" should {

    "Find Rulers" in {
      report("vectors", log ⇒ {
        val image1 = ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard1.jpg"))
        val width = 1200
        val height = image1.getHeight * width / image1.getWidth()

        def fn(detector: DetectLine[GrayU8]) = {
          log.draw(gfx ⇒ {
            val found: util.List[LineParametric2D_F32] = detector.detect(ConvertBufferedImage.convertFromSingle(image1, null, classOf[GrayU8]))
            gfx.drawImage(image1, 0, 0, width, height, null)
            found.asScala.foreach(line ⇒ {
              System.out.println(line)
              if (Math.abs(line.slope.x) > Math.abs(line.slope.y)) {
                val x1 = 0
                val y1 = (line.p.y - line.p.x * line.slope.y / line.slope.x).toInt
                val x2 = image1.getWidth
                val y2 = y1 + (x2 * line.slope.y / line.slope.x).toInt
                gfx.setColor(Color.RED)
                gfx.drawLine(
                  x1 * width / image1.getWidth, y1 * height / image1.getHeight,
                  x2 * width / image1.getWidth, y2 * height / image1.getHeight)
              } else {
                val y1 = 0
                val x1 = (line.p.x - line.p.y * line.slope.x / line.slope.y).toInt
                val y2 = image1.getHeight
                val x2 = x1 + (y2 * line.slope.x / line.slope.y).toInt
                gfx.setColor(Color.GREEN)
                gfx.drawLine(
                  x1 * width / image1.getWidth, y1 * height / image1.getHeight,
                  x2 * width / image1.getWidth, y2 * height / image1.getHeight)
              }
            })
          }, width = width, height = height)
        }

        val edgeThreshold: Float = 100
        val maxLines: Int = 100
        val localMaxRadius = 4
        val minCounts = 5
        val minDistanceFromOrigin = 1
        val resolutionAngle = Math.PI / (3 * 180)
        val totalHorizontalDivisions = 8
        val totalVerticalDivisions = 8
        log.h2("HoughPolar")
        fn(log.code(() ⇒ {
          FactoryDetectLineAlgs.houghPolar(new ConfigHoughPolar(localMaxRadius, minCounts, 2, resolutionAngle, edgeThreshold, maxLines), classOf[GrayU8], classOf[GrayS16])
        }))
        log.h2("HoughFoot")
        fn(log.code(() ⇒ {
          FactoryDetectLineAlgs.houghFoot(new ConfigHoughFoot(localMaxRadius, minCounts, minDistanceFromOrigin, edgeThreshold, maxLines), classOf[GrayU8], classOf[GrayS16])
        }))
        log.h2("HoughFootSubimage")
        fn(log.code(() ⇒ {
          FactoryDetectLineAlgs.houghFootSub(new ConfigHoughFootSubimage(localMaxRadius, minCounts, minDistanceFromOrigin, edgeThreshold, maxLines, totalHorizontalDivisions, totalVerticalDivisions), classOf[GrayU8], classOf[GrayS16])
        }))
      })
    }

    "Find Vectors" in {
      report("segments", log ⇒ {
        val image1 = ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard1.jpg"))
        val width = 1200
        val height = image1.getHeight * width / image1.getWidth()

        def fn(detector: DetectLineSegment[GrayF32]) = {
          log.draw(gfx ⇒ {
            val found: util.List[LineSegment2D_F32] = detector.detect(ConvertBufferedImage.convertFromSingle(image1, null, classOf[GrayF32]))
            gfx.drawImage(image1, 0, 0, width, height, null)
            gfx.setColor(Color.GREEN)
            found.asScala.foreach(line ⇒ {
              gfx.drawLine(
                (line.a.x * width / image1.getWidth).toInt, (line.a.y * height / image1.getHeight).toInt,
                (line.b.x * width / image1.getWidth).toInt, (line.b.y * height / image1.getHeight).toInt)
            })
          }, width = width, height = height)
        }

        fn(log.code(() ⇒ {
          FactoryDetectLineAlgs.lineRansac(40, 30, 2.36, true, classOf[GrayF32], classOf[GrayF32])
        }))
      })
    }

    "Feature Decection" in {
      report("features", log ⇒ {

        def draw(primaryImage: BufferedImage, points: util.ArrayList[Point2D_F64], descriptions: FastQueue[BrightFeature]) = {
          log.draw(gfx ⇒ {
            gfx.drawImage(primaryImage, 0, 0, null)
            gfx.setStroke(new BasicStroke(2))
            points.asScala.zip(descriptions.toList.asScala).foreach(x ⇒ {
              val (pt, d) = x
              if (d.white) {
                gfx.setColor(Color.GREEN)
              } else {
                gfx.setColor(Color.RED)
              }
              gfx.drawRect(pt.x.toInt - 4, pt.y.toInt - 4, 9, 9)
            })
          }, width = primaryImage.getWidth, height = primaryImage.getHeight())
        }

        def draw2(primaryImage: BufferedImage, points: util.ArrayList[Point2D_F64], descriptions: FastQueue[TupleDesc_B]) = {
          log.draw(gfx ⇒ {
            gfx.drawImage(primaryImage, 0, 0, null)
            gfx.setStroke(new BasicStroke(2))
            points.asScala.zip(descriptions.toList.asScala).foreach(x ⇒ {
              val (pt, d) = x
              gfx.drawRect(pt.x.toInt - 4, pt.y.toInt - 4, 9, 9)
            })
          }, width = primaryImage.getWidth, height = primaryImage.getHeight())
        }

        def test(image1: BufferedImage, featureDetector: DetectDescribePoint[GrayF32, BrightFeature]) = {
          val (pointsA, descriptionsA) = describe(featureDetector, image1)
          draw(image1, pointsA, descriptionsA)
        }

        def test2(image1: BufferedImage, featureDetector: DetectDescribePoint[GrayF32, TupleDesc_B]) = {
          val (pointsA, descriptionsA) = describe2(featureDetector, image1)
          draw2(image1, pointsA, descriptionsA)
        }

        test(ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard1.jpg")), {
          val detectThreshold = 5
          val extractRadius = 1
          val maxFeaturesPerScale = 500
          val initialSampleSize = 3
          val initialSize = 15
          val numberScalesPerOctave = 4
          val numberOfOctaves = 8
          val fastHessian = new ConfigFastHessian(detectThreshold, extractRadius, maxFeaturesPerScale, initialSampleSize, initialSize, numberScalesPerOctave, numberOfOctaves)
          val samplePeriod = 0.65
          val windowSize = 1.0471975511965976
          val radius = 1
          val weightSigma = -1.0
          val sampleWidth = 6
          val stabilityConfig = new Stability
          FactoryDetectDescribe.surfStable(fastHessian, stabilityConfig, new ConfigSlidingIntegral(samplePeriod, windowSize, radius, weightSigma, sampleWidth), classOf[GrayF32])
        })

        test2(ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard1.jpg")), {
          val imageType: Class[GrayF32] = classOf[GrayF32]
          val derivType: Class[GrayF[_]] = GImageDerivativeOps.getDerivativeType(imageType)
          val corner: GeneralFeatureDetector[GrayF32, GrayF[_]] = FactoryDetectPoint.createShiTomasi(new ConfigGeneralDetector(1000, 5, 1), false, derivType)
          val detector: InterestPointDetector[GrayF32] = FactoryInterestPoint.wrapPoint(corner, 1, imageType, derivType)
          val describe: DescribeRegionPoint[GrayF32, TupleDesc_B] = FactoryDescribeRegionPoint.brief(new ConfigBrief(true), imageType)
          FactoryDetectDescribe.fuseTogether(detector, null, describe)
        })

      })
    }

    "Composite Images" in {
      report("composite", log ⇒ {
        import boofcv.alg.tracker.klt.PkltConfig
        val config = new PkltConfig
        config.templateRadius = 3
        config.pyramidScaling = Array[Int](1, 2, 4, 8)
        val featureDetector: DetectDescribePoint[GrayF32, BrightFeature] = FactoryDetectDescribe.surfStable(new ConfigFastHessian(1, 2, 300, 1, 9, 4, 4), null, null, classOf[GrayF32])
        val image1 = ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard1.jpg"))
        val width = 1200
        val height = image1.getHeight * width / image1.getWidth()

        log.draw(gfx ⇒ {
          gfx.drawImage(image1, 0, 0, width, height, null)
        })
        val (pointsA, descriptionsA) = log.code(() ⇒ {
          describe(featureDetector, image1)
        })
        log.draw(gfx ⇒ {
          val hints = new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC)
          hints.put(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
          gfx.asInstanceOf[Graphics2D].setRenderingHints(hints)

          gfx.drawImage(image1, 0, 0, width, height, null)

          gfx.setColor(Color.YELLOW)
          pointsA.asScala.zip(descriptionsA.toList.asScala).foreach(entry ⇒ {
            val (pt: Point2D_F64, desc: BrightFeature) = entry
            val x = ((pt.x / image1.getWidth) * width).toInt
            val y = ((pt.y / image1.getHeight) * height).toInt
            gfx.fillRect((x - 1), (y - 1), 3, 3)
          })
        })

        def rectify(primaryImage: BufferedImage, secondaryImages: BufferedImage*)(expand: Boolean = true): List[BufferedImage] = {
          val transforms: Map[BufferedImage, Homography2D_F64] = secondaryImages.map(secondaryImage ⇒ {
            val (pointsB, descriptionsB) = log.code(() ⇒ {
              describe(featureDetector, secondaryImage)
            })

            val pairs: util.ArrayList[AssociatedPair] = log.code(() ⇒ {
              associate(featureDetector, pointsA, descriptionsA, pointsB, descriptionsB)
            })

            log.draw(gfx ⇒ {
              val hints = new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC)
              hints.put(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
              gfx.asInstanceOf[Graphics2D].setRenderingHints(hints)

              gfx.drawImage(primaryImage, 0, 0, width, height, null)
              gfx.drawImage(secondaryImage, width, 0, width, height, null)

              gfx.setColor(Color.YELLOW)
              pairs.asScala.foreach(pair ⇒ {
                val x1 = ((pair.p1.x / primaryImage.getWidth) * width).toInt
                val y1 = ((pair.p1.y / primaryImage.getHeight) * height).toInt
                val x2 = width + ((pair.p2.x / primaryImage.getWidth) * width).toInt
                val y2 = ((pair.p2.y / primaryImage.getHeight) * height).toInt
                gfx.drawLine(x1, y1, x2, y2)
              })
            }, width = 2 * width)

            secondaryImage → log.code(() ⇒ {
              val modelMatcher: ModelMatcher[Homography2D_F64, AssociatedPair] = FactoryMultiViewRobust.homographyRansac(null, new ConfigRansac(60, 3));
              if (!modelMatcher.process(pairs)) throw new RuntimeException("Model Matcher failed!")
              modelMatcher.getModelParameters
            })
          }).toMap

          val boundsPoints: List[(Double, Double)] = log.code(() ⇒ {
            transforms.flatMap(x ⇒ {
              val (secondaryImage, transformParameters) = x
              val fromBtoA = transformParameters.invert(null)
              List((0, 0), (0, secondaryImage.getHeight), (secondaryImage.getWidth, 0), (secondaryImage.getWidth, secondaryImage.getHeight)).map(xx ⇒ {
                val (x, y) = xx
                tranform(x, y, fromBtoA)
              })
            }).toList
          }) ++ List[(Double, Double)]((0, 0), (0, primaryImage.getHeight), (primaryImage.getWidth, 0), (primaryImage.getWidth, primaryImage.getHeight))

          val (offsetX: Double, offsetY: Double) = log.code(() ⇒ {
            val renderMinX = boundsPoints.map(_._1).min
            val renderMaxX = boundsPoints.map(_._1).max
            val renderMinY = boundsPoints.map(_._2).min
            val renderMaxY = boundsPoints.map(_._2).max
            if (expand) (Math.max(0, -renderMinX), Math.max(0, -renderMinY))
            else (0.0, 0.0)
          })

          val (renderWidth: Double, renderHeight: Double) = log.code(() ⇒ {
            val renderMinX = boundsPoints.map(_._1).min
            val renderMaxX = boundsPoints.map(_._1).max
            val renderMinY = boundsPoints.map(_._2).min
            val renderMaxY = boundsPoints.map(_._2).max
            if (expand) (renderMaxX - renderMinX, renderMaxY - renderMinY)
            else (primaryImage.getWidth.toDouble, primaryImage.getHeight.toDouble)
          })

          List(log.code(() ⇒ {
            val output = new BufferedImage(renderWidth.toInt, renderHeight.toInt, primaryImage.getType)
            output.getGraphics.drawImage(primaryImage, offsetX.toInt, offsetY.toInt, null)
            output
          })) ++ transforms.map(x ⇒ {
            val (secondaryImage, transformParameters) = x

            val distortion: ImageDistort[Planar[GrayF32], Planar[GrayF32]] = log.code(() ⇒ {
              val scale = 1
              val fromAToWork = new Homography2D_F64(scale, 0, offsetX, 0, scale, offsetY, 0, 0, 1)
              val fromWorkToA = fromAToWork.invert(null)
              val fromWorkToB: Homography2D_F64 = fromWorkToA.concat(transformParameters, null)

              val interpolation = FactoryInterpolation.bilinearPixelS(classOf[GrayF32], BorderType.ZERO)
              val model = new PixelTransformHomography_F32
              val distort = DistortSupport.createDistortPL(classOf[GrayF32], model, interpolation, false)
              model.set(fromWorkToB)
              distort.setRenderAll(false)
              distort
            })

            log.code(() ⇒ {
              val boofImage = ConvertBufferedImage.convertFromMulti(secondaryImage, null, true, classOf[GrayF32])
              val work: Planar[GrayF32] = boofImage.createNew(renderWidth.toInt, renderHeight.toInt)
              distortion.apply(boofImage, work)
              val output = new BufferedImage(renderWidth.toInt, renderHeight.toInt, primaryImage.getType)
              ConvertBufferedImage.convertTo(work, output, true)
              output
            })
          }).toList

        }

        val images: List[BufferedImage] = rectify(image1
          , ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard2.jpg"))
          , ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard3.jpg"))
        )(false)
      })
    }

  }

  def tranform(x0: Int, y0: Int, fromBtoWork: Homography2D_F64): (Double, Double) = {
    val result = new Point2D_F64
    HomographyPointOps_F64.transform(fromBtoWork, new Point2D_F64(x0, y0), result)
    val rx = result.x
    val ry = result.y
    (rx, ry)
  }

  def describe(detDesc: PointTracker[GrayF32], image: BufferedImage): util.List[PointTrack] = {
    val input = ConvertBufferedImage.convertFromSingle(image, null, classOf[GrayF32])
    detDesc.process(input)
    detDesc.getActiveTracks(null)
  }

  def describe(detDesc: DetectDescribePoint[GrayF32, BrightFeature], image: BufferedImage): (util.ArrayList[Point2D_F64], FastQueue[BrightFeature]) = {
    val points: util.ArrayList[Point2D_F64] = new util.ArrayList[Point2D_F64]()
    val input = ConvertBufferedImage.convertFromSingle(image, null, classOf[GrayF32])
    val descriptions: FastQueue[BrightFeature] = UtilFeature.createQueue[BrightFeature](detDesc, 100)
    detDesc.detect(input)
    var i = 0
    while (i < detDesc.getNumberOfFeatures) {
      points.add(detDesc.getLocation(i).copy)
      val grow: BrightFeature = descriptions.grow
      val v = detDesc.getDescription(i)
      grow.setTo(v)
      i += 1;
    }
    (points, descriptions)
  }

  def describe2(detDesc: DetectDescribePoint[GrayF32, TupleDesc_B], image: BufferedImage): (util.ArrayList[Point2D_F64], FastQueue[TupleDesc_B]) = {
    val points: util.ArrayList[Point2D_F64] = new util.ArrayList[Point2D_F64]()
    val input = ConvertBufferedImage.convertFromSingle(image, null, classOf[GrayF32])
    val descriptions: FastQueue[TupleDesc_B] = UtilFeature.createQueue[TupleDesc_B](detDesc, 100)
    detDesc.detect(input)
    var i = 0
    while (i < detDesc.getNumberOfFeatures) {
      points.add(detDesc.getLocation(i).copy)
      val grow: TupleDesc_B = descriptions.grow
      val v = detDesc.getDescription(i)
      grow.setTo(v)
      i += 1;
    }
    (points, descriptions)
  }

  def associate(detDesc: DetectDescribePoint[GrayF32, BrightFeature], pointsA: util.ArrayList[Point2D_F64], descriptionsA: FastQueue[BrightFeature],
                pointsB: util.ArrayList[Point2D_F64], descriptionsB: FastQueue[BrightFeature]): util.ArrayList[AssociatedPair] = {
    val descriptionType: Class[BrightFeature] = detDesc.getDescriptionType
    val scorer: ScoreAssociation[BrightFeature] = FactoryAssociation.defaultScore(descriptionType)
    val associate = FactoryAssociation.greedy(scorer, java.lang.Double.MAX_VALUE, true)
    associate.setSource(descriptionsA)
    associate.setDestination(descriptionsB)
    associate.associate
    val matches: FastQueue[AssociatedIndex] = associate.getMatches
    val pairs = new util.ArrayList[AssociatedPair]()
    var i = 0
    while (i < matches.size) {
      val m = matches.get(i)
      val a = pointsA.get(m.src)
      val b = pointsB.get(m.dst)
      pairs.add(new AssociatedPair(a, b, false))
      i += 1
    }
    pairs
  }

}