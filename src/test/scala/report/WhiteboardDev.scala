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

package report

import java.awt.image.BufferedImage
import java.awt.{BasicStroke, Color, Graphics, Graphics2D, RenderingHints}

import util.{ReportNotebook, ScalaNotebookOutput}
import java.util
import javax.imageio.ImageIO

import boofcv.abst.feature.describe.ConfigSurfDescribe.Stability
import boofcv.abst.feature.detdesc.DetectDescribePoint
import boofcv.abst.feature.detect.interest.ConfigFastHessian
import boofcv.abst.feature.detect.line.{DetectLine, DetectLineSegment}
import boofcv.abst.feature.orientation.ConfigSlidingIntegral
import boofcv.alg.color.{ColorHsv, ColorLab, ColorXyz, ColorYuv}
import boofcv.alg.distort.impl.DistortSupport
import boofcv.alg.distort.{ImageDistort, PixelTransformHomography_F32}
import boofcv.alg.filter.binary.{BinaryImageOps, GThresholdImageOps}
import boofcv.alg.misc.ImageStatistics
import boofcv.alg.segmentation.ImageSegmentationOps
import boofcv.core.image.border.BorderType
import boofcv.factory.feature.associate.FactoryAssociation
import boofcv.factory.feature.detdesc.FactoryDetectDescribe
import boofcv.factory.feature.detect.line.{ConfigHoughFoot, FactoryDetectLineAlgs}
import boofcv.factory.geo.{ConfigHomography, ConfigRansac, FactoryMultiViewRobust}
import boofcv.factory.interpolate.FactoryInterpolation
import boofcv.factory.segmentation.{ConfigFh04, FactoryImageSegmentation}
import boofcv.gui.binary.VisualizeBinaryData
import boofcv.gui.feature.VisualizeRegions
import boofcv.gui.image.VisualizeImageData
import boofcv.io.image.ConvertBufferedImage
import boofcv.struct.feature._
import boofcv.struct.geo.AssociatedPair
import boofcv.struct.image.{GrayF32, GrayS32, ImageType, Planar, _}
import georegression.geometry.UtilPolygons2D_F32
import georegression.metric.Intersection2D_F32
import georegression.struct.homography.Homography2D_F64
import georegression.struct.line.{LineParametric2D_F32, LineSegment2D_F32}
import georegression.struct.point.{Point2D_F32, Point2D_F64}
import georegression.struct.shapes.{Quadrilateral_F32, Rectangle2D_F32}
import georegression.transform.homography.HomographyPointOps_F64
import org.ddogleg.fitting.modelset.ModelMatcher
import org.ddogleg.struct.{FastQueue, GrowQueue_I32}
import org.jtransforms.fft.FloatFFT_2D
import org.scalatest.{MustMatchers, WordSpec}

import scala.collection.JavaConverters._
import scala.collection.immutable

class WhiteboardDev extends WordSpec with MustMatchers with ReportNotebook {

  "Whiteboard Image Processing" should {

    "Identify whiteboard region" in {
      report("quadrangle", log ⇒ {

        val image1 = ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard1.jpg"))
        val rulerDetector: DetectLine[GrayU8] = log.code(() ⇒ {
          val localMaxRadius = 10
          val minCounts = 5
          val minDistanceFromOrigin = 1
          val edgeThreshold: Float = 100
          val maxLines: Int = 20
          FactoryDetectLineAlgs.houghFoot(new ConfigHoughFoot(localMaxRadius, minCounts, minDistanceFromOrigin, edgeThreshold, maxLines), classOf[GrayU8], classOf[GrayS16])
        })
        val featureDetector1: DetectDescribePoint[GrayF32, BrightFeature] = {
          val detectThreshold = 10
          val extractRadius = 1
          val maxFeaturesPerScale = 500
          val initialSampleSize = 1
          val initialSize = 1
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
        }
        val transformModel: ModelMatcher[Homography2D_F64, AssociatedPair] = {
          val maxIterations = 100
          val inlierThreshold = 7
          val normalize = true
          FactoryMultiViewRobust.homographyRansac(new ConfigHomography(normalize), new ConfigRansac(maxIterations, inlierThreshold))
        }

        val found: util.List[LineParametric2D_F32] = rulerDetector.detect(ConvertBufferedImage.convertFromSingle(image1, null, classOf[GrayU8]))
        val horizontals = found.asScala.filter(line ⇒ Math.abs(line.slope.x) > Math.abs(line.slope.y)).toList
        val verticals = found.asScala.filter(line ⇒ Math.abs(line.slope.x) <= Math.abs(line.slope.y)).toList
        val candidateQuadrangles: List[Quadrilateral_F32] = log.code(() ⇒ {
          val imageBounds = new Rectangle2D_F32(0, 0, image1.getWidth, image1.getHeight)
          cross(pairs(horizontals), pairs(verticals)).map(xa ⇒ {
            val ((left: LineParametric2D_F32, right: LineParametric2D_F32), (top: LineParametric2D_F32, bottom: LineParametric2D_F32)) = xa
            new Quadrilateral_F32(
              Intersection2D_F32.intersection(left, top, null),
              Intersection2D_F32.intersection(left, bottom, null),
              Intersection2D_F32.intersection(right, top, null),
              Intersection2D_F32.intersection(right, bottom, null))
          }).filter((quad: Quadrilateral_F32) ⇒
            Intersection2D_F32.contains(imageBounds, quad.a.x, quad.a.y) &&
              Intersection2D_F32.contains(imageBounds, quad.b.x, quad.b.y) &&
              Intersection2D_F32.contains(imageBounds, quad.c.x, quad.c.y) &&
              Intersection2D_F32.contains(imageBounds, quad.d.x, quad.d.y)
          )
        })

        var bestQuadrangle: Quadrilateral_F32 = scale(rotate(log.code(() ⇒ {
          candidateQuadrangles.maxBy(quad ⇒ {
            val bounds = new Rectangle2D_F32()
            UtilPolygons2D_F32.bounding(quad, bounds)
            val area = quad.area()
            val squareness = area / bounds.area()
            assert(squareness >= 0 && squareness <= 1.01)
            area * Math.pow(squareness, 2)
          })
        })), 1.0f)

        log.draw(gfx ⇒ {
          gfx.drawImage(image1, 0, 0, null)
          gfx.setStroke(new BasicStroke(3))
          gfx.setColor(Color.YELLOW)
          draw(gfx, bestQuadrangle)
          gfx.setColor(Color.RED)
          draw(gfx, scale(bestQuadrangle, 0.9f))
        }, width = image1.getWidth, height = image1.getHeight)

        optimizeQuadrangle(log, image1, bestQuadrangle)

        val (areaHeight, areaWidth) = log.code(() ⇒ {
          (
            (bestQuadrangle.getSideLength(0) + bestQuadrangle.getSideLength(2)).toInt / 2,
            (bestQuadrangle.getSideLength(1) + bestQuadrangle.getSideLength(3)).toInt / 2
          )
        })
        val transform: Homography2D_F64 = log.code(() ⇒ {
          val pairs: util.ArrayList[AssociatedPair] = new util.ArrayList(List(
            new AssociatedPair(0, 0, bestQuadrangle.a.x, bestQuadrangle.a.y),
            new AssociatedPair(0, areaHeight, bestQuadrangle.c.x, bestQuadrangle.c.y),
            new AssociatedPair(areaWidth, 0, bestQuadrangle.b.x, bestQuadrangle.b.y),
            new AssociatedPair(areaWidth, areaHeight, bestQuadrangle.d.x, bestQuadrangle.d.y)
          ).asJava)
          if (!transformModel.process(pairs)) throw new RuntimeException("Model Matcher failed!")
          transformModel.getModelParameters
        })

        val primaryImage: BufferedImage = log.code(() ⇒ {
          val distortion: ImageDistort[Planar[GrayF32], Planar[GrayF32]] = {
            val interpolation = FactoryInterpolation.bilinearPixelS(classOf[GrayF32], BorderType.ZERO)
            val model = new PixelTransformHomography_F32
            val distort = DistortSupport.createDistortPL(classOf[GrayF32], model, interpolation, false)
            model.set(transform)
            distort.setRenderAll(false)
            distort
          }
          val boofImage = ConvertBufferedImage.convertFromMulti(image1, null, true, classOf[GrayF32])
          val work: Planar[GrayF32] = boofImage.createNew(areaWidth.toInt, areaHeight.toInt)
          distortion.apply(boofImage, work)
          val output = new BufferedImage(areaWidth.toInt, areaHeight.toInt, image1.getType)
          ConvertBufferedImage.convertTo(work, output, true)
          output
        })

        val images: List[BufferedImage] = rectify(log, featureDetector1, false)(primaryImage,
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard2.jpg")),
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard3.jpg"))
        )

        val tileBounds = new Rectangle2D_F32(750, 750, 2000, 1500)
        images.foreach(img ⇒ log.draw(gfx ⇒ {
          gfx.drawImage(primaryImage, 0, 0, null)
          gfx.setStroke(new BasicStroke(3))
          gfx.setColor(Color.YELLOW)
          gfx.drawRect(tileBounds.p0.x.toInt, tileBounds.p0.x.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt)
          gfx.setColor(Color.RED)
          gfx.drawRect(tileBounds.p0.x.toInt, tileBounds.p0.x.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt)
        }, height = primaryImage.getHeight, width = primaryImage.getWidth))
        images.foreach(img ⇒ log.code(() ⇒ {
          img.getSubimage(tileBounds.p0.x.toInt, tileBounds.p0.x.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt)
        }))
      })
    }

    "Process Tiles" in {
      report("tilesRgb", log ⇒ {

        val images: Seq[BufferedImage] = List(
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("whiteboard_tile_1.png")),
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("whiteboard_tile_2.png")),
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("whiteboard_tile_3.png"))
        )

        sketchExperiment2(log)(images.head, images.tail: _*)

        colorSpaces(log)(images.head, images.tail: _*)

        removeGradualLighting(log)(images.head, images.tail: _*)

        apply(log, img ⇒ {
          GThresholdImageOps.threshold(img, null, ImageStatistics.mean(img), true)
        })(images.head, images.tail: _*)

        apply(log, img ⇒ {
          GThresholdImageOps.threshold(img, null, GThresholdImageOps.computeOtsu(img, 0, 255), true)
        })(images.head, images.tail: _*)

        apply(log, img ⇒ {
          GThresholdImageOps.threshold(img, null, GThresholdImageOps.computeEntropy(img, 0, 255), true)
        })(images.head, images.tail: _*)

        apply(log, img ⇒ {
          val binary = new GrayU8(img.width, img.height)
          GThresholdImageOps.localGaussian(img, binary, 42, 1.0, true, null, null)
        })(images.head, images.tail: _*)

        sketch(log, {
          val regionSize = 50
          val thresholdEdge = 10
          val thresholdAngle = 1.0
          val connectLines = true
          FactoryDetectLineAlgs.lineRansac(regionSize, thresholdEdge, thresholdAngle, connectLines, classOf[GrayF32], classOf[GrayF32])
        })(images.head, images.tail: _*)

      })
    }

    "Colorize Tiles" in {
      report("tiles_color", log ⇒ {

        val images: Seq[BufferedImage] = List(
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("whiteboard_tile_1.png")),
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("whiteboard_tile_2.png")),
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("whiteboard_tile_3.png"))
        )

        val normalizedLighting = removeGradualLighting(log)(images.head, images.tail: _*)

        applyMulti(log, rgb ⇒ {
          val hsv = rgb.createSameShape()
          ColorHsv.rgbToHsv_F32(rgb, hsv)
          GThresholdImageOps.threshold(hsv.getBand(2), null, ImageStatistics.mean(hsv.getBand(2)), true)
        })(images.head, images.tail: _*)

        applyMulti(log, rgb ⇒ {
          val hsv = rgb.createSameShape()
          ColorHsv.rgbToHsv_F32(rgb, hsv)
          GThresholdImageOps.threshold(hsv.getBand(2), null, GThresholdImageOps.computeOtsu(hsv.getBand(2), 0, 255), true)
        })(images.head, images.tail: _*)

        apply(log, img ⇒ {
          GThresholdImageOps.threshold(img, null, ImageStatistics.mean(img), true)
        })(images.head, images.tail: _*)

        apply(log, (img: GrayF32) ⇒ {
          GThresholdImageOps.threshold(img, null, GThresholdImageOps.computeOtsu(img, 0, 255), true)
        })(images.head, images.tail: _*)

        apply(log, img ⇒ {
          GThresholdImageOps.threshold(img, null, GThresholdImageOps.computeEntropy(img, 0, 255), true)
        })(images.head, images.tail: _*)

        apply(log, img ⇒ {
          GThresholdImageOps.threshold(img, null, GThresholdImageOps.computeEntropy(img, 0, 255), true)
        })(images.head, images.tail: _*)

        apply(log, img ⇒ {
          val binary = new GrayU8(img.width, img.height)
          GThresholdImageOps.localGaussian(img, binary, 42, 1.0, true, null, null)
        })(normalizedLighting.head, normalizedLighting.tail: _*)

        colorSpaces(log)(images.head, images.tail: _*)

        sketch(log, {
          val regionSize = 50
          val thresholdEdge = 10
          val thresholdAngle = 1.0
          val connectLines = true
          FactoryDetectLineAlgs.lineRansac(regionSize, thresholdEdge, thresholdAngle, connectLines, classOf[GrayF32], classOf[GrayF32])
        })(images.head, images.tail: _*)

      })
    }

    "Align Tiles" in {
      report("tiles_align", log ⇒ {

        val images: Seq[BufferedImage] = List(
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("whiteboard_tile_1.png")),
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("whiteboard_tile_2.png")),
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("whiteboard_tile_3.png"))
        )

        val featureDetector1: DetectDescribePoint[GrayF32, BrightFeature] = {
          val detectThreshold = 10
          val extractRadius = 1
          val maxFeaturesPerScale = 500
          val initialSampleSize = 1
          val initialSize = 1
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
        }

        def merge(img: immutable.Seq[BufferedImage]): List[BufferedImage] = rectify(log, featureDetector1, false)(img.head, img.tail: _*)

        val normalizedLighting: immutable.Seq[BufferedImage] = removeGradualLighting(log)(images.head, images.tail: _*)

        merge(normalizedLighting)

        applyMulti(log, rgb ⇒ {
          val hsv = rgb.createSameShape()
          ColorHsv.rgbToHsv_F32(rgb, hsv)
          GThresholdImageOps.threshold(hsv.getBand(2), null, ImageStatistics.mean(hsv.getBand(2)), true)
        })(images.head, images.tail: _*)

        applyMulti(log, rgb ⇒ {
          val hsv = rgb.createSameShape()
          ColorHsv.rgbToHsv_F32(rgb, hsv)
          GThresholdImageOps.threshold(hsv.getBand(2), null, GThresholdImageOps.computeOtsu(hsv.getBand(2), 0, 255), true)
        })(images.head, images.tail: _*)

        apply(log, img ⇒ {
          GThresholdImageOps.threshold(img, null, ImageStatistics.mean(img), true)
        })(images.head, images.tail: _*)

        apply(log, (img: GrayF32) ⇒ {
          GThresholdImageOps.threshold(img, null, GThresholdImageOps.computeOtsu(img, 0, 255), true)
        })(images.head, images.tail: _*)

        val entropyImg = apply(log, img ⇒ {
          GThresholdImageOps.threshold(img, null, GThresholdImageOps.computeEntropy(img, 0, 255), true)
        })(images.head, images.tail: _*)

        merge(entropyImg)

        apply(log, img ⇒ {
          GThresholdImageOps.threshold(img, null, GThresholdImageOps.computeEntropy(img, 0, 255), true)
        })(normalizedLighting.head, normalizedLighting.tail: _*)

        apply(log, img ⇒ {
          val binary = new GrayU8(img.width, img.height)
          GThresholdImageOps.localGaussian(img, binary, 42, 1.0, true, null, null)
        })(normalizedLighting.head, normalizedLighting.tail: _*)

        colorSpaces(log)(images.head, images.tail: _*)

        sketch(log, {
          val regionSize = 50
          val thresholdEdge = 10
          val thresholdAngle = 1.0
          val connectLines = true
          FactoryDetectLineAlgs.lineRansac(regionSize, thresholdEdge, thresholdAngle, connectLines, classOf[GrayF32], classOf[GrayF32])
        })(images.head, images.tail: _*)

      })
    }

    "Run entire workflow" in {
      report("whiteboard", log ⇒ {

        val image1 = ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard1.jpg"))
        val rulerDetector: DetectLine[GrayU8] = log.code(() ⇒ {
          val localMaxRadius = 10
          val minCounts = 5
          val minDistanceFromOrigin = 1
          val edgeThreshold: Float = 100
          val maxLines: Int = 20
          FactoryDetectLineAlgs.houghFoot(new ConfigHoughFoot(localMaxRadius, minCounts, minDistanceFromOrigin, edgeThreshold, maxLines), classOf[GrayU8], classOf[GrayS16])
        })
        val featureDetector1: DetectDescribePoint[GrayF32, BrightFeature] = {
          val detectThreshold = 10
          val extractRadius = 1
          val maxFeaturesPerScale = 500
          val initialSampleSize = 1
          val initialSize = 1
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
        }
        val transformModel: ModelMatcher[Homography2D_F64, AssociatedPair] = {
          val maxIterations = 100
          val inlierThreshold = 7
          val normalize = true
          FactoryMultiViewRobust.homographyRansac(new ConfigHomography(normalize), new ConfigRansac(maxIterations, inlierThreshold))
        }

        val found: util.List[LineParametric2D_F32] = rulerDetector.detect(ConvertBufferedImage.convertFromSingle(image1, null, classOf[GrayU8]))
        val horizontals = found.asScala.filter(line ⇒ Math.abs(line.slope.x) > Math.abs(line.slope.y)).toList
        val verticals = found.asScala.filter(line ⇒ Math.abs(line.slope.x) <= Math.abs(line.slope.y)).toList
        val candidateQuadrangles: List[Quadrilateral_F32] = log.code(() ⇒ {
          val imageBounds = new Rectangle2D_F32(0, 0, image1.getWidth, image1.getHeight)
          cross(pairs(horizontals), pairs(verticals)).map(xa ⇒ {
            val ((left: LineParametric2D_F32, right: LineParametric2D_F32), (top: LineParametric2D_F32, bottom: LineParametric2D_F32)) = xa
            new Quadrilateral_F32(
              Intersection2D_F32.intersection(left, top, null),
              Intersection2D_F32.intersection(left, bottom, null),
              Intersection2D_F32.intersection(right, top, null),
              Intersection2D_F32.intersection(right, bottom, null))
          }).filter((quad: Quadrilateral_F32) ⇒
            Intersection2D_F32.contains(imageBounds, quad.a.x, quad.a.y) &&
              Intersection2D_F32.contains(imageBounds, quad.b.x, quad.b.y) &&
              Intersection2D_F32.contains(imageBounds, quad.c.x, quad.c.y) &&
              Intersection2D_F32.contains(imageBounds, quad.d.x, quad.d.y)
          )
        })
        var bestQuadrangle: Quadrilateral_F32 = scale(rotate(log.code(() ⇒ {
          candidateQuadrangles.maxBy(quad ⇒ {
            val bounds = new Rectangle2D_F32()
            UtilPolygons2D_F32.bounding(quad, bounds)
            val area = quad.area()
            val squareness = area / bounds.area()
            assert(squareness >= 0 && squareness <= 1.01)
            area * Math.pow(squareness, 2)
          })
        })), 1.0f)

        log.draw(gfx ⇒ {
          gfx.drawImage(image1, 0, 0, null)
          gfx.setStroke(new BasicStroke(3))
          gfx.setColor(Color.YELLOW)
          draw(gfx, bestQuadrangle)
          gfx.setColor(Color.RED)
          draw(gfx, scale(bestQuadrangle, 0.9f))
        }, width = image1.getWidth, height = image1.getHeight)

        optimizeQuadrangle(log, image1, bestQuadrangle)

        val (areaHeight, areaWidth) = log.code(() ⇒ {
          (
            (bestQuadrangle.getSideLength(0) + bestQuadrangle.getSideLength(2)).toInt / 2,
            (bestQuadrangle.getSideLength(1) + bestQuadrangle.getSideLength(3)).toInt / 2
          )
        })
        val transform: Homography2D_F64 = log.code(() ⇒ {
          val pairs: util.ArrayList[AssociatedPair] = new util.ArrayList(List(
            new AssociatedPair(0, 0, bestQuadrangle.a.x, bestQuadrangle.a.y),
            new AssociatedPair(0, areaHeight, bestQuadrangle.c.x, bestQuadrangle.c.y),
            new AssociatedPair(areaWidth, 0, bestQuadrangle.b.x, bestQuadrangle.b.y),
            new AssociatedPair(areaWidth, areaHeight, bestQuadrangle.d.x, bestQuadrangle.d.y)
          ).asJava)
          if (!transformModel.process(pairs)) throw new RuntimeException("Model Matcher failed!")
          transformModel.getModelParameters
        })

        val primaryImage: BufferedImage = log.code(() ⇒ {
          val distortion: ImageDistort[Planar[GrayF32], Planar[GrayF32]] = {
            val interpolation = FactoryInterpolation.bilinearPixelS(classOf[GrayF32], BorderType.ZERO)
            val model = new PixelTransformHomography_F32
            val distort = DistortSupport.createDistortPL(classOf[GrayF32], model, interpolation, false)
            model.set(transform)
            distort.setRenderAll(false)
            distort
          }
          val boofImage = ConvertBufferedImage.convertFromMulti(image1, null, true, classOf[GrayF32])
          val work: Planar[GrayF32] = boofImage.createNew(areaWidth.toInt, areaHeight.toInt)
          distortion.apply(boofImage, work)
          val output = new BufferedImage(areaWidth.toInt, areaHeight.toInt, image1.getType)
          ConvertBufferedImage.convertTo(work, output, true)
          output
        })

        val images: List[BufferedImage] = rectify(log, featureDetector1, false)(primaryImage,
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard2.jpg")),
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard3.jpg"))
        )

        val tileBounds = new Rectangle2D_F32(750, 750, 2000, 1500)
        log.draw(gfx ⇒ {
          gfx.drawImage(primaryImage, 0, 0, null)
          gfx.setStroke(new BasicStroke(3))
          gfx.setColor(Color.YELLOW)
          gfx.drawRect(tileBounds.p0.x.toInt, tileBounds.p0.x.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt)
          gfx.setColor(Color.RED)
          gfx.drawRect(tileBounds.p0.x.toInt, tileBounds.p0.x.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt)
        }, height = primaryImage.getHeight, width = primaryImage.getWidth)

        var tileImages = rectify(log, featureDetector1, false)(
          images.head.getSubimage(tileBounds.p0.x.toInt, tileBounds.p0.x.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt),
          images.tail.map(_.getSubimage(tileBounds.p0.x.toInt, tileBounds.p0.x.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt)): _*)


        apply(log, img ⇒ {
          val binary = new GrayU8(img.width, img.height)
          val radius = 50
          val scale = 1.0
          val localGaussian = GThresholdImageOps.localGaussian(img, binary, radius, scale, true, null, null)
          var result = localGaussian
          result = BinaryImageOps.erode8(result, 2, null)
          result = BinaryImageOps.thin(result, 10, null)
          result
        })(
          images.head.getSubimage(tileBounds.p0.x.toInt, tileBounds.p0.x.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt),
          images.tail.map(_.getSubimage(tileBounds.p0.x.toInt, tileBounds.p0.x.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt)): _*)

        sketchExperiment(log)(
          images.head.getSubimage(tileBounds.p0.x.toInt, tileBounds.p0.x.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt),
          images.tail.map(_.getSubimage(tileBounds.p0.x.toInt, tileBounds.p0.x.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt)): _*)

      })

    }

  }

  def pairs[T](list: List[T]): List[(T, T)] = {
    (0 until list.size - 1).flatMap(i ⇒ {
      (i + 1 until list.size).map(j ⇒ {
        list(i) → list(j)
      })
    }).toList
  }

  def cross[T, U](a: List[T], b: List[U]): List[(T, U)] = {
    (0 until a.size).flatMap(i ⇒ {
      (0 until b.size).map(j ⇒ {
        a(i) → b(j)
      })
    }).toList
  }

  def optimizeQuadrangle(log: ScalaNotebookOutput, image: BufferedImage, startQuad: Quadrilateral_F32) = {
    val single = ConvertBufferedImage.convertFromSingle(image, null, classOf[GrayF32])
    val shapeThresholdImage = GThresholdImageOps.threshold(single, null, ImageStatistics.mean(single), true)
    val detectShape = log.code(() ⇒ {
      VisualizeBinaryData.renderBinary(shapeThresholdImage, false, null)
    })

    //
    //    import breeze.linalg.DenseVector
    //    import breeze.optimize.{ApproximateGradientFunction, LBFGS}
    //    def fn(v: DenseVector[Double]): Double = {
    //      val trialQuad: Quadrilateral_F32 = new Quadrilateral_F32(v(0).toFloat, v(1).toFloat, v(2).toFloat, v(3).toFloat, v(4).toFloat, v(5).toFloat, v(6).toFloat, v(7).toFloat)
    //      val stepSize = 2
    //      (0 until image.getWidth by stepSize).flatMap(x ⇒ {
    //        (0 until image.getHeight by stepSize).mapCoords(y ⇒ {
    //          val pt = new Point2D_F32(x, y)
    //          val distance = Distance2D_F32.distance(trialQuad, pt)
    //          val inside = if (Intersection2D_F32.contains(trialQuad, pt)) 1 else -1
    //          val pixel = shapeThresholdImage.get(x, y)
    //          (distance * inside * pixel)
    //        })
    //      }).sum
    //    }
    //    val startingPoint: DenseVector[Double] = DenseVector[Double](startQuad.a.x, startQuad.a.y, startQuad.b.x, startQuad.b.y,
    //      startQuad.c.x, startQuad.c.y, startQuad.d.x, startQuad.d.y)
    //    val gradientFunction = new ApproximateGradientFunction(fn _)
    //    val lbfgs: LBFGS[DenseVector[Double]] = new LBFGS[DenseVector[Double]](maxIter = 100, m = 3)
    //    val minimized: DenseVector[Double] = lbfgs.minimize(gradientFunction, startingPoint)
    //    val optimizedQuad = new Quadrilateral_F32(minimized(0).toFloat, minimized(1).toFloat, minimized(2).toFloat, minimized(3).toFloat,
    //      minimized(4).toFloat, minimized(5).toFloat, minimized(6).toFloat, minimized(7).toFloat)
    val optimizedQuad = startQuad
    log.draw(gfx ⇒ {
      gfx.drawImage(image, 0, 0, null)
      gfx.setStroke(new BasicStroke(3))
      gfx.setColor(Color.YELLOW)
      draw(gfx, optimizedQuad)
      gfx.setColor(Color.RED)
      draw(gfx, scale(optimizedQuad, 0.9f))
    }, width = image.getWidth, height = image.getHeight)
  }

  def scale(quad: Quadrilateral_F32, size: Float) = {
    val center = UtilPolygons2D_F32.center(quad, null)
    new Quadrilateral_F32(
      mix(quad.a, center, size),
      mix(quad.b, center, size),
      mix(quad.c, center, size),
      mix(quad.d, center, size)
    )
  }

  def draw(gfx: Graphics, quad: Quadrilateral_F32) = {
    gfx.drawPolygon(
      Array(
        quad.b.x.toInt,
        quad.a.x.toInt,
        quad.c.x.toInt,
        quad.d.x.toInt
      ),
      Array(
        quad.b.y.toInt,
        quad.a.y.toInt,
        quad.c.y.toInt,
        quad.d.y.toInt
      ), 4)
  }

  def apply(log: ScalaNotebookOutput, op: GrayF32 ⇒ GrayU8)(primaryImage: BufferedImage, secondaryImages: BufferedImage*) = {
    (List(primaryImage) ++ secondaryImages).map((img: BufferedImage) ⇒ {
      log.code(() ⇒ {
        val single = ConvertBufferedImage.convertFromSingle(img, null, classOf[GrayF32])
        val result = op(single)
        VisualizeBinaryData.renderBinary(result, false, null)
      })
    })
  }

  def applyMulti(log: ScalaNotebookOutput, op: Planar[GrayF32] ⇒ GrayU8)(primaryImage: BufferedImage, secondaryImages: BufferedImage*) = {
    (List(primaryImage) ++ secondaryImages).map((img: BufferedImage) ⇒ {
      log.code(() ⇒ {
        val multi = ConvertBufferedImage.convertFromMulti(img, null, false, classOf[GrayF32])
        val result = op(multi)
        VisualizeBinaryData.renderBinary(result, false, null)
      })
    })
  }

  def sketch[T <: TupleDesc[_]](log: ScalaNotebookOutput, detector: DetectLineSegment[GrayF32])(primaryImage: BufferedImage, secondaryImages: BufferedImage*) = {
    (List(primaryImage) ++ secondaryImages).map(img ⇒ {
      val segments: util.List[LineSegment2D_F32] = detector.detect(ConvertBufferedImage.convertFromSingle(img, null, classOf[GrayF32]))
      log.draw(gfx ⇒ {
        gfx.drawImage(img, 0, 0, null)
        gfx.setColor(Color.GREEN)
        segments.asScala.foreach(line ⇒ {
          gfx.drawLine(
            (line.a.x).toInt, (line.a.y).toInt,
            (line.b.x).toInt, (line.b.y).toInt)
        })
      }, width = img.getWidth, height = img.getHeight())
      segments
    })
  }

  def sketchExperiment[T <: TupleDesc[_]](log: ScalaNotebookOutput)(primaryImage: BufferedImage, secondaryImages: BufferedImage*) = {
    val detector = {
      val regionSize = 20
      val thresholdEdge = 50
      val thresholdAngle = 1.0
      val connectLines = true
      FactoryDetectLineAlgs.lineRansac(regionSize, thresholdEdge, thresholdAngle, connectLines, classOf[GrayF32], classOf[GrayF32])
    }
    (List(primaryImage) ++ secondaryImages).map(img ⇒ {

      val colorBand = {
        val rgb: Planar[GrayF32] = ConvertBufferedImage.convertFromMulti(img, null, true, classOf[GrayF32])
        val hsv = rgb.createSameShape()
        ColorHsv.rgbToHsv_F32(rgb, hsv)
        val bandImg: GrayF32 = hsv.getBand(2)
        val to = ConvertBufferedImage.convertTo(bandImg, null)
        VisualizeImageData.standard(bandImg, to)
      }

      val thresholded = {
        val single = ConvertBufferedImage.convertFromSingle(colorBand, null, classOf[GrayF32])
        val binary = new GrayU8(single.width, single.height)
        val radius = 50
        val scale = 1.0
        val localGaussian = GThresholdImageOps.localGaussian(single, binary, radius, scale, true, null, null)
        var result = localGaussian
        result = BinaryImageOps.erode8(result, 2, null)
        result = BinaryImageOps.thin(result, 10, null)
        VisualizeBinaryData.renderBinary(result, false, null)
      }

      val edgeDetectorSourceImage = thresholded
      val segments: util.List[LineSegment2D_F32] = detector.detect(ConvertBufferedImage.convertFromSingle(edgeDetectorSourceImage, null, classOf[GrayF32]))
      log.draw(gfx ⇒ {
        gfx.drawImage(edgeDetectorSourceImage, 0, 0, null)
        gfx.setStroke(new BasicStroke(3))
        gfx.setColor(Color.GREEN)
        segments.asScala.foreach(line ⇒ {
          gfx.drawLine(
            (line.a.x).toInt, (line.a.y).toInt,
            (line.b.x).toInt, (line.b.y).toInt)
        })
      }, width = edgeDetectorSourceImage.getWidth, height = edgeDetectorSourceImage.getHeight())
      segments
    })
  }

  def sketchExperiment2[T <: TupleDesc[_]](log: ScalaNotebookOutput)(primaryImage: BufferedImage, secondaryImages: BufferedImage*) = {
    val detector = {
      val regionSize = 20
      val thresholdEdge = 50
      val thresholdAngle = 1.0
      val connectLines = true
      FactoryDetectLineAlgs.lineRansac(regionSize, thresholdEdge, thresholdAngle, connectLines, classOf[GrayF32], classOf[GrayF32])
    }
    (List(primaryImage) ++ secondaryImages).map(img ⇒ resizeForFFT(img)).map(img ⇒ {
      val rgb: Planar[GrayF32] = ConvertBufferedImage.convertFromMulti(img, null, true, classOf[GrayF32])
      val hsv = rgb.createSameShape()
      ColorHsv.rgbToHsv_F32(rgb, hsv)

      val normalizedLighting: GrayF32 = freqFilter(hsv.getBand(2))
      val normalizedLightImg = log.code(() ⇒ {
        val to: BufferedImage = ConvertBufferedImage.convertTo(normalizedLighting, null)
        VisualizeImageData.standard(normalizedLighting, to)
      })
      val hsvNormalized = hsv.clone()
      hsvNormalized.setBand(2, normalizedLighting)
      val rgbNormalized = rgb.clone()
      ColorHsv.hsvToRgb_F32(hsvNormalized, rgbNormalized)
      log.code(() ⇒ {
        ConvertBufferedImage.convertTo_F32(rgbNormalized, null, false)
      })

      val colorBand = {
        val bandImg: GrayF32 = hsv.getBand(2)
        val to = ConvertBufferedImage.convertTo(bandImg, null)
        VisualizeImageData.standard(bandImg, to)
        //normalizedLightImg // override above
      }

      val thresholded: BufferedImage = {
        val single = ConvertBufferedImage.convertFromSingle(colorBand, null, classOf[GrayF32])
        val binary = new GrayU8(single.width, single.height)
        val radius = 60
        val scale = 1.0
        val localGaussian = GThresholdImageOps.localGaussian(single, binary, radius, scale, true, null, null)
        var result = localGaussian
        log.code(() ⇒ {
          VisualizeBinaryData.renderBinary(result, false, null)
        })
        result = BinaryImageOps.erode4(result, 1, null)
        result = BinaryImageOps.erode8(result, 1, null)
        //result = BinaryImageOps.thin(result, 10, null)
        log.code(() ⇒ {
          VisualizeBinaryData.renderBinary(result, false, null)
        })
      }

      val segmentation: BufferedImage = {
        val input = ConvertBufferedImage.convertFrom(thresholded, null: GrayF32)
        val imageType = ImageType.single(classOf[GrayF32])
        val alg = FactoryImageSegmentation.fh04(new ConfigFh04(100, 30), imageType)
        val segmentation = new GrayS32(thresholded.getWidth, thresholded.getHeight)
        alg.segment(input, segmentation)
        val superpixels = alg.getTotalSuperpixels
        log.code(() ⇒ {
          VisualizeRegions.regions(segmentation, superpixels, null)
        })

        val regionMemberCount = new GrowQueue_I32
        regionMemberCount.resize(superpixels)
        ImageSegmentationOps.countRegionPixels(segmentation, superpixels, regionMemberCount.data)

        val segmentColors: Map[Int, Array[Float]] = (0 until segmentation.getWidth).flatMap(x ⇒ (0 until segmentation.getHeight).map(y ⇒ {
          segmentation.get(x, y) → rgbNormalized.bands.map(_.get(x, y))
        })).groupBy(x ⇒ x._1).mapValues(_.map(_._2)).mapValues((pixels: immutable.Seq[Array[Float]]) ⇒ {
          (0 until 3).map(band ⇒ pixels.map(_ (band)).sum / pixels.size).toArray
        })

        val segmentColor: ColorQueue_F32 = new ColorQueue_F32(3)
        segmentColor.resize(superpixels)
        val averageLuminosity = ImageStatistics.mean(hsv.getBand(2))
        (0 until superpixels).foreach(i ⇒ {
          val count = regionMemberCount.get(i)
          val avgColor = segmentColors(i)
          val hsvColor = new Array[Float](3)
          val rgbColor = new Array[Float](3)
          ColorHsv.rgbToHsv(avgColor(0), avgColor(1), avgColor(2), hsvColor)
          val isWhite = hsvColor(1) < 0.05 && hsvColor(2) > averageLuminosity
          val isBlack = hsvColor(1) < 0.05 && hsvColor(2) < averageLuminosity
          if (count > 50 && count < 50000 && !isWhite) {
            hsvColor(2) = if (isBlack) 0.0f else 255.0f
            hsvColor(1) = if (isBlack) 0.0f else 1.0f
            ColorHsv.hsvToRgb(hsvColor(0), hsvColor(1), hsvColor(2), rgbColor)
            segmentColor.getData()(i) = rgbColor
          } else {
            //segmentColor.getTrainingData()(i) = Array(0.0f, 0.0f, 0.0f)
            segmentColor.getData()(i) = Array(255.0f, 255.0f, 255.0f)
          }
        })

        log.code(() ⇒ {
          VisualizeRegions.regionsColor(segmentation, segmentColor, null)
        })
      }


      val edgeDetectorSourceImage = segmentation
      val segments: util.List[LineSegment2D_F32] = detector.detect(ConvertBufferedImage.convertFromSingle(edgeDetectorSourceImage, null, classOf[GrayF32]))
      log.draw(gfx ⇒ {
        gfx.drawImage(edgeDetectorSourceImage, 0, 0, null)
        gfx.setStroke(new BasicStroke(3))
        gfx.setColor(Color.GREEN)
        segments.asScala.foreach(line ⇒ {
          gfx.drawLine(
            (line.a.x).toInt, (line.a.y).toInt,
            (line.b.x).toInt, (line.b.y).toInt)
        })
      }, width = edgeDetectorSourceImage.getWidth, height = edgeDetectorSourceImage.getHeight())
      segments
    })
  }

  def resizeForFFT(img: BufferedImage): BufferedImage = {
    def normalize(x: Double) = Math.pow(2.0, Math.ceil(Math.log(x) / Math.log(2))).toInt

    val image = new BufferedImage(normalize(img.getWidth), normalize(img.getHeight), BufferedImage.TYPE_INT_RGB)
    val gfx = image.getGraphics()
    val hints = new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC)
    gfx.asInstanceOf[Graphics2D].setRenderingHints(hints)
    gfx.drawImage(img, 0, 0, image.getWidth, image.getHeight, null)
    image
  }

  def freqFilter(bandImg: GrayF32): GrayF32 = {
    val minWidthFreq = 32
    val minAreaFreq = 256
    val copy = bandImg.createSameShape()
    val tensor: Array[Array[Float]] = (0 until bandImg.getWidth).map(x ⇒ (0 until bandImg.getHeight).map(y ⇒ bandImg.get(x, y)).toArray).toArray
    val fft = new FloatFFT_2D(bandImg.getWidth, bandImg.getHeight)
    fft.realForward(tensor)
    (0 until bandImg.getWidth).foreach(x ⇒ (0 until bandImg.getHeight).foreach(y ⇒ {
      if (!(x == 0 && y == 0) && (x < minWidthFreq && y < minWidthFreq) && ((x * y) < minAreaFreq)) tensor(x)(y) = 0.0f
    }))
    fft.realInverse(tensor, true)
    (0 until bandImg.getWidth).foreach(x ⇒ (0 until bandImg.getHeight).foreach(y ⇒ copy.set(x, y, tensor(x)(y))))
    copy
  }

  def colorSpaces[T <: TupleDesc[_]](log: ScalaNotebookOutput)(primaryImage: BufferedImage, secondaryImages: BufferedImage*) = (List(primaryImage) ++ secondaryImages).map(img ⇒ {
    log.draw(gfx ⇒ {
      import boofcv.io.image.ConvertBufferedImage
      import boofcv.struct.image.GrayF32
      val rgb: Planar[GrayF32] = ConvertBufferedImage.convertFromMulti(img, null, true, classOf[GrayF32])
      (0 until rgb.getNumBands).foreach(band ⇒ gfx.drawImage(ConvertBufferedImage.convertTo(rgb.getBand(band), null), img.getWidth * band, 0, null))

      val hsv = rgb.createSameShape()
      ColorHsv.rgbToHsv_F32(rgb, hsv)
      (0 until hsv.getNumBands).foreach(band ⇒ {
        val bandImg: GrayF32 = hsv.getBand(band)
        val to = ConvertBufferedImage.convertTo(bandImg, null)
        val display = VisualizeImageData.standard(bandImg, to)
        gfx.drawImage(display, img.getWidth * band, img.getHeight, null)
      })

      val yuv = rgb.createSameShape()
      ColorYuv.rgbToYuv_F32(rgb, yuv)
      (0 until yuv.getNumBands).foreach(band ⇒ {
        val bandImg: GrayF32 = yuv.getBand(band)
        val to = ConvertBufferedImage.convertTo(bandImg, null)
        val display = VisualizeImageData.standard(bandImg, to)
        gfx.drawImage(display, img.getWidth * band, img.getHeight * 2, null)
      })

      val xyz = rgb.createSameShape()
      ColorXyz.rgbToXyz_F32(rgb, xyz)
      (0 until xyz.getNumBands).foreach(band ⇒ {
        val bandImg: GrayF32 = xyz.getBand(band)
        val to = ConvertBufferedImage.convertTo(bandImg, null)
        val display = VisualizeImageData.standard(bandImg, to)
        gfx.drawImage(display, img.getWidth * band, img.getHeight * 3, null)
      })

      val lab = rgb.createSameShape()
      ColorLab.rgbToLab_F32(rgb, lab)
      (0 until lab.getNumBands).foreach(band ⇒ {
        val bandImg: GrayF32 = lab.getBand(band)
        val to = ConvertBufferedImage.convertTo(bandImg, null)
        val display = VisualizeImageData.standard(bandImg, to)
        gfx.drawImage(display, img.getWidth * band, img.getHeight * 4, null)
      })

    }, width = img.getWidth * 3, height = img.getHeight() * 5)
  })

  def removeGradualLighting[T <: TupleDesc[_]](log: ScalaNotebookOutput)(primaryImage: BufferedImage, secondaryImages: BufferedImage*) = (List(primaryImage) ++ secondaryImages)
    .map(img ⇒ resizeForFFT(img))
    .map(img ⇒ {
      log.draw(gfx ⇒ {
        val rgb: Planar[GrayF32] = ConvertBufferedImage.convertFromMulti(img, null, true, classOf[GrayF32])
        val hsv = rgb.createSameShape()
        ColorHsv.rgbToHsv_F32(rgb, hsv)
        val bandImg: GrayF32 = freqFilter(hsv.getBand(2))
        val to = ConvertBufferedImage.convertTo(bandImg, null)
        val display = VisualizeImageData.standard(bandImg, to)
        gfx.drawImage(display, 0, 0, null)
      }, width = img.getWidth * 1, height = img.getHeight() * 1)
    })

  def rectify[T <: TupleDesc[_]](log: ScalaNotebookOutput, featureDetector: DetectDescribePoint[GrayF32, T], expand: Boolean = true)(primaryImage: BufferedImage, secondaryImages: BufferedImage*): List[BufferedImage] = {
    val (pointsA, descriptionsA) = log.code(() ⇒ {
      describe[T](featureDetector, primaryImage)
    })

    def draw(primaryImage: BufferedImage, points: util.ArrayList[Point2D_F64], descriptions: FastQueue[T]) = {
      log.draw(gfx ⇒ {
        gfx.drawImage(primaryImage, 0, 0, null)
        points.asScala.zip(descriptions.toList.asScala).foreach(x ⇒ {
          val (pt, d) = x
          if (d.isInstanceOf[BrightFeature]) {
            if (d.asInstanceOf[BrightFeature].white) {
              gfx.setColor(Color.GREEN)
            } else {
              gfx.setColor(Color.RED)
            }
          } else {
            gfx.setColor(Color.YELLOW)
          }
          gfx.drawRect(pt.x.toInt - 4, pt.y.toInt - 4, 9, 9)
        })
      }, width = primaryImage.getWidth, height = primaryImage.getHeight())
    }

    draw(primaryImage, pointsA, descriptionsA)

    def findTransform(secondaryImage: BufferedImage) = {
      val (pointsB, descriptionsB) = log.code(() ⇒ {
        describe[T](featureDetector, secondaryImage)
      })
      draw(secondaryImage, pointsB, descriptionsB)
      val pairs: util.ArrayList[AssociatedPair] = associate[T](featureDetector, pointsA, descriptionsA, pointsB, descriptionsB)
      val modelMatcher: ModelMatcher[Homography2D_F64, AssociatedPair] = FactoryMultiViewRobust.homographyRansac(null, new ConfigRansac(60, 3));
      if (!modelMatcher.process(pairs)) throw new RuntimeException("Model Matcher failed!")
      modelMatcher.getModelParameters
    }

    val transforms: List[Homography2D_F64] = secondaryImages.map(findTransform).toList

    val boundsPoints: List[(Double, Double)] = log.code(() ⇒ {
      val boundsPoints = secondaryImages.zip(transforms).toMap.flatMap(x ⇒ {
        val (secondaryImage, transformParameters) = x
        val fromBtoA: Homography2D_F64 = transformParameters.invert(null)
        List((0, 0), (0, secondaryImage.getHeight), (secondaryImage.getWidth, 0), (secondaryImage.getWidth, secondaryImage.getHeight)).map(xx ⇒ {
          val (x: Int, y: Int) = xx
          transformXY(x, y, fromBtoA)
        })
      }).toList ++ List[(Double, Double)]((0, 0), (0, primaryImage.getHeight), (primaryImage.getWidth, 0), (primaryImage.getWidth, primaryImage.getHeight))
      System.out.println("renderMinX = " + boundsPoints.map(_._1).min)
      System.out.println("renderMaxX = " + boundsPoints.map(_._1).max)
      System.out.println("renderMinY = " + boundsPoints.map(_._2).min)
      System.out.println("renderMaxY = " + boundsPoints.map(_._2).max)
      boundsPoints
    })
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

    def transform(secondaryImage: BufferedImage, transformParameters: Homography2D_F64) = {
      val distortion: ImageDistort[Planar[GrayF32], Planar[GrayF32]] = log.code(() ⇒ {
        val interpolation = FactoryInterpolation.bilinearPixelS(classOf[GrayF32], BorderType.ZERO)
        val model = new PixelTransformHomography_F32
        val distort = DistortSupport.createDistortPL(classOf[GrayF32], model, interpolation, false)
        model.set(transformParameters)
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
    }

    val secondaryTransformed = secondaryImages.zip(transforms).map(x ⇒ {
      val (secondaryImage, transformParameters: Homography2D_F64) = x
      val transform1 = transform(secondaryImage, transformParameters)
      val refinement: Homography2D_F64 = findTransform(transform1)
      val transform2 = transform(secondaryImage, transformParameters.concat(refinement, null))
      transform2
    }).toList

    List({
      val output = new BufferedImage(renderWidth.toInt, renderHeight.toInt, primaryImage.getType)
      output.getGraphics.drawImage(primaryImage, offsetX.toInt, offsetY.toInt, null)
      output
    }) ++ secondaryTransformed

  }

  def transformXY(x0: Int, y0: Int, fromBtoWork: Homography2D_F64): (Double, Double) = {
    val result = new Point2D_F64
    HomographyPointOps_F64.transform(fromBtoWork, new Point2D_F64(x0, y0), result)
    val rx = result.x
    val ry = result.y
    (rx, ry)
  }

  def describe[T <: TupleDesc[_]](detDesc: DetectDescribePoint[GrayF32, T], image: BufferedImage): (util.ArrayList[Point2D_F64], FastQueue[T]) = {
    val points: util.ArrayList[Point2D_F64] = new util.ArrayList[Point2D_F64]()
    val input = ConvertBufferedImage.convertFromSingle(image, null, classOf[GrayF32])
    val descriptionType: Class[T] = detDesc.getDescriptionType
    val descriptions: FastQueue[T] = new FastQueue[T](100, descriptionType, true) {
      override protected def createInstance: T = detDesc.createDescription
    }
    //UtilFeature.createQueue(detDesc, 100)
    detDesc.detect(input)
    var i = 0
    while (i < detDesc.getNumberOfFeatures) {
      points.add(detDesc.getLocation(i).copy)
      val a: TupleDesc[T] = descriptions.grow.asInstanceOf[TupleDesc[T]]
      val b: T = detDesc.getDescription(i)
      a.setTo(b)
      i += 1;
    }
    (points, descriptions)
  }

  def associate[T <: TupleDesc[_]](detDesc: DetectDescribePoint[GrayF32, T], pointsA: util.ArrayList[Point2D_F64], descriptionsA: FastQueue[T],
                                   pointsB: util.ArrayList[Point2D_F64], descriptionsB: FastQueue[T]): util.ArrayList[AssociatedPair] = {
    val scorer = FactoryAssociation.defaultScore(detDesc.getDescriptionType)
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

  def scale(quad: Rectangle2D_F32, size: Float) = {
    val center = mix(quad.p0, quad.p1, 0.5f)
    val a = mix(quad.p0, center, size)
    val b = mix(quad.p1, center, size)
    new Rectangle2D_F32(
      a.x, a.y, b.x, b.y
    )
  }

  def mix(a: Point2D_F32, b: Point2D_F32, d: Float): Point2D_F32 = {
    return new Point2D_F32(a.x * d + b.x * (1 - d), a.y * d + b.y * (1 - d))
  }

  def rotate(r: Quadrilateral_F32): Quadrilateral_F32 = {
    val center = UtilPolygons2D_F32.center(r, null)
    if (r.a.x < center.x && r.a.y < center.y) r
    return new Quadrilateral_F32(r.d, r.c, r.b, r.a)
  }

}