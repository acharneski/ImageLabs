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

import java.awt.image.BufferedImage
import java.awt.{BasicStroke, Color, Graphics}
import java.util
import javax.imageio.ImageIO

import boofcv.abst.feature.detect.line.DetectLine
import boofcv.alg.color.ColorHsv
import boofcv.alg.distort.impl.DistortSupport
import boofcv.alg.distort.{ImageDistort, PixelTransformHomography_F32}
import boofcv.alg.filter.binary.{BinaryImageOps, GThresholdImageOps}
import boofcv.alg.misc.ImageStatistics
import boofcv.alg.segmentation.ImageSegmentationOps
import boofcv.core.image.border.BorderType
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
import georegression.struct.line.LineParametric2D_F32
import georegression.struct.point.Point2D_F32
import georegression.struct.shapes.{Quadrilateral_F32, Rectangle2D_F32}
import org.ddogleg.fitting.modelset.ModelMatcher
import org.ddogleg.struct.GrowQueue_I32
import org.scalatest.{MustMatchers, WordSpec}

import scala.collection.JavaConverters._
import scala.collection.immutable

class WhiteboardWorkflow extends WordSpec with MustMatchers with MarkdownReporter {

  "Whiteboard Image Processing Demo" should {
    "Optimize whiteboard image" in {
      report("workflow", log ⇒ {

        log.p("First, we load an photo of a whiteboard")
        val sourceImage = log.code(()⇒{
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard1.jpg"))
        })

        log.p("We start looking for long edges which can be used to find the board:")
        val found: util.List[LineParametric2D_F32] = log.code(() ⇒ {
          val rulerDetector: DetectLine[GrayU8] = log.code(() ⇒ {
            val localMaxRadius = 10
            val minCounts = 5
            val minDistanceFromOrigin = 1
            val edgeThreshold: Float = 100
            val maxLines: Int = 20
            FactoryDetectLineAlgs.houghFoot(new ConfigHoughFoot(localMaxRadius, minCounts, minDistanceFromOrigin, edgeThreshold, maxLines), classOf[GrayU8], classOf[GrayS16])
          })
          rulerDetector.detect(ConvertBufferedImage.convertFromSingle(sourceImage, null, classOf[GrayU8]))
        })
        log.draw(gfx ⇒ {
          gfx.drawImage(sourceImage, 0, 0, null)
          found.asScala.foreach(line ⇒ {
            if(Math.abs(line.slope.x) > Math.abs(line.slope.y)) {
              val x1 = 0
              val y1 = (line.p.y - line.p.x * line.slope.y / line.slope.x).toInt
              val x2 = sourceImage.getWidth
              val y2 = y1 + (x2 * line.slope.y / line.slope.x).toInt
              gfx.setColor(Color.RED)
              gfx.drawLine(
                x1, y1,
                x2, y2)
            } else {
              val y1 = 0
              val x1 = (line.p.x - line.p.y * line.slope.x / line.slope.y).toInt
              val y2 = sourceImage.getHeight
              val x2 = x1 + (y2 * line.slope.x / line.slope.y).toInt
              gfx.setColor(Color.GREEN)
              gfx.drawLine(
                x1, y1,
                x2, y2)
            }
          })
        }, width = sourceImage.getWidth, height = sourceImage.getHeight)

        log.p("This can then be searched for the largest, most upright, and rectangular shape")
        var bestQuadrangle: Quadrilateral_F32 = log.code(() ⇒ {
          val horizontals = found.asScala.filter(line ⇒ Math.abs(line.slope.x) > Math.abs(line.slope.y)).toList
          val verticals = found.asScala.filter(line ⇒ Math.abs(line.slope.x) <= Math.abs(line.slope.y)).toList
          val imageBounds = new Rectangle2D_F32(0, 0, sourceImage.getWidth, sourceImage.getHeight)
          val candidateQuadrangles: List[Quadrilateral_F32] = cross(pairs(horizontals), pairs(verticals)).map(xa ⇒ {
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
          scale(rotate(
            candidateQuadrangles.maxBy(quad ⇒ {
              val bounds = new Rectangle2D_F32()
              UtilPolygons2D_F32.bounding(quad, bounds)
              val area = quad.area()
              val squareness = area / bounds.area()
              assert(squareness >= 0 && squareness <= 1.01)
              area * Math.pow(squareness, 2)
            })
          ), 1.0f)
        })
        log.draw(gfx ⇒ {
          gfx.drawImage(sourceImage, 0, 0, null)
          gfx.setStroke(new BasicStroke(3))
          gfx.setColor(Color.RED)
          draw(gfx, bestQuadrangle)
        }, width = sourceImage.getWidth, height = sourceImage.getHeight)

        log.p("We then distort the image using a homographic transform back into a rectangle. First we estimate the correct size of the image:")
        val (areaHeight, areaWidth) = log.code(() ⇒ {
          (
            (bestQuadrangle.getSideLength(0) + bestQuadrangle.getSideLength(2)).toInt / 2,
            (bestQuadrangle.getSideLength(1) + bestQuadrangle.getSideLength(3)).toInt / 2
          )
        })

        log.p("We derive the transform:")
        val transform: Homography2D_F64 = log.code(() ⇒ {
          val transformModel: ModelMatcher[Homography2D_F64, AssociatedPair] = {
            val maxIterations = 100
            val inlierThreshold = 7
            val normalize = true
            FactoryMultiViewRobust.homographyRansac(new ConfigHomography(normalize), new ConfigRansac(maxIterations, inlierThreshold))
          }
          val pairs: util.ArrayList[AssociatedPair] = new util.ArrayList(List(
            new AssociatedPair(0, 0, bestQuadrangle.a.x, bestQuadrangle.a.y),
            new AssociatedPair(0, areaHeight, bestQuadrangle.c.x, bestQuadrangle.c.y),
            new AssociatedPair(areaWidth, 0, bestQuadrangle.b.x, bestQuadrangle.b.y),
            new AssociatedPair(areaWidth, areaHeight, bestQuadrangle.d.x, bestQuadrangle.d.y)
          ).asJava)
          if (!transformModel.process(pairs)) throw new RuntimeException("Model Matcher failed!")
          transformModel.getModelParameters
        })

        log.p("And we transform the image:")
        val primaryImage: BufferedImage = log.code(() ⇒ {
          val distortion: ImageDistort[Planar[GrayF32], Planar[GrayF32]] = {
            val interpolation = FactoryInterpolation.bilinearPixelS(classOf[GrayF32], BorderType.ZERO)
            val model = new PixelTransformHomography_F32
            val distort = DistortSupport.createDistortPL(classOf[GrayF32], model, interpolation, false)
            model.set(transform)
            distort.setRenderAll(false)
            distort
          }
          val boofImage = ConvertBufferedImage.convertFromMulti(sourceImage, null, true, classOf[GrayF32])
          val work: Planar[GrayF32] = boofImage.createNew(areaWidth.toInt, areaHeight.toInt)
          distortion.apply(boofImage, work)
          val output = new BufferedImage(areaWidth.toInt, areaHeight.toInt, sourceImage.getType)
          ConvertBufferedImage.convertTo(work, output, true)
          output
        })

        log.p("Now we refine our selection using some region selection, perhaps by manual selection")
        val tileBounds = log.code(()⇒{
          new Rectangle2D_F32(100, 40, 2700, 2100)
        })
        log.draw(gfx ⇒ {
          gfx.drawImage(primaryImage, 0, 0, null)
          gfx.setStroke(new BasicStroke(3))
          gfx.setColor(Color.RED)
          gfx.drawRect(tileBounds.p0.x.toInt, tileBounds.p0.x.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt)
        }, height = primaryImage.getHeight, width = primaryImage.getWidth)
        val tile = log.code(()⇒{
          primaryImage.getSubimage(tileBounds.p0.x.toInt, tileBounds.p0.x.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt)
        })

        val rgb: Planar[GrayF32] = ConvertBufferedImage.convertFromMulti(tile, null, true, classOf[GrayF32])
        val hsv = rgb.createSameShape()
        ColorHsv.rgbToHsv_F32(rgb, hsv)

        log.p("Dectection of markings uses the luminosity")
        val colorBand = log.code(() ⇒ {
          val bandImg: GrayF32 = hsv.getBand(2)
          val to = ConvertBufferedImage.convertTo(bandImg, null)
          VisualizeImageData.standard(bandImg, to)
        })

        log.p("...by detecting local variations within a gaussian radius")
        val localGaussian = log.code(() ⇒ {
          val single = ConvertBufferedImage.convertFromSingle(colorBand, null, classOf[GrayF32])
          val binary = new GrayU8(single.width, single.height)
          val radius = 60
          val scale = 1.0
          GThresholdImageOps.localGaussian(single, binary, radius, scale, true, null, null)
        })
        log.code(() ⇒ {
          VisualizeBinaryData.renderBinary(localGaussian, false, null)
        })

        log.p("This binarization is then refined by eroding and thinning operations")
        val thresholdImg: BufferedImage = log.code(() ⇒ {
          var prefiltered = localGaussian
          prefiltered = BinaryImageOps.erode4(prefiltered, 1, null)
          prefiltered = BinaryImageOps.erode8(prefiltered, 1, null)
          VisualizeBinaryData.renderBinary(prefiltered, false, null)
        })

        log.p("We can now identify segments which may be markings:")
        val (superpixels, segmentation) = log.code(() ⇒ {
          val input = ConvertBufferedImage.convertFrom(thresholdImg, null: GrayF32)
          val imageType = ImageType.single(classOf[GrayF32])
          val alg = FactoryImageSegmentation.fh04(new ConfigFh04(100, 30), imageType)
          val segmentation = new GrayS32(thresholdImg.getWidth, thresholdImg.getHeight)
          alg.segment(input, segmentation)
          (alg.getTotalSuperpixels, segmentation)
        })
        log.code(() ⇒ {
          VisualizeRegions.regions(segmentation, superpixels, null)
        })

        log.p("For each segment, we categorize and colorize each using some logic")
        val segmentationImg: BufferedImage = log.code(() ⇒ {
          val regionMemberCount = new GrowQueue_I32
          regionMemberCount.resize(superpixels)
          ImageSegmentationOps.countRegionPixels(segmentation, superpixels, regionMemberCount.data)
          val avgColors: Map[Int, Array[Float]] = (0 until segmentation.getWidth).flatMap(x ⇒ (0 until segmentation.getHeight).map(y ⇒ {
            segmentation.get(x, y) → rgb.bands.map(_.get(x,y))
          })).groupBy(x⇒x._1).mapValues(_.map(_._2)).mapValues((pixels: immutable.Seq[Array[Float]]) ⇒ {
            (0 until 3).map(band⇒pixels.map(_(band)).sum / pixels.size).toArray
          })
          val segmentColors: ColorQueue_F32 = new ColorQueue_F32(3)
          segmentColors.resize(superpixels)
          val averageLuminosity = ImageStatistics.mean(hsv.getBand(2))
          (0 until superpixels).foreach(i ⇒ {
            val count = regionMemberCount.get(i)
            val avgColor = avgColors(i)
            val hsvColor = new Array[Float](3)
            val rgbColor = new Array[Float](3)
            ColorHsv.rgbToHsv(avgColor(0),avgColor(1),avgColor(2),hsvColor)
            val isWhite = hsvColor(1) < 0.05 && hsvColor(2) > averageLuminosity
            val isBlack = hsvColor(1) < 0.05 && hsvColor(2) < averageLuminosity
            if (count > 50 && count < 50000 && !isWhite) {
              hsvColor(2) = if(isBlack) 0.0f else 255.0f
              hsvColor(1) = if(isBlack) 0.0f else 1.0f
              ColorHsv.hsvToRgb(hsvColor(0),hsvColor(1),hsvColor(2),rgbColor)
              segmentColors.getData()(i) = rgbColor
            } else {
              segmentColors.getData()(i) = Array(255.0f, 255.0f, 255.0f)
            }
          })
          VisualizeRegions.regionsColor(segmentation, segmentColors, null)
        })

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

  def scale(quad: Quadrilateral_F32, size: Float) = {
    val center = UtilPolygons2D_F32.center(quad, null)
    new Quadrilateral_F32(
      mix(quad.a, center, size),
      mix(quad.b, center, size),
      mix(quad.c, center, size),
      mix(quad.d, center, size)
    )
  }

  def mix(a: Point2D_F32, b: Point2D_F32, d: Float): Point2D_F32 = {
    return new Point2D_F32(a.x * d + b.x * (1 - d), a.y * d + b.y * (1 - d))
  }

  def scale(quad: Rectangle2D_F32, size: Float) = {
    val center = mix(quad.p0, quad.p1, 0.5f)
    val a = mix(quad.p0, center, size)
    val b = mix(quad.p1, center, size)
    new Rectangle2D_F32(
      a.x, a.y, b.x, b.y
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

  def rotate(r: Quadrilateral_F32): Quadrilateral_F32 = {
    val center = UtilPolygons2D_F32.center(r, null)
    if (r.a.x < center.x && r.a.y < center.y) r
    return new Quadrilateral_F32(r.d, r.c, r.b, r.a)
  }

}