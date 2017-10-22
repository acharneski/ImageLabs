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
import java.awt.{BasicStroke, Color, Graphics}
import java.io.FileInputStream
import java.nio.charset.Charset

import util.{ReportNotebook, ScalaNotebookOutput}
import java.util
import javax.imageio.ImageIO

import boofcv.abst.feature.detect.line.DetectLine
import boofcv.alg.color.ColorHsv
import boofcv.alg.distort.impl.DistortSupport
import boofcv.alg.distort.{ImageDistort, PixelTransformHomography_F32}
import boofcv.alg.filter.binary.{BinaryImageOps, GThresholdImageOps}
import boofcv.alg.filter.blur.GBlurImageOps
import boofcv.alg.misc.ImageStatistics
import boofcv.core.image.border.BorderType
import boofcv.factory.feature.detect.line.{ConfigHoughFoot, FactoryDetectLineAlgs}
import boofcv.factory.geo.{ConfigHomography, ConfigRansac, FactoryMultiViewRobust}
import boofcv.factory.interpolate.FactoryInterpolation
import boofcv.factory.segmentation._
import boofcv.gui.binary.VisualizeBinaryData
import boofcv.gui.feature.VisualizeRegions
import boofcv.gui.image.VisualizeImageData
import boofcv.io.image.ConvertBufferedImage
import boofcv.struct.feature._
import boofcv.struct.geo.AssociatedPair
import boofcv.struct.image.{GrayF32, GrayS32, ImageType, Planar, _}
import com.simiacryptus.util.ml.DensityTree
import georegression.geometry.UtilPolygons2D_F32
import georegression.metric.Intersection2D_F32
import georegression.struct.homography.Homography2D_F64
import georegression.struct.line.LineParametric2D_F32
import georegression.struct.point.Point2D_F32
import georegression.struct.shapes.{Quadrilateral_F32, Rectangle2D_F32}
import org.apache.commons.io.IOUtils
import org.ddogleg.fitting.modelset.ModelMatcher
import org.scalatest.{MustMatchers, WordSpec}

import scala.collection.JavaConverters._

class WhiteboardWorkflow extends WordSpec with MustMatchers with ReportNotebook {


  "Whiteboard Image Processing Demo" should {
    "Optimize whiteboard image" in {
      report("workflow", log ⇒ {
        log.p("First, we cache an photo of a whiteboard")
        val sourceImage = log.code(() ⇒ {
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard1.jpg"))
        })

        log.h2("Region Selection")
        val primaryImage: BufferedImage = rectifyQuadrangle(log, sourceImage)

        log.p("Now we refine our selection using some region selection, perhaps by manual selection")
        val tileBounds = log.code(() ⇒ {
          new Rectangle2D_F32(100, 40, 2700, 2100)
        })
        log.draw(gfx ⇒ {
          gfx.drawImage(primaryImage, 0, 0, null)
          gfx.setStroke(new BasicStroke(3))
          gfx.setColor(Color.RED)
          gfx.drawRect(tileBounds.p0.x.toInt, tileBounds.p0.y.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt)
        }, height = primaryImage.getHeight, width = primaryImage.getWidth)
        val tile = log.code(() ⇒ {
          primaryImage.getSubimage(tileBounds.p0.x.toInt, tileBounds.p0.y.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt)
        })

        log.h2("Color Normalization")
        val rgb: Planar[GrayF32] = ConvertBufferedImage.convertFromMulti(tile, null, true, classOf[GrayF32])
        // Pre-filter image to improve segmentation results
        GBlurImageOps.median(rgb, rgb, 1)
        GBlurImageOps.gaussian(rgb, rgb, 0.5, -1, null)
        // Prepare HSV
        val hsv = rgb.createSameShape()
        ColorHsv.rgbToHsv_F32(rgb, hsv)

        log.p("Here is an alternate method using direct-color segmentation:")
        val (superpixels3, segmentation3, filteredRgb) = findSuperpixels_Hybrid(log, hsv, rgb)
        val filteredHsv = rgb.createSameShape()
        ColorHsv.rgbToHsv_F32(filteredRgb, filteredHsv)
        colorize(log, filteredRgb, filteredHsv, superpixels3, segmentation3, "hybrid_segments")
      })
    }
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

  private def colorize(log: ScalaNotebookOutput, rgb: Planar[GrayF32], hsv: Planar[GrayF32], superpixels: Int, segmentation: GrayS32, name: String) = {
    log.p("For each segment, we categorize and colorize each using some logic")
    val (minHue, maxHue) = (ImageStatistics.min(hsv.getBand(0)), ImageStatistics.max(hsv.getBand(0)))
    val averageLuminosity = ImageStatistics.mean(hsv.getBand(2))
    val varianceLuminosity = ImageStatistics.variance(hsv.getBand(2), averageLuminosity)
    val superpixelParameters: Map[Int, Array[Double]] = log.code(() ⇒ {
      val regions = (0 until segmentation.getWidth).flatMap(x ⇒ (0 until segmentation.getHeight).map(y ⇒ {
        segmentation.get(x, y) → ((x, y) → rgb.bands.map(_.get(x, y)))
      })).groupBy(x ⇒ x._1).mapValues(_.map(t ⇒ t._2))
      regions.mapValues(pixels ⇒ {
        val rgvValues = pixels.map(_._2)
        val hsvValues = rgvValues.map(rgb ⇒ {
          val hsv = new Array[Float](3)
          ColorHsv.rgbToHsv(rgb(0), rgb(1), rgb(2), hsv)
          hsv
        })

        def statsHsv(fn: Array[Float] ⇒ (Float, Float)): (Float, Float) = {
          val stats = hsvValues.map((hsv: Array[Float]) ⇒ {
            val (weight, value) = fn(hsv)
            (weight, value * weight, value * value * weight)
          }).reduce((xa, xb) ⇒ (xa._1 + xb._1, xa._2 + xb._2, xa._3 + xb._3))
          val mean = stats._2 / stats._1
          val stdDev = Math.sqrt(Math.abs((stats._3 / stats._1) - mean * mean)).toFloat
          (mean, stdDev)
        }

        // Superpixel color statistics:
        val (hueMean1, hueStdDev1) = statsHsv((hsv: Array[Float]) ⇒ {
          (hsv(2) * hsv(1) * (1 - hsv(2)), hsv(0))
        })
        val (hueMean2, hueStdDev2) = statsHsv((hsv: Array[Float]) ⇒ {
          (hsv(2) * hsv(1) * (1 - hsv(2)), ((Math.PI + hsv(0)) % (2 * Math.PI)).toFloat)
        })
        val (hueMean: Float, hueStdDev: Float) = if (hueStdDev1 < hueStdDev2) {
          (hueMean1, hueStdDev1)
        } else {
          (((Math.PI + hueMean2) % (2 * Math.PI)).toFloat, hueStdDev2)
        }
        val (lumMean, lumStdDev) = statsHsv((hsv: Array[Float]) ⇒ {
          (1, hsv(2))
        })
        val (chromaMean, chromaStdDev) = statsHsv((hsv: Array[Float]) ⇒ {
          (1, hsv(2) * hsv(1))
        })
        // Superpixel geometry statistics:
        val xMax = pixels.map(_._1._1).max
        val xMin = pixels.map(_._1._1).min
        val yMax = pixels.map(_._1._2).max
        val yMin = pixels.map(_._1._2).min
        val length = Math.max(xMax - xMin, yMax - yMin)
        val area = pixels.size
        val width = area / length
        Array[Double](hueMean, hueStdDev, lumMean, lumStdDev, chromaMean, width, length)
      }).toArray.toMap
    })

    val fileOutputStream = log.file(name + ".csv")
    try {
      IOUtils.write(
        (0 until superpixels).map(i ⇒ superpixelParameters(i).mkString(",")).mkString("\n"),
        fileOutputStream, Charset.forName("UTF-8"))
    } finally {
      fileOutputStream.close()
    }
    clusterAnalysis_density(log, name)

    log.p("Now, we recolor the image by classifying each superpixel as white, black, or color:");
    val colorizedImg = log.code(() ⇒ {
      val segmentColors: ColorQueue_F32 = new ColorQueue_F32(3)
      segmentColors.resize(superpixels)
      (0 until superpixels).foreach(i ⇒ {
        segmentColors.getData()(i) = {
          val p = superpixelParameters(i)
          val (hueMean: Float, hueStdDev: Float, lumMean: Float, lumStdDev: Float, chromaMean: Float, width: Int, length: Int) =
            (p(0).floatValue(), p(1).floatValue(), p(2).floatValue(), p(3).floatValue(), p(4).floatValue(), p(5).intValue(), p(6).intValue())
          var isColored = false
          var isBlack = false
          var isWhite = false
          if (lumStdDev < 1.5) {
            isWhite = true
          } else {
            if (hueStdDev < 0.05) {
              isColored = true
            } else {
              if (chromaMean < 5.0) {
                isBlack = true
              } else {
                isColored = true
              }
            }
          }
          val aspect = length.toDouble / width
          val isMarkingShape = aspect > 2 && width < 35
          val isMarking = isMarkingShape && !isWhite
          if (isMarking) {
            if (isBlack) {
              Array(0.0f, 0.0f, 0.0f)
            } else {
              val rgb = new Array[Float](3)
              ColorHsv.hsvToRgb(hueMean, 1.0f, 255.0f, rgb)
              rgb
            }
          } else {
            Array(255.0f, 255.0f, 255.0f)
          }
        }
      })
      VisualizeRegions.regionsColor(segmentation, segmentColors, null)
    })

    log.p("Finally, we trim all the whitespace: ");
    log.draw(gfx ⇒ {
      val pixels = (0 until colorizedImg.getWidth).flatMap(x ⇒
        (0 until colorizedImg.getHeight).filterNot(y ⇒
          util.Arrays.equals(colorizedImg.getRaster.getPixel(x, y, null: Array[Int]), Array(255, 255, 255))).map(y ⇒ x → y))
      var (xmin, xmax) = (pixels.map(_._1).min, pixels.map(_._1).max)
      var (ymin, ymax) = (pixels.map(_._2).min, pixels.map(_._2).max)
      val regionAspect = (xmax - xmin).toDouble / (ymax - ymin)
      val imageAspect = colorizedImg.getWidth.toDouble / colorizedImg.getHeight
      if (regionAspect < imageAspect) {
        val width = (colorizedImg.getWidth * (ymax - ymin)) / colorizedImg.getHeight
        xmin = Math.max(0, (xmax + xmin - width) / 2)
        xmax = xmin + width
      } else {
        val height = (colorizedImg.getHeight * (xmax - xmin)) / colorizedImg.getWidth
        ymin = Math.max(0, (ymax + ymin - height) / 2)
        ymax = ymin + height
      }
      gfx.drawImage(colorizedImg, 0, 0, colorizedImg.getWidth, colorizedImg.getHeight, xmin, ymin, xmax, ymax, null)
    }, width = colorizedImg.getWidth, height = colorizedImg.getHeight)
  }

  private def clusterAnalysis_density[T](log: ScalaNotebookOutput, name: String): Unit = {
    val stream = new FileInputStream("reports/WhiteboardWorkflow/etc/" + name + ".csv")
    val data = try {
      IOUtils.toString(stream, Charset.forName("UTF-8")).split("\n").map(_.split(",").map(java.lang.Double.parseDouble(_)))
    } finally {
      stream.close()
    }
    val superpixels = data.length
    val superpixelParameters = (0 until superpixels).map(i ⇒ i → data(i)).filterNot(_._2.contains(Double.NaN)).toMap

    def stats[T](numbers: Seq[T])(implicit n: Numeric[T]) = {
      val sum0 = numbers.size
      val sum1 = numbers.map(n.toDouble).sum
      val sum2 = numbers.map(n.toDouble).map(x ⇒ x * x).sum
      val mean = sum1 / sum0
      val stdDev = Math.sqrt((sum2 / sum0) - (mean * mean))
      Map("c" → numbers.size, "m" → mean) ++ (if (0.0 < stdDev) Map("v" → stdDev) else Map.empty)
    }

    def summary(superpixels: Array[Array[Double]]) = {
      Map(
        "hueMean" → stats(superpixels.map(_ (0))),
        "hueStdDev" → stats(superpixels.map(_ (1))),
        "lumMean" → stats(superpixels.map(_ (2))),
        "lumStdDev" → stats(superpixels.map(_ (3))),
        "chromaMean" → stats(superpixels.map(_ (4))),
        "width" → stats(superpixels.map(_ (5))),
        "length" → stats(superpixels.map(_ (6)))
      )
    }

    log.p("To help interpret the structure of this data set, we newTrainer a density tree:");
    val densityModel = log.code(() ⇒ {
      val tree = new DensityTree("hueMean", "hueStdDev", "lumMean", "lumStdDev", "chromaMean", "width", "length")
      tree.setSplitSizeThreshold(2)
      tree.setMinFitness(2)
      tree.setMaxDepth(3)
      new tree.Node((0 until superpixels).map(superpixelParameters(_)).toArray)
    })
    val fileOutputStream = log.file(name + "_tree.txt")
    try {
      IOUtils.write(densityModel.toString(), fileOutputStream, Charset.forName("UTF-8"))
    } finally {
      fileOutputStream.close()
    }

  }

  private def distance(a: Array[Double], b: Array[Double]) = {
    Math.sqrt((0 until a.length).map(i ⇒ {
      val y = a(i) - b(i)
      y * y
    }).sum)
  }

  private def findSuperpixels_Hybrid(log: ScalaNotebookOutput, hsv: Planar[GrayF32], rgb: Planar[GrayF32]) = {
    val finalBinaryMask = threshold(log, hsv, rgb)
    val thresholdImg = log.code(() ⇒ {
      VisualizeBinaryData.renderBinary(finalBinaryMask, false, null)
    })

    log.p("Use threshold mask to generate a background image")
    val averageRGB = log.code(() ⇒ {
      (0 until 3).map(b ⇒ ImageStatistics.mean(rgb.getBand(b)))
    })
    val maskedBackground: Planar[GrayF32] = log.code(() ⇒ {
      val mask = BinaryImageOps.dilate8(BinaryImageOps.thin(finalBinaryMask.clone(), 2, null), 5, null)
      val maskedBackground: Planar[GrayF32] = rgb.clone()
      (0 until maskedBackground.getWidth).foreach(x ⇒
        (0 until maskedBackground.getHeight).foreach(y ⇒
          (0 until maskedBackground.getNumBands).foreach(b ⇒
            if (mask.get(x, y) != 0) {
              maskedBackground.getBand(b).set(x, y, averageRGB(b))
            })))
      List(150.0, 50.0, 15.0, 5.0).foreach(blurRadius ⇒ {
        val nextIteration: Planar[GrayF32] = maskedBackground.clone()
        GBlurImageOps.gaussian(nextIteration, nextIteration, blurRadius, -1, null)
        (0 until maskedBackground.getWidth).foreach(x ⇒
          (0 until maskedBackground.getHeight).foreach(y ⇒
            (0 until maskedBackground.getNumBands).foreach(b ⇒
              if (mask.get(x, y) != 0) {
                maskedBackground.getBand(b).set(x, y, nextIteration.getBand(b).get(x, y))
              })))
      })
      maskedBackground
    })
    log.code(() ⇒ {
      ConvertBufferedImage.convertTo(maskedBackground, null, false)
    })

    log.p("Use threshold mask to generate a mask the foreground image (contrasted mapCoords the background)")
    val maskedForground: Planar[GrayF32] = log.code(() ⇒ {
      val maskedForground: Planar[GrayF32] = rgb.clone()
      (0 until maskedForground.getWidth).foreach(x ⇒
        (0 until maskedForground.getHeight).foreach(y ⇒
          (0 until maskedForground.getNumBands).foreach(b ⇒
            if (finalBinaryMask.get(x, y) == 0) {
              maskedForground.getBand(b).set(x, y, 255.0f)
            } else {
              val forground = maskedForground.getBand(b).get(x, y)
              val background = maskedBackground.getBand(b).get(x, y)
              maskedForground.getBand(b).set(x, y, forground * (255.0f / background))
            })))
      maskedForground
    })
    log.code(() ⇒ {
      ConvertBufferedImage.convertTo(maskedForground, null, false)
    })

    log.p("We can identify segments which may be markings using the masked color image:")
    val (superpixels, segmentation) = log.code(() ⇒ {
      val imageType = ImageType.pl(3, classOf[GrayF32])
      val alg = FactoryImageSegmentation.fh04(new ConfigFh04(0.1f, 10), imageType)
      val segmentation = new GrayS32(rgb.getWidth, rgb.getHeight)
      alg.segment(maskedForground, segmentation)
      (alg.getTotalSuperpixels, segmentation)
    })
    log.code(() ⇒ {
      VisualizeRegions.regions(segmentation, superpixels, null)
    })
    (superpixels, segmentation, maskedForground)
  }

  private def threshold(log: ScalaNotebookOutput, hsv: Planar[GrayF32], rgb: Planar[GrayF32]) = {
    log.p("Dectection of markings uses the luminosity")
    val colorBand = log.code(() ⇒ {
      val bandImg: GrayF32 = hsv.getBand(2)
      val to = ConvertBufferedImage.convertTo(bandImg, null)
      VisualizeImageData.standard(bandImg, to)
    })
    log.p("...by detecting local variations")
    val binaryMask = log.code(() ⇒ {
      val single = ConvertBufferedImage.convertFromSingle(colorBand, null, classOf[GrayF32])
      val binary = new GrayU8(single.width, single.height)
      GThresholdImageOps.localSauvola(single, binary, 50, 0.2f, true)
    })
    log.code(() ⇒ {
      VisualizeBinaryData.renderBinary(binaryMask, false, null)
    })
    binaryMask
  }


  private def rectifyQuadrangle(log: ScalaNotebookOutput, sourceImage: BufferedImage) = {
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
      gfx.setStroke(new BasicStroke(3))
      found.asScala.foreach(line ⇒ {
        if (Math.abs(line.slope.x) > Math.abs(line.slope.y)) {
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
    primaryImage
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