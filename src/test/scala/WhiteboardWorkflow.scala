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
import java.awt.{BasicStroke, Color, Graphics, Graphics2D, RenderingHints}
import java.util
import javax.imageio.ImageIO

import boofcv.abst.feature.describe.ConfigSurfDescribe.Stability
import boofcv.abst.feature.detdesc.DetectDescribePoint
import boofcv.abst.feature.detect.interest.ConfigFastHessian
import boofcv.abst.feature.detect.line.{DetectLine, DetectLineSegment}
import boofcv.abst.feature.orientation.ConfigSlidingIntegral
import boofcv.alg.descriptor.UtilFeature
import boofcv.alg.distort.impl.DistortSupport
import boofcv.alg.distort.{ImageDistort, PixelTransformHomography_F32}
import boofcv.core.image.border.BorderType
import boofcv.factory.feature.associate.FactoryAssociation
import boofcv.factory.feature.detdesc.FactoryDetectDescribe
import boofcv.factory.feature.detect.line.{ConfigHoughFoot, ConfigHoughFootSubimage, ConfigHoughPolar, FactoryDetectLineAlgs}
import boofcv.factory.geo.{ConfigHomography, ConfigRansac, FactoryMultiViewRobust}
import boofcv.factory.interpolate.FactoryInterpolation
import boofcv.io.image.ConvertBufferedImage
import boofcv.struct.feature.{AssociatedIndex, BrightFeature}
import boofcv.struct.geo.AssociatedPair
import boofcv.struct.image.{GrayF32, GrayS16, GrayU8, Planar}
import georegression.geometry.UtilPolygons2D_F32
import georegression.metric.Intersection2D_F32
import georegression.struct.homography.Homography2D_F64
import georegression.struct.line.{LineParametric2D_F32, LineSegment2D_F32}
import georegression.struct.point.{Point2D_F32, Point2D_F64}
import georegression.struct.shapes.{Quadrilateral_F32, Rectangle2D_F32}
import georegression.transform.homography.HomographyPointOps_F64
import org.ddogleg.fitting.modelset.ModelMatcher
import org.ddogleg.struct.FastQueue
import org.scalatest.{MustMatchers, WordSpec}

import scala.collection.JavaConverters._

class WhiteboardWorkflow extends WordSpec with MustMatchers with MarkdownReporter {

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

  "Image Workflows" should {
    "Rectify whiteboard image sets" in {
      report("whiteboard", log ⇒ {

        val image1 = ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard1.jpg"))
        val detector: DetectLine[GrayU8] = log.code(() ⇒ {
          val localMaxRadius = 4
          val minCounts = 5
          val minDistanceFromOrigin = 1
          val edgeThreshold: Float = 100
          val maxLines: Int = 20
          FactoryDetectLineAlgs.houghFoot(new ConfigHoughFoot(localMaxRadius, minCounts, minDistanceFromOrigin, edgeThreshold, maxLines), classOf[GrayU8], classOf[GrayS16])
        })
        val featureDetector: DetectDescribePoint[GrayF32, BrightFeature] = {
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
          val radius = 8
          val weightSigma = -1.0
          val sampleWidth = 6
          val stabilityConfig = new Stability
          FactoryDetectDescribe.surfStable(fastHessian, stabilityConfig, new ConfigSlidingIntegral(samplePeriod, windowSize, radius, weightSigma, sampleWidth), classOf[GrayF32])
        }
        val modelMatcher: ModelMatcher[Homography2D_F64, AssociatedPair] = {
          val maxIterations = 60
          val inlierThreshold = 5
          val normalize = true
          FactoryMultiViewRobust.homographyRansac(new ConfigHomography(normalize), new ConfigRansac(maxIterations, inlierThreshold))
        }

        val found: util.List[LineParametric2D_F32] = detector.detect(ConvertBufferedImage.convertFromSingle(image1, null, classOf[GrayU8]))
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
        val bestQuadrangle: Quadrilateral_F32 = scale(rotate(log.code(() ⇒ {
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

        val (areaHeight, areaWidth) = log.code(()⇒{(
            (bestQuadrangle.getSideLength(0)+bestQuadrangle.getSideLength(2)).toInt / 2,
            (bestQuadrangle.getSideLength(1)+bestQuadrangle.getSideLength(3)).toInt / 2
        )})
        val transform: Homography2D_F64 = log.code(() ⇒ {
          val pairs: util.ArrayList[AssociatedPair] = new util.ArrayList(List(
            new AssociatedPair(0,0,bestQuadrangle.a.x,bestQuadrangle.a.y),
            new AssociatedPair(0,areaHeight,bestQuadrangle.c.x,bestQuadrangle.c.y),
            new AssociatedPair(areaWidth,0,bestQuadrangle.b.x,bestQuadrangle.b.y),
            new AssociatedPair(areaWidth,areaHeight,bestQuadrangle.d.x,bestQuadrangle.d.y)
          ).asJava)
          if (!modelMatcher.process(pairs)) throw new RuntimeException("Model Matcher failed!")
          modelMatcher.getModelParameters
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

        val images: List[BufferedImage] = rectify(log,featureDetector,false)(primaryImage,
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard2.jpg")),
          ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard3.jpg"))
        )

        val tileBounds = new Rectangle2D_F32(1000,1000,1500,1500)
        val superTileBounds = scale(tileBounds, 1.5f)
        log.draw(gfx⇒{
          gfx.drawImage(primaryImage,0,0,null)
          gfx.setStroke(new BasicStroke(3))
          gfx.setColor(Color.YELLOW)
          gfx.drawRect(tileBounds.p0.x.toInt, tileBounds.p0.x.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt)
          gfx.setColor(Color.RED)
          gfx.drawRect(superTileBounds.p0.x.toInt, superTileBounds.p0.x.toInt, superTileBounds.getWidth.toInt, superTileBounds.getHeight.toInt)
        },height=primaryImage.getHeight,width=primaryImage.getWidth)

        val tileImages = rectify(log,featureDetector,false)(
          images.head.getSubimage(tileBounds.p0.x.toInt, tileBounds.p0.x.toInt, tileBounds.getWidth.toInt, tileBounds.getHeight.toInt),
          images.tail.map(_.getSubimage(superTileBounds.p0.x.toInt, superTileBounds.p0.x.toInt, superTileBounds.getWidth.toInt, superTileBounds.getHeight.toInt)):_*)

      })
    }

  }

  def rectify(log:ScalaMarkdownPrintStream,featureDetector : DetectDescribePoint[GrayF32, BrightFeature], expand: Boolean = true)(primaryImage: BufferedImage, secondaryImages: BufferedImage*): List[BufferedImage] = {
    val (pointsA, descriptionsA) = log.code(() ⇒ {
      describe(featureDetector, primaryImage)
    })
    def draw(primaryImage: BufferedImage, points: util.ArrayList[Point2D_F64], descriptions: FastQueue[BrightFeature]) = {
      log.draw(gfx ⇒ {
        gfx.drawImage(primaryImage, 0, 0, null)
        points.asScala.zip(descriptions.toList.asScala).foreach(x ⇒ {
          val (pt,d) = x
          if(d.white) {
            gfx.setColor(Color.GREEN)
          } else {
            gfx.setColor(Color.RED)
          }
          gfx.drawRect(pt.x.toInt - 4, pt.y.toInt - 4, 9, 9)
        })
      }, width = primaryImage.getWidth, height = primaryImage.getHeight())
    }
    draw(primaryImage, pointsA, descriptionsA)
    def findTransform(secondaryImage:BufferedImage) = {
      val (pointsB, descriptionsB) = log.code(() ⇒ {
        describe(featureDetector, secondaryImage)
      })
      draw(secondaryImage, pointsB, descriptionsB)
      val pairs: util.ArrayList[AssociatedPair] = associate(featureDetector, pointsA, descriptionsA, pointsB, descriptionsB)
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

  def describe(detDesc: DetectDescribePoint[GrayF32, BrightFeature], image: BufferedImage): (util.ArrayList[Point2D_F64], FastQueue[BrightFeature]) = {
    val points: util.ArrayList[Point2D_F64] = new util.ArrayList[Point2D_F64]()
    val input = ConvertBufferedImage.convertFromSingle(image, null, classOf[GrayF32])
    val descriptions: FastQueue[BrightFeature] = UtilFeature.createQueue(detDesc, 100)
    detDesc.detect(input)
    var i = 0
    while (i < detDesc.getNumberOfFeatures) {
      points.add(detDesc.getLocation(i).copy)
      descriptions.grow.setTo(detDesc.getDescription(i))
      i += 1;
    }
    (points, descriptions)
  }

  def associate(detDesc: DetectDescribePoint[GrayF32, BrightFeature], pointsA: util.ArrayList[Point2D_F64], descriptionsA: FastQueue[BrightFeature],
                pointsB: util.ArrayList[Point2D_F64], descriptionsB: FastQueue[BrightFeature]): util.ArrayList[AssociatedPair] = {
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

  def mix(a: Point2D_F32, b: Point2D_F32, d: Float): Point2D_F32 = {
    return new Point2D_F32(a.x * d + b.x * (1 - d), a.y * d + b.y * (1 - d))
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

  def scale(quad: Rectangle2D_F32, size: Float) = {
    val center = mix(quad.p0, quad.p1, 0.5f)
    val a = mix(quad.p0, center, size)
    val b = mix(quad.p1, center, size)
    new Rectangle2D_F32(
      a.x,a.y,b.x,b.y
    )
  }

  def draw(gfx : Graphics, quad: Quadrilateral_F32) = {
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
    if(r.a.x < center.x && r.a.y < center.y) r
    return new Quadrilateral_F32(r.d,r.c,r.b,r.a)
  }

}