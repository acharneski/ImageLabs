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
import java.awt.{Color, Graphics, Graphics2D, RenderingHints}
import java.util
import javax.imageio.ImageIO

import boofcv.abst.feature.detdesc.DetectDescribePoint
import boofcv.abst.feature.detect.interest.ConfigFastHessian
import boofcv.abst.feature.detect.line.{DetectLine, DetectLineSegment}
import boofcv.alg.descriptor.UtilFeature
import boofcv.alg.distort.impl.DistortSupport
import boofcv.alg.distort.{ImageDistort, PixelTransformHomography_F32}
import boofcv.core.image.border.BorderType
import boofcv.factory.feature.associate.FactoryAssociation
import boofcv.factory.feature.detdesc.FactoryDetectDescribe
import boofcv.factory.feature.detect.line.{ConfigHoughFoot, ConfigHoughFootSubimage, ConfigHoughPolar, FactoryDetectLineAlgs}
import boofcv.factory.geo.{ConfigRansac, FactoryMultiViewRobust}
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
        val width = 1200
        val height = image1.getHeight * width / image1.getWidth()
        val edgeThreshold: Float = 100
        val maxLines: Int = 20
        val localMaxRadius = 4
        val minCounts = 5
        val minDistanceFromOrigin = 1
        val resolutionAngle = Math.PI / (3 * 180)
        val totalHorizontalDivisions = 8
        val totalVerticalDivisions = 8

        def mix(a: Point2D_F32, b: Point2D_F32, d: Float): Point2D_F32 = {
          return new Point2D_F32(a.x * d + b.x * (1 - d), a.y * d + b.y * (1 - d))
        }
        def shrink(quad: Quadrilateral_F32, size: Float) = {
          val center = UtilPolygons2D_F32.center(quad, null)
          new Quadrilateral_F32(
            mix(quad.a, center, size),
            mix(quad.b, center, size),
            mix(quad.c, center, size),
            mix(quad.d, center, size)
          )
        }
        def draw(gfx : Graphics, quad: Quadrilateral_F32) = {
          gfx.drawPolygon(
            Array(
              quad.b.x.toInt * width / image1.getWidth,
              quad.a.x.toInt * width / image1.getWidth,
              quad.c.x.toInt * width / image1.getWidth,
              quad.d.x.toInt * width / image1.getWidth
            ),
            Array(
              quad.b.y.toInt * height / image1.getHeight,
              quad.a.y.toInt * height / image1.getHeight,
              quad.c.y.toInt * height / image1.getHeight,
              quad.d.y.toInt * height / image1.getHeight
            ), 4)
        }

        val detector: DetectLine[GrayU8] = log.code(() ⇒ {
          FactoryDetectLineAlgs.houghFoot(new ConfigHoughFoot(localMaxRadius, minCounts, minDistanceFromOrigin, edgeThreshold, maxLines), classOf[GrayU8], classOf[GrayS16])
        })
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

        def rotate(r: Quadrilateral_F32): Quadrilateral_F32 = {
          val center = UtilPolygons2D_F32.center(r, null)
          if(r.a.x < center.x && r.a.y < center.y) r
          return new Quadrilateral_F32(r.d,r.c,r.b,r.a)
        }
        val bestQuadrangle: Quadrilateral_F32 = shrink(rotate(log.code(() ⇒ {
          candidateQuadrangles.maxBy(quad ⇒ {
            val bounds = new Rectangle2D_F32()
            UtilPolygons2D_F32.bounding(quad, bounds)
            val area = quad.area()
            val squareness = area / bounds.area()
            assert(squareness >= 0 && squareness <= 1.01)
            area * Math.pow(squareness, 2)
          })
        })), 1.0f/0.9f)

        log.draw((gfx: Graphics) ⇒ {
          gfx.drawImage(image1, 0, 0, width, height, null)
          gfx.setColor(Color.YELLOW)
          draw(gfx, bestQuadrangle)
          gfx.setColor(Color.RED)
          draw(gfx, shrink(bestQuadrangle, 0.9f))
        }, width = width, height = height)


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
          val modelMatcher: ModelMatcher[Homography2D_F64, AssociatedPair] = FactoryMultiViewRobust.homographyRansac(null, new ConfigRansac(60, 3));
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

        val featureDetector: DetectDescribePoint[GrayF32, BrightFeature] = FactoryDetectDescribe.surfStable(new ConfigFastHessian(1, 2, 300, 1, 9, 4, 4), null, null, classOf[GrayF32])
        val (pointsA, descriptionsA) = log.code(() ⇒ {
          describe(featureDetector, primaryImage)
        })

        def rectify(primaryImage: BufferedImage, secondaryImages: BufferedImage*)(expand: Boolean = true): List[BufferedImage] = {
          val transforms: Map[BufferedImage, Homography2D_F64] = secondaryImages.map(secondaryImage ⇒ {
            val (pointsB, descriptionsB) = log.code(() ⇒ {
              describe(featureDetector, secondaryImage)
            })

            val pairs: util.ArrayList[AssociatedPair] = log.code(() ⇒ {
              associate(featureDetector, pointsA, descriptionsA, pointsB, descriptionsB)
            })

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

          List({
            val output = new BufferedImage(renderWidth.toInt, renderHeight.toInt, primaryImage.getType)
            output.getGraphics.drawImage(primaryImage, offsetX.toInt, offsetY.toInt, null)
            output
          }) ++ transforms.map(x ⇒ {
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

        val images: List[BufferedImage] = rectify(primaryImage
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

}