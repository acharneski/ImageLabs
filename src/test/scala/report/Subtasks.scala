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

import java.awt.{Color, Graphics}

import util.ReportNotebook
import java.util
import javax.imageio.ImageIO

import boofcv.abst.feature.detect.line.DetectLine
import boofcv.factory.feature.detect.line.{ConfigHoughFoot, ConfigHoughFootSubimage, ConfigHoughPolar, FactoryDetectLineAlgs}
import boofcv.io.image.ConvertBufferedImage
import boofcv.struct.image.{GrayS16, GrayU8}
import georegression.geometry.UtilPolygons2D_F32
import georegression.metric.Intersection2D_F32
import georegression.struct.homography.Homography2D_F64
import georegression.struct.line.LineParametric2D_F32
import georegression.struct.point.{Point2D_F32, Point2D_F64}
import georegression.struct.shapes.{Quadrilateral_F32, Rectangle2D_F32}
import georegression.transform.homography.HomographyPointOps_F64
import org.scalatest.{MustMatchers, WordSpec}

import scala.collection.JavaConverters._

class Subtasks extends WordSpec with MustMatchers with ReportNotebook {

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


  "Image Processing Subtasks" should {
    "Find Quadrangle" in {
      report("quadrangle", log ⇒ {
        val image1 = ImageIO.read(getClass.getClassLoader.getResourceAsStream("Whiteboard1.jpg"))
        val width = 1200
        val height = image1.getHeight * width / image1.getWidth()

        def fn(detector: DetectLine[GrayU8]) = {
          val found: util.List[LineParametric2D_F32] = detector.detect(ConvertBufferedImage.convertFromSingle(image1, null, classOf[GrayU8]))
          val horizontals = found.asScala.filter(line ⇒ Math.abs(line.slope.x) > Math.abs(line.slope.y)).toList
          val verticals = found.asScala.filter(line ⇒ Math.abs(line.slope.x) <= Math.abs(line.slope.y)).toList
          log.draw(gfx ⇒ {
            gfx.drawImage(image1, 0, 0, width, height, null)
            horizontals.foreach(line ⇒ {
              val x1 = 0
              val y1 = (line.p.y - line.p.x * line.slope.y / line.slope.x).toInt
              val x2 = image1.getWidth
              val y2 = y1 + (x2 * line.slope.y / line.slope.x).toInt
              gfx.setColor(Color.RED)
              gfx.drawLine(
                x1 * width / image1.getWidth, y1 * height / image1.getHeight,
                x2 * width / image1.getWidth, y2 * height / image1.getHeight)
            })
            verticals.foreach(line ⇒ {
              val y1 = 0
              val x1 = (line.p.x - line.p.y * line.slope.x / line.slope.y).toInt
              val y2 = image1.getHeight
              val x2 = x1 + (y2 * line.slope.x / line.slope.y).toInt
              gfx.setColor(Color.GREEN)
              gfx.drawLine(
                x1 * width / image1.getWidth, y1 * height / image1.getHeight,
                x2 * width / image1.getWidth, y2 * height / image1.getHeight)
            })
          }, width = width, height = height)

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

          val bestQuadrangle: Quadrilateral_F32 = log.code(() ⇒ {
            candidateQuadrangles.maxBy(quad ⇒ {
              val bounds = new Rectangle2D_F32()
              UtilPolygons2D_F32.bounding(quad, bounds)
              val area = quad.area()
              val squareness = area / bounds.area()
              assert(squareness >= 0 && squareness <= 1.01)
              area * Math.pow(squareness, 4)
            })
          })

          def draw(gfx: Graphics, quad: Quadrilateral_F32) = {
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

          log.draw((gfx: Graphics) ⇒ {
            gfx.drawImage(image1, 0, 0, width, height, null)
            gfx.setColor(Color.YELLOW)
            draw(gfx, bestQuadrangle)
            gfx.setColor(Color.RED)
            draw(gfx, shrink(bestQuadrangle, 0.9f))
          }, width = width, height = height)
        }

        val edgeThreshold: Float = 100
        val maxLines: Int = 20
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


  }

  def tranform(x0: Int, y0: Int, fromBtoWork: Homography2D_F64): (Double, Double) = {
    val result = new Point2D_F64
    HomographyPointOps_F64.transform(fromBtoWork, new Point2D_F64(x0, y0), result)
    val rx = result.x
    val ry = result.y
    (rx, ry)
  }


}